"""SimClock + CDM replay + the per-CDM pipeline (design §2.4).

Server-authoritative SIM clock: 1x = 3 sim-min per real second (180x), speeds 1/2/4/8. The clock is HELD while a
decision/LLM call runs and (optionally, bounded) while an approval waits. CDM k of an event is released at
tca_sim - time_to_tca; outcome.revealed fires at tca_sim. Every pipeline event is stamped with the CDM's own release
time, so the ledger content-hash sequence is deterministic for a given scenario + seed in mock/replay mode.
"""
from __future__ import annotations

import math

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import structlog

from app.approvals import Approval, ApprovalError, Approvals
from app.scenarios import card, load_catalogue, resolve
from app.util import days, dumps, fnum, hhmm, iso, parse_ts, uid
from decisions import expected_cost
from decisions import policy as pol
from llm.factsheet import build_facts, fmt_miss, fmt_pct, fmt_sci, fmt_ttca

log = structlog.get_logger("sim")

SIM_PER_REAL_S = 180.0  # 1x = 3 sim-min per real second
SPEEDS = (1, 2, 4, 8)
TIER_LEVEL = {"CRITICAL": "CRIT", "ELEVATED": "WARN", "SAFE": "INFO"}


@dataclass
class EventState:
    key: str
    role: str
    tca: datetime
    rows: list[dict]
    meta: dict
    released: list[dict] = field(default_factory=list)
    outcome: dict | None = None
    t2_forecast: dict | None = None
    last_forecast: dict | None = None


class SimBusy(Exception):
    pass


class Sim:
    def __init__(self, settings, bus, engine, decisions, explainer, ledger):
        self.s = settings
        self.bus = bus
        self.engine = engine
        self.dec = decisions
        self.exp = explainer
        self.ledger = ledger
        self.lock = asyncio.Lock()
        self.approvals = Approvals()
        self.policy = self.default_policy()
        self.policy_version = pol.policy_version(self.policy)
        self.catalogue: list[dict] = []
        self.catalogue_source = ""
        self.scenario: dict | None = None
        self.run_id = ""
        self.events: dict[str, EventState] = {}
        self.schedule: list[tuple] = []
        self.ptr = 0
        self.sim_time = parse_ts("2026-10-31T06:00:00Z")
        self.speed = 1
        self.paused = True
        self.holds: dict[str, float] = {}
        self.realtime = True
        self.logs: list[dict] = []
        self.counters = {"decisions": {"AUTO": 0, "ESCALATE": 0, "BLOCK": 0}, "ai_calls": 0, "fallbacks": 0,
                         "refusals": 0, "llm_verified": 0, "llm_rejected": 0, "degraded": 0}
        self.decisions_log: list[dict] = []
        self.outcomes: list[dict] = []
        self._preempt = False  # a scenario load is waiting: stop advancing, cancel the in-flight AI call
        self._ai_task: asyncio.Task | None = None
        bus.sim_time_fn = lambda: iso(self.sim_time)
        bus.snapshot_fn = self.state
        bus.ledger_refs_fn = self._refs

    # ------------------------------------------------------------------ helpers
    def _refs(self, type_: str, data: dict) -> dict:
        return {"policy_version": self.policy_version, "model": self.engine.forecaster_version,
                "run_id": self.run_id}

    def emit(self, type_, data, key=None, at=None, actor=None):
        return self.bus.publish(type_, data, event_key=key, sim_time=iso(at or self.sim_time), actor=actor)

    def logline(self, level: str, code: str, msg: str, key: str | None = None, at: datetime | None = None):
        at = at or self.sim_time
        entry = {"level": level, "code": code, "msg": msg, "t": hhmm(at)}
        env = self.bus.publish("log.entry", entry, event_key=key, sim_time=iso(at))
        self.logs.append({**entry, "event_key": key, "seq": env["seq"], "sim_time": iso(at)})
        if len(self.logs) > 400:
            self.logs = self.logs[-400:]

    def degraded(self, component: str, reason: str, fallback: str, key: str | None = None, at=None):
        self.counters["degraded"] += 1
        self.emit("system.degraded", {"component": component, "reason": reason, "fallback": fallback}, key, at)
        self.logline("WARN", "SYS", f"{component} {reason} → {fallback}", key, at)

    @property
    def held(self) -> bool:
        return bool(self.holds)

    def clock(self) -> dict:
        reason = None
        if self.holds:
            k = next(iter(self.holds))
            reason = "AI working" if k == "ai" else ("awaiting analyst" if k.startswith("approval:") else k)
        return {"sim_time": iso(self.sim_time), "speed": self.speed, "paused": self.paused, "held": self.held,
                "hold_reason": reason, "sim_per_real_s": SIM_PER_REAL_S * self.speed}

    # ------------------------------------------------------------------ scenarios
    def reload_catalogue(self) -> None:
        self.catalogue, self.catalogue_source = load_catalogue(self.s.scenarios_dir)

    def cards(self) -> list[dict]:
        return [card(s) for s in self.catalogue]

    async def load_scenario(self, sid: str, seed: int | None = None, *, wait_s: float = 15.0) -> dict:
        self.reload_catalogue()
        sc = next((s for s in self.catalogue if s["id"] == sid), None)
        if sc is None:
            raise KeyError(sid)
        # pre-empt the running pipeline: the realtime loop stops between CDMs and an in-flight AI call is cancelled
        self._preempt = True
        if self._ai_task is not None and not self._ai_task.done():
            self._ai_task.cancel()
        try:
            await asyncio.wait_for(self.lock.acquire(), timeout=wait_s)
        except asyncio.TimeoutError:
            self._preempt = False
            raise SimBusy("pipeline busy (AI call in flight); retry shortly")
        try:
            self._preempt = False
            await self._load(sc, seed)
        finally:
            self.lock.release()
        return self.state()

    def _prepare(self, sc: dict, seed: int) -> tuple[dict, list, list]:
        """CPU/IO part of a scenario load (runs in a worker thread): resolve the cast, fetch rows + meta once,
        build the release schedule, batch the live forecasts."""
        r = resolve(sc, self.engine, seed)
        evs, sched = [], []
        for c in r["cast"]:
            rows = self.engine.event_rows(c["event_key"])
            tca = parse_ts(c["tca_sim"])
            ev = EventState(c["event_key"], c.get("role", "focal"), tca, rows, self.engine.event_meta(c["event_key"]))
            evs.append(ev)
            for i, row in enumerate(rows):
                t = tca - days(fnum(row.get("time_to_tca")) or 0.0)
                if t > tca:
                    t = tca
                sched.append((t, 0, ev.key, i))
            sched.append((tca, 1, ev.key, -1))
        sched.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        self.engine._load_bakeoff()
        self.engine.prepare_forecasts([e.key for e in evs], (r.get("policy") or {}).get("tau_watch", self.policy["tau_watch"]))
        return r, evs, sched

    async def _load(self, sc: dict, seed: int | None) -> None:
        seed = int(seed if seed is not None else (sc.get("seed") or self.s.seed))
        r, evs, sched = await asyncio.to_thread(self._prepare, sc, seed)
        self.scenario = r
        self.run_id = uid(f"run:{r['id']}:{seed}")
        self.approvals.reset()
        self.holds = {}
        self.events = {}
        self.logs = []
        self.decisions_log = []
        self.outcomes = []
        self.counters = {"decisions": {"AUTO": 0, "ESCALATE": 0, "BLOCK": 0}, "ai_calls": 0, "fallbacks": 0,
                         "refusals": 0, "llm_verified": 0, "llm_rejected": 0, "degraded": 0}
        self.policy = dict(self.default_policy(), **(r.get("policy") or {}))
        self.policy_version = pol.policy_version(self.policy)
        self.sim_time = parse_ts(r["d0_sim"])
        self.paused = True
        self.events = {ev.key: ev for ev in evs}
        self.schedule = sched
        self.ptr = 0
        preload_until = parse_ts(r.get("preload_until_sim") or r["d0_sim"])
        n_pre = sum(1 for it in sched if it[1] == 0 and it[0] <= preload_until)
        self.emit("scenario.loaded", {"id": r["id"], "title": r.get("title"), "seed": seed, "d0": r["d0_sim"],
                                      "cast": [c["event_key"] for c in r["cast"]],
                                      "tca": {c["event_key"]: c["tca_sim"] for c in r["cast"]},
                                      "preloaded_cdms": n_pre, "missing_events": r.get("missing_events", []),
                                      "run_id": self.run_id}, at=self.sim_time)
        self.logline("INFO", "SCN", f"Scenario {r['id']} loaded · {len(r['cast'])} events · seed {seed}")
        for deg in self.engine.degraded:
            self.degraded(deg["component"], deg["reason"], deg["fallback"])
        # preload history silently (physics + rules + forecast only: no decisions, no LLM, no approvals), in one
        # ledger transaction, yielding to the event loop every few CDMs so the API stays responsive
        n = 0
        with self.ledger.transaction():
            while self.ptr < len(self.schedule) and self.schedule[self.ptr][0] <= preload_until:
                t, kind, key, idx = self.schedule[self.ptr]
                self.ptr += 1
                if kind == 0:
                    await self._cdm(self.events[key], idx, t, preload=True)
                    n += 1
                    if n % 25 == 0:
                        await asyncio.sleep(0)
        if preload_until > self.sim_time:
            self.sim_time = preload_until

    # ------------------------------------------------------------------ clock
    async def run(self) -> None:
        """Realtime loop: 4 Hz advance, 1 Hz clock.tick. Never dies."""
        last = time.monotonic()
        last_tick = 0.0
        period = 1.0 / max(self.s.tick_hz, 1.0)
        while True:
            try:
                await asyncio.sleep(period)
                now = time.monotonic()
                dt = now - last
                last = now
                self._expire_holds()
                if not self.paused and not self.held and not self.lock.locked():
                    target = self.sim_time + timedelta(seconds=dt * SIM_PER_REAL_S * self.speed)
                    async with self.lock:
                        await self._advance_to(target)
                if now - last_tick >= 1.0:
                    last_tick = now
                    self.bus.publish("clock.tick", self.clock(), sim_time=iso(self.sim_time))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # keep the loop alive
                log.exception("simloop_error", error=str(e))
                await asyncio.sleep(0.5)

    def _expire_holds(self) -> None:
        now = time.monotonic()
        for k, t0 in list(self.holds.items()):
            if k.startswith("approval:") and now - t0 > self.s.approval_hold_max_s:
                self.holds.pop(k, None)
                self.logline("WARN", "HUMAN", f"Approval {k.split(':', 1)[1][:8]} still pending · clock resumed after "
                                              f"{int(self.s.approval_hold_max_s)} s")

    async def _advance_to(self, target: datetime, *, ignore_holds: bool = False) -> None:
        while self.ptr < len(self.schedule) and self.schedule[self.ptr][0] <= target:
            if self._preempt:
                return
            if self.held and not ignore_holds:
                return
            t, kind, key, idx = self.schedule[self.ptr]
            self.ptr += 1
            self.sim_time = max(self.sim_time, t)
            ev = self.events[key]
            if kind == 0:
                await self._cdm(ev, idx, t)
            else:
                await self._outcome(ev, t)
            await asyncio.sleep(0)  # let API requests in between CDMs
        if self._preempt:
            return
        if not (self.held and not ignore_holds):
            self.sim_time = max(self.sim_time, target)

    async def next_cdm(self) -> dict:
        async with self.lock:
            nxt = next((it for it in self.schedule[self.ptr:] if it[1] == 0), None)
            # releasing approval holds: the analyst chose to move on
            for k in [k for k in self.holds if k.startswith("approval:")]:
                self.holds.pop(k, None)
            if nxt is None:
                if self.ptr < len(self.schedule):
                    await self._advance_to(self.schedule[-1][0], ignore_holds=True)
                return self.clock()
            await self._advance_to(nxt[0], ignore_holds=True)
        return self.clock()

    async def run_headless(self, until: datetime | None = None) -> None:
        """Process the whole schedule without wall-clock waits and without approval holds (tests, replay)."""
        self.realtime = False
        self.policy["hold_on_approval"] = False
        async with self.lock:
            await self._advance_to(until or (self.schedule[-1][0] if self.schedule else self.sim_time), ignore_holds=True)

    def set_speed(self, speed: int) -> dict:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of {list(SPEEDS)}")
        self.speed = speed
        self.bus.publish("clock.tick", self.clock(), sim_time=iso(self.sim_time))
        return self.clock()

    def set_paused(self, paused: bool) -> dict:
        self.paused = paused
        self.bus.publish("clock.tick", self.clock(), sim_time=iso(self.sim_time))
        return self.clock()

    # ------------------------------------------------------------------ pipeline
    async def _cdm(self, ev: EventState, idx: int, at: datetime, preload: bool = False) -> None:
        key = ev.key
        row = ev.rows[idx]
        cdm_id = uid(f"{key}:{idx}")
        n_known = idx + 1
        ttca = fnum(row.get("time_to_tca"))
        miss = fnum(row.get("miss_distance"))
        flags = []
        if (fnum(row.get("cov_unknown_t")) or 0) > 0:
            flags.append("COV_UNKNOWN_T")
        if (fnum(row.get("cov_unknown_c")) or 0) > 0:
            flags.append("COV_UNKNOWN_C")
        rtn = [fnum(row.get(f"relative_position_{a}")) for a in "rtn"]
        rec: dict = {"cdm_index": idx, "cdm_id": cdm_id, "created_sim": iso(at), "ttca_d": ttca, "miss_m": miss,
                     "preloaded": preload, "ledger_seqs": []}
        ev.released.append(rec)

        def put(type_, data, actor=None):
            env = self.emit(type_, data, key, at, actor)
            if env.get("ledger_seq"):
                rec["ledger_seqs"].append(env["ledger_seq"])
            return env

        split, eid = key.split(":", 1)
        # per-axis 1-sigma of target, chaser and combined (sqrt(t^2 + c^2), correlations ignored); default
        # covariances (>= 1e7 m, the 10-Earth-radii sentinel) are not sigmas and are sent as None
        sig_t = [fnum(row.get(f"t_sigma_{a}")) for a in "rtn"]
        sig_c = [fnum(row.get(f"c_sigma_{a}")) for a in "rtn"]
        sig_comb = [math.sqrt(a * a + b * b) if (a is not None and b is not None and a < 1e7 and b < 1e7) else None
                    for a, b in zip(sig_t, sig_c)]
        put("cdm.received", {"cdm_id": cdm_id, "cdm_index": idx, "n_known": n_known, "n_total": len(ev.rows),
                             "ttca_d": ttca, "created_sim": iso(at), "miss_m": miss,
                             "rel_speed_ms": fnum(row.get("relative_speed")), "rtn_m": rtn,
                             "sigma_t_m": sig_t, "sigma_c_m": sig_c, "sigma_comb_m": sig_comb,
                             "c_object_type": row.get("c_object_type"), "flags": flags, "preloaded": preload,
                             "source": {"split": split, "event_id": int(eid) if eid.isdigit() else eid, "row": idx}},
            actor="engine")
        if not preload:
            self.logline("CDM", "CDM", f"{key} CDM {n_known}/{len(ev.rows)} T−{fmt_ttca(ttca)} miss {fmt_miss(miss)} · "
                                       f"source {split} row {idx}", key, at)
        # physics + rules
        pc, pc_src = self.engine.pc_for(key, idx, row)
        tier, rules_src = self.engine.assess(row, pc, self.policy["tier_mode"])
        reported = fnum(row.get("risk")) if pc["status"] != "COV_UNKNOWN" else None
        disagree = abs(pc["log10"] - reported) if (pc.get("log10") is not None and reported is not None) else None
        rec.update(pc=pc, tier=tier, reported_log10=reported, row=row)
        put("risk.assessed", {"cdm_id": cdm_id, "cdm_index": idx, "pc": pc, "reported_log10": reported,
                              "disagree_dex": disagree, "tier": tier,
                              "source": {"pc": pc_src, "rules": rules_src}}, actor="engine")
        if not preload:
            if pc["status"] == "COV_UNKNOWN":
                self.logline("REFUSE", "PC", f"{key} covariance default (10 R⊕) · Pc not computable · CARA App. P.7", key, at)
            pc_txt = "n/a" if pc.get("log10") is None else fmt_sci(10 ** pc["log10"])
            self.logline(TIER_LEVEL[tier["tier"]], "TIER",
                         f"{key} {tier['tier']} · {'; '.join(tier['reasons'][:2]) or 'no rule fired'} · Pc tier "
                         f"{tier['pc_tier']} ({pc_txt})", key, at)
        # forecast (only ttca >= 2 d)
        fc = None
        if ttca is not None and ttca >= 2.0:
            annotated = [dict(r["row"], _tier=r["tier"]["tier"]) for r in ev.released]
            fc = self.engine.forecast(key, annotated, tier["tier"], self.policy["tau_watch"])
            if fc.get("valid"):
                # WATCH follows the served forecaster's own alert flag (model-card alert_threshold) unless the desk
                # moved τ_watch; ML can only add attention (never on a CRITICAL event, never lowers a tier)
                watch = pol.watch_flag(fc, tier["tier"], self.policy)
                fc["watch"] = watch
                fc["watch_threshold"], fc["watch_source"] = pol.watch_rule(fc, self.policy)
                ev.t2_forecast = fc
                ev.last_forecast = fc
                put("forecast.updated", {"cdm_id": cdm_id, "model_version": fc.get("model_version"), "valid": True,
                                         "model": fc.get("model"), "p_source": fc.get("p_source"), "note": fc.get("note"),
                                         "p_tiers": fc.get("p_tiers"), "p_critical": fc.get("p_critical"),
                                         "calib_bin": fc.get("calib_bin"), "baseline": fc.get("baseline"),
                                         "watch": watch, "rank": self._rank(key, fc),
                                         "watch_threshold": fc.get("watch_threshold"), "watch_source": fc.get("watch_source"),
                                         "fallback": bool(fc.get("fallback")),
                                         **{k: fc[k] for k in ("alert_threshold", "cost_ratio_R", "alert") if k in fc}},
                actor="ml")
                if not preload:
                    self.logline("WARN" if watch else "INFO", "FCST",
                                 f"{key} chance this ends CRITICAL {fmt_pct(fc.get('p_critical'))} · persistence "
                                 f"{(fc.get('baseline') or {}).get('persistence_tier')}" + (" · WATCH" if watch else ""),
                                 key, at)
        rec["forecast"] = fc
        g = pol.gate(tier, pc, ttca, fc, self.policy)
        rec["gate"] = g
        put("policy.gated", {"cdm_id": cdm_id, **{k: g[k] for k in ("allowed", "must_escalate", "blocked", "reasons",
                                                                    "flagged", "watch", "mcp_passed")}}, actor="rules")
        if preload:
            return
        # superseded approvals
        for a in self.approvals.expire_event(key, f"superseded by CDM {n_known}"):
            self.holds.pop(f"approval:{a.approval_id}", None)
        history = [{"ttca": r["ttca_d"], "miss_m": r["miss_m"],
                    "pc": None if r["pc"].get("log10") is None else 10 ** r["pc"]["log10"],
                    "pc_status": r["pc"]["status"]} for r in ev.released]
        facts = build_facts(event_key=key, row=row, cdm_index=idx, n_known=n_known, pc=pc, tier=tier, forecast=fc,
                            gate=g, history=history, policy=self.policy)
        if not g["flagged"]:
            v = pol.unflagged_verdict()
            rec["verdict"] = v
            self.counters["decisions"]["AUTO"] += 1
            put("policy.verdict", {"cdm_id": cdm_id, "decision_id": None, **v}, actor="rules")
            facts.update(verdict=v["verdict"], verdict_action=v["action"], verdict_reasons=v["reasons"],
                         verdict_primary_reasons=v.get("primary_reasons", []))
            tpl = self.exp.template(facts)
            put("explanation.ready", {"cdm_id": cdm_id, "source": "template", "verified": True,
                                      "headline": tpl["headline"], "text": tpl["text"]}, actor="llm")
            return
        # flagged: hold the clock while the AI layers work (as a task, so a scenario load can cancel it)
        self.holds["ai"] = time.monotonic()
        task = asyncio.ensure_future(self._decide(ev, idx, at, rec, facts, g, fc, tier, put))
        self._ai_task = task
        try:
            await asyncio.wait({task})
        except asyncio.CancelledError:
            task.cancel()
            raise
        finally:
            self._ai_task = None
            self.holds.pop("ai", None)
        if task.cancelled():
            log.info("ai_call_cancelled", event_key=key, cdm_index=idx)
            return
        task.result()

    def served_alert_threshold(self) -> float | None:
        """The served forecaster's alert threshold (model card; organizer model first), or None."""
        for name in getattr(self.engine, "PRED_SOURCES", ()):
            thr = ((getattr(self.engine, "pred_sources", {}) or {}).get(name) or {}).get("alert_threshold")
            if thr is not None:
                return float(thr)
        return None

    def default_policy(self) -> dict:
        """DEFAULT_POLICY with the desk's τ_watch = the served forecaster's alert threshold (falls back to 0.20)."""
        return pol.served_policy(self.served_alert_threshold())

    def _rank(self, key: str, fc: dict) -> int:
        p = fc.get("p_critical") or 0.0
        others = [e.last_forecast.get("p_critical") or 0.0 for k, e in self.events.items()
                  if k != key and e.last_forecast and e.outcome is None]
        return 1 + sum(1 for o in others if o > p)

    async def _decide(self, ev, idx, at, rec, facts, g, fc, tier, put) -> None:
        key, cdm_id = ev.key, rec["cdm_id"]
        decision_id = uid(f"{key}:{idx}:decision")
        base = expected_cost.baseline(fc, tier["tier"], g["allowed"], self.policy["loss"], ttca_d=rec["ttca_d"])
        prepared = self.dec.prepare(facts, g["allowed"], iso(at))
        put("decision.requested", {"cdm_id": cdm_id, "decision_id": decision_id, "provider_chain": self.dec.chain_names(),
                                   "request_hash": prepared["request_hash"], "allowed": g["allowed"],
                                   "mode": self.dec.mode}, actor="decisions")
        self.counters["ai_calls"] += 1
        d = await self.dec.decide(facts, g["allowed"], base, self.policy["loss"], int(self.scenario.get("seed", 0)), prepared)
        for deg in d.get("degraded", []):
            self.degraded(deg["component"], deg["reason"], deg["fallback"], key, at)
        if d["provider"] == "mock" and self.dec.mode != "mock":
            self.counters["fallbacks"] += 1
        if d.get("refused"):
            self.counters["refusals"] += 1
        agree = (d.get("answer") == base["action"]) if not d.get("refused") else None
        made = {"cdm_id": cdm_id, "decision_id": decision_id, "provider": d["provider"], "model": d.get("model"),
                "answer": d.get("answer"), "probabilities": d.get("probabilities"), "confidence": d.get("confidence"),
                "model_confidence": d.get("model_confidence"), "probability_kind": d.get("probability_kind"), "routed": d.get("routed"),
                "evidence_sufficient_p": d.get("evidence_sufficient_p"), "refused": bool(d.get("refused")),
                "latency_ms": d.get("latency_ms"), "cached": d.get("cached", False), "request_hash": d.get("request_hash"),
                "baseline": base, "agree": agree, "mode": self.dec.mode, "attempts": d.get("attempts", []),
                "note": "confidence = the decision model's probability for its chosen action (not a collision "
                        "probability); model_confidence = the provider's own confidence field"}
        put("decision.made", made, actor=f"decisions:{d['provider']}")
        rec["decision"] = made
        if d.get("refused"):
            self.logline("REFUSE", "DEC", f"{key} {d['provider']} declined · escalating", key, at)
        else:
            self.logline("AGENT", "DEC", f"{key} {d['provider']} {pol.SHORT.get(d['answer'], d['answer'])} "
                                         f"{fmt_pct(d.get('confidence'))} · cost rule {pol.SHORT.get(base['action'])} · "
                                         f"{'agree' if agree else 'disagree'}", key, at)
        v = pol.verdict(g, d, base, self.policy, mock=d["provider"] == "mock")
        rec["verdict"] = v
        self.counters["decisions"][v["verdict"]] += 1
        put("policy.verdict", {"cdm_id": cdm_id, "decision_id": decision_id, **v}, actor="rules")
        lvl = {"AUTO": "INFO", "ESCALATE": "WARN", "BLOCK": "REFUSE"}[v["verdict"]]
        self.logline(lvl, "POLICY", f"{key} {v['verdict']} · {'; '.join(v.get('primary_reasons') or v['reasons'][:2])}"
                                    + (" · awaiting analyst" if v["verdict"] != "AUTO" else ""), key, at)
        self.decisions_log.append({"event_key": key, "cdm_index": idx, "sim_time": iso(at), "verdict": v["verdict"],
                                   "action": v["action"], "provider": d["provider"], "answer": d.get("answer"),
                                   "confidence": d.get("confidence"), "baseline": base["action"],
                                   "reasons": v["reasons"]})
        if v["verdict"] != "AUTO":
            ttca_h = None if rec["ttca_d"] is None else round(rec["ttca_d"] * 24, 2)
            a = self.approvals.add(Approval(approval_id=uid(f"{key}:{idx}:approval"), decision_id=decision_id,
                                            event_key=key, cdm_index=idx, options=list(g["allowed"]),
                                            suggested=v["action"], verdict=v["verdict"], reasons=v["reasons"],
                                            ttca_h=ttca_h, mcp_passed=g["mcp_passed"], created_sim=iso(at),
                                            primary_reasons=v.get("primary_reasons", [])))
            rec["approval_id"] = a.approval_id
            put("approval.required", {"approval_id": a.approval_id, "decision_id": decision_id, "cdm_id": cdm_id,
                                      "options": a.options, "suggested": a.suggested, "verdict": v["verdict"],
                                      "ttca_h": ttca_h, "mcp_passed": a.mcp_passed, "reasons": v["reasons"],
                                      "primary_reasons": v.get("primary_reasons", []),
                                      "tca_sim": iso(ev.tca)}, actor="system")
            try:
                self.ledger.approval_put(a.approval_id, self.run_id, key, a.status, dumps(a.to_dict()))
            except Exception:
                pass
            if self.policy.get("hold_on_approval") and self.realtime:
                self.holds[f"approval:{a.approval_id}"] = time.monotonic()
        # explanation: template first (always), then the LLM if it verifies
        facts.update(decision_provider=d["provider"], decision_answer=d.get("answer"),
                     decision_confidence=d.get("confidence"), evidence_sufficient_p=d.get("evidence_sufficient_p"),
                     baseline_action=base["action"], verdict=v["verdict"], verdict_action=v["action"],
                     verdict_reasons=v["reasons"], verdict_primary_reasons=v.get("primary_reasons", []),
                     baseline_track_too_late=bool(base.get("track_note")))
        tpl = self.exp.template(facts)
        put("explanation.ready", {"cdm_id": cdm_id, "source": "template", "verified": True,
                                  "headline": tpl["headline"], "text": tpl["text"]}, actor="llm")
        rec["explanation"] = {"source": "template", "text": tpl["text"]}
        if self.exp.client is not None or self.s.llm_mode in ("live", "replay"):
            res = await self.exp.llm(facts)
            if res["status"] == "ok":
                self.counters["llm_verified"] += 1
                put("explanation.ready", {"cdm_id": cdm_id, "source": "llm", "verified": True,
                                          "headline": res.get("headline"), "text": res["text"],
                                          "prompt_hash": res.get("prompt_hash")}, actor="llm")
                rec["explanation"] = {"source": "llm", "text": res["text"]}
                self.logline("AGENT", "EXPLAIN", f"{key} explanation AI-written · verified", key, at)
            elif res["status"] in ("rejected", "error"):
                self.counters["llm_rejected"] += res["status"] == "rejected"
                try:
                    self.ledger.append("llm", {"type": "llm.rejected", "data": {"cdm_id": cdm_id, "reason": res.get("reason")}},
                                       event_key=key, actor="llm", sim_time=iso(at), type_="llm.rejected")
                except Exception:
                    pass
                if res["status"] == "error":
                    self.degraded("llm", res.get("reason", "error"), "template", key, at)
                else:
                    self.logline("WARN", "EXPLAIN", f"{key} LLM text rejected ({res.get('reason')}) · template kept", key, at)

    async def _outcome(self, ev: EventState, at: datetime) -> None:
        key = ev.key
        for a in self.approvals.expire_event(key, "TCA reached"):
            self.holds.pop(f"approval:{a.approval_id}", None)
        final = self.engine.final_outcome(key)
        fw = ev.t2_forecast
        forecast_was = None
        verdict = pers_v = model_v = None
        if fw:
            pers = (fw.get("baseline") or {}).get("persistence_tier")
            p = fw.get("p_critical")
            forecast_was = {"p_critical": p, "persistence_tier": pers, "watch": fw.get("watch"),
                            "model_version": fw.get("model_version")}
        if final and fw:
            crit = final["tier"] == "CRITICAL"

            def score(flag):
                return ("TP" if flag else "FN") if crit else ("FP" if flag else "TN")
            pers_flag = forecast_was["persistence_tier"] == "CRITICAL"
            model_flag = (forecast_was["p_critical"] or 0) >= (fw.get("watch_threshold") or self.policy["tau_watch"])
            pers_v, model_v = score(pers_flag), score(model_flag)
            verdict = score(pers_flag or bool(forecast_was["watch"]))
        data = {"final": None if final is None else {"tier": final["tier"], "log10_pc": final["log10_pc"],
                                                     "miss_m": final["miss_m"]},
                "hidden": final is None, "forecast_was": forecast_was, "verdict": verdict,
                "persistence_verdict": pers_v, "model_verdict": model_v,
                "note": None if fw else "no CDM at ≥ 2 d before TCA: nothing to score"}
        ev.outcome = data
        self.outcomes.append({"event_key": key, **data})
        self.emit("outcome.revealed", data, key, at, actor="system")
        if final is None:
            self.logline("INFO", "OUTCOME", f"{key} TCA reached · truth hidden (blind test)", key, at)
        else:
            self.logline("CRIT" if final["tier"] == "CRITICAL" else "INFO", "OUTCOME",
                         f"{key} TCA · final {final['tier']} miss {fmt_miss(final['miss_m'])}"
                         + (f" · T−2 d call {verdict} (persistence {pers_v}, model {model_v})" if verdict else ""), key, at)

    # ------------------------------------------------------------------ human + policy
    def decide_approval(self, approval_id: str, action: str, note: str | None, actor: str = "analyst") -> dict:
        a0 = self.approvals.items.get(approval_id)
        wait = 0.0
        if a0 is not None:
            wait = (self.sim_time - parse_ts(a0.created_sim)).total_seconds() / 60
        a = self.approvals.decide(approval_id, action, actor, note, iso(self.sim_time), round(wait, 1))
        self.holds.pop(f"approval:{approval_id}", None)
        self.emit("human.action", {"approval_id": approval_id, "decision_id": a.decision_id, "action": action,
                                   "actor": actor, "note": note, "wait_sim_min": round(wait, 1)}, a.event_key,
                  actor=actor)
        self.logline("INFO", "HUMAN", f"{a.event_key} analyst chose {pol.SHORT.get(action, action)}"
                                      + (f" · “{note}”" if note else ""), a.event_key)
        try:
            self.ledger.approval_put(a.approval_id, self.run_id, a.event_key, a.status, dumps(a.to_dict()))
        except Exception:
            pass
        return a.to_dict()

    def update_policy(self, patch: dict, actor: str = "analyst") -> dict:
        new, diff = pol.clean_patch(patch, self.policy)
        mode_changed = new["tier_mode"] != self.policy["tier_mode"]
        self.policy = new
        self.policy_version = pol.policy_version(new)
        try:
            self.ledger.policy_put(self.policy_version, dumps(new), actor)
        except Exception:
            pass
        if diff:
            self.emit("policy.updated", {"version": self.policy_version, "diff": diff, "actor": actor,
                                         "policy": new}, actor=actor)
            self.logline("INFO", "POLICY", "desk: " + ", ".join(f"{k} {v['from']} → {v['to']}" for k, v in diff.items()
                                                                if k != "loss") or "loss table updated")
        if mode_changed:
            self._rerun_rules()
        return {"policy": new, "version": self.policy_version, "diff": diff}

    def _rerun_rules(self) -> int:
        """Tier mode changed: rules re-run server-side on every released CDM of the current scenario; one
        risk.assessed (rerun=True) per event carries its latest CDM's new tier so the markers re-colour."""
        mode = self.policy["tier_mode"]
        n_changed = 0
        for ev in self.events.values():
            last = None
            for rec in ev.released:
                if "pc" not in rec or "row" not in rec:
                    continue
                old = (rec.get("tier") or {}).get("tier")
                tier, src = self.engine.assess(rec["row"], rec["pc"], mode)
                rec["tier"] = tier
                rec["rules_src"] = src
                last = rec
                if rec is ev.released[-1] and old != tier["tier"]:
                    n_changed += 1
            if last is not None:
                pc = last["pc"]
                reported = last.get("reported_log10")
                disagree = abs(pc["log10"] - reported) if (pc.get("log10") is not None and reported is not None) else None
                self.emit("risk.assessed", {"cdm_id": last["cdm_id"], "cdm_index": last["cdm_index"], "pc": pc,
                                            "reported_log10": reported, "disagree_dex": disagree, "tier": last["tier"],
                                            "rerun": True, "tier_mode": mode,
                                            "source": {"pc": "unchanged", "rules": last.get("rules_src")}},
                          ev.key, actor="engine")
        self.logline("INFO", "TIER", f"Tier mode {mode}: rules re-run on {sum(len(e.released) for e in self.events.values())} "
                                     f"released CDMs · {n_changed} event tier{'s' if n_changed != 1 else ''} changed")
        return n_changed

    # ------------------------------------------------------------------ views
    def event_summary(self, ev: EventState) -> dict:
        latest = ev.released[-1] if ev.released else None
        out = {"event_key": ev.key, "role": ev.role, "tca_sim": iso(ev.tca), "n_cdm_total": len(ev.rows),
               "n_released": len(ev.released), "mission_id": ev.meta.get("mission_id"),
               "c_object_type": ev.meta.get("c_object_type"), "outcome": ev.outcome, "latest": None,
               "forecast": None, "verdict": None, "approval_id": None,
               "mcp_sim": iso(ev.tca - timedelta(hours=24)), "ttca_now_h": round((ev.tca - self.sim_time).total_seconds() / 3600, 2)}
        if latest and "tier" in latest:
            pc = latest["pc"]
            out["latest"] = {"cdm_index": latest["cdm_index"], "cdm_id": latest["cdm_id"], "ttca_d": latest["ttca_d"],
                             "miss_m": latest["miss_m"], "log10_pc": pc.get("log10"), "pc_status": pc["status"],
                             "tier": latest["tier"]["tier"], "pc_tier": latest["tier"]["pc_tier"],
                             "miss_tier": latest["tier"]["miss_tier"], "rule_ids": latest["tier"]["rule_ids"],
                             "created_sim": latest["created_sim"]}
            out["verdict"] = (latest.get("verdict") or {}).get("verdict")
            out["approval_id"] = latest.get("approval_id")
        if ev.last_forecast:
            out["forecast"] = {"p_critical": ev.last_forecast.get("p_critical"), "watch": ev.last_forecast.get("watch"),
                               "persistence_tier": (ev.last_forecast.get("baseline") or {}).get("persistence_tier")}
        return out

    def kpis(self) -> dict:
        active = [e for e in self.events.values() if e.released and e.outcome is None and "tier" in e.released[-1]]
        now = self.sim_time

        def within(e, km):
            dt = (e.tca - now).total_seconds() / 3600
            m = e.released[-1]["miss_m"]
            return 0 <= dt <= 24 and m is not None and m < km * 1000
        crit = [e for e in active if e.released[-1]["tier"]["tier"] == "CRITICAL"]
        pend = self.approvals.pending()
        oldest = None
        if pend:
            oldest = round(max((now - parse_ts(a.created_sim)).total_seconds() / 60 for a in pend), 1)
        sb = {"persistence": {"TP": 0, "FP": 0, "FN": 0, "TN": 0}, "model": {"TP": 0, "FP": 0, "FN": 0, "TN": 0}}
        for o in self.outcomes:
            if o.get("persistence_verdict"):
                sb["persistence"][o["persistence_verdict"]] += 1
            if o.get("model_verdict"):
                sb["model"][o["model_verdict"]] += 1
        refused = sum(1 for e in active if e.released[-1]["pc"]["status"] == "COV_UNKNOWN")
        return {"within_1km_24h": sum(within(e, 1) for e in active), "within_5km_24h": sum(within(e, 5) for e in active),
                "critical": len(crit),
                "critical_miss_only": sum(1 for e in crit if e.released[-1]["tier"]["pc_tier"] != "CRITICAL"),
                "critical_by_pc": sum(1 for e in crit if e.released[-1]["tier"]["pc_tier"] == "CRITICAL"),
                "elevated": sum(1 for e in active if e.released[-1]["tier"]["tier"] == "ELEVATED"),
                "watch": sum(1 for e in active if e.last_forecast and e.last_forecast.get("watch")),
                "refused": refused + self.counters["refusals"], "awaiting": len(pend), "awaiting_oldest_sim_min": oldest,
                "decisions": dict(self.counters["decisions"]), "ai_calls": self.counters["ai_calls"],
                "fallbacks": self.counters["fallbacks"], "llm_verified": self.counters["llm_verified"],
                "llm_rejected": self.counters["llm_rejected"], "scoreboard": sb}

    def mode(self) -> dict:
        return {"decisions": self.dec.mode, "llm": self.s.llm_mode, "engine": self.engine.data_source,
                "orbits": "reconstructed", "scenarios": self.catalogue_source,
                "decisions_badge": {"mock": "DECISIONS: MOCK (cost rule)", "replay": "DECISIONS: REPLAY (cache)"}.get(
                    self.dec.mode, "DECISIONS: LIVE " + " → ".join(self.dec.chain_names()))}

    def state(self) -> dict:
        sc = self.scenario or {}
        return {"v": 1, "clock": self.clock(),
                "scenario": None if not sc else {"id": sc.get("id"), "title": sc.get("title"), "tag": sc.get("tag"),
                                                  "teaches": sc.get("teaches"), "seed": sc.get("seed"),
                                                  "d0": sc.get("d0_sim"), "run_id": self.run_id,
                                                  "done": self.ptr >= len(self.schedule)},
                "cast": [self.event_summary(e) for e in self.events.values()],
                "approvals_open": [a.to_dict() for a in self.approvals.pending()],
                "policy": self.policy, "policy_version": self.policy_version,
                "kpis": self.kpis(), "mode": self.mode(),
                "ledger": {"seq": self.ledger.seq, "verified": self.ledger.verified},
                "log": self.logs[-100:]}
