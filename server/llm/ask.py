"""J3 Ask the desk: map a question to ONE whitelisted query, run it server-side, answer from its result only.

Whitelist: events_within(km, hours) · events_by_rule(rule|tier) · event_history(event_key) · ledger_for(event_key).
Routing is deterministic (keyword rules); anything else is refused. Decision/approval questions are refused with
the policy verdict, and hypothetical numbers ("Pc if we burn…") are refused because they are not in the data.
"""
from __future__ import annotations

import re
from datetime import timedelta

from llm.factsheet import fmt_miss, fmt_sci, fmt_ttca

OUT_OF_SCOPE = "I can only answer from the CDM data and the ledger."
KEY = re.compile(r"\b(train|test|adhoc)\s*[:#]?\s*(\d+)\b", re.I)


def _latest(ev):
    return ev.released[-1] if ev.released and "tier" in ev.released[-1] else None


def _pc(rec):
    lp = rec["pc"].get("log10")
    return "not computable (default covariance)" if lp is None else fmt_sci(10 ** lp)


async def answer(c, q: str) -> dict:
    sim = c.sim
    low = q.lower()
    m = KEY.search(q)
    key = f"{m.group(1).lower()}:{m.group(2)}" if m else None
    # 1) decisions belong to the analyst
    if re.search(r"\b(should (we|i)|do we need to|approve|go/no-go|burn|manoeuvre now|maneuver now|dismiss)\b", low):
        if re.search(r"\bif we\b|\bafter (a|the) burn\b|cm/s|m/s", low):
            return _refuse("That needs a number that is not in the CDM data (a hypothetical manoeuvre); "
                           "flight dynamics computes it.", "hypothetical")
        ev = sim.events.get(key) if key else None
        if ev and _latest(ev) and _latest(ev).get("verdict"):
            v = _latest(ev)["verdict"]
            return _refuse(f"That decision is yours; the policy verdict for {ev.key} is {v['verdict']} because "
                           f"{'; '.join(v['reasons'])}.", "decision", query={"name": "event_history", "args": {"event_key": ev.key}})
        return _refuse("That decision is yours; the system only gates and explains.", "decision")
    if re.search(r"\b(what if|if we|would be|predict the pc|after (a|the) burn)\b", low):
        return _refuse("That needs a number that is not in the CDM data.", "hypothetical")
    # 2) ledger_for
    if "ledger" in low or "audit" in low:
        if not key:
            return _refuse("Name an event (e.g. train:2563) to see its ledger rows.", "need_key")
        rows = c.ledger.entries(after=0, limit=50, event_key=key)
        lines = [f"#{r['seq']} {r['sim_time'][11:16]} {r['payload'].get('type')}" for r in rows[-12:]]
        return _ok(f"{len(rows)} ledger rows for {key}" + (": " + " · ".join(lines) if lines else "."),
                   {"name": "ledger_for", "args": {"event_key": key}}, {"seqs": [r["seq"] for r in rows]})
    # 3) event_history
    if key:
        ev = sim.events.get(key)
        if ev is None or not ev.released:
            return _refuse(f"{key} has no released CDMs in this scenario.", "unknown_event")
        if "covariance" in low and _latest(ev)["pc"]["status"] == "COV_UNKNOWN" and "pc" in low:
            return _refuse("The covariance is a default, so no Pc can be given for this CDM.", "cov_unknown")
        parts = [f"CDM {r['cdm_index'] + 1} T−{fmt_ttca(r['ttca_d'])}: miss {fmt_miss(r['miss_m'])}, Pc {_pc(r)}, "
                 f"{r['tier']['tier']}" for r in ev.released if "tier" in r]
        lv = _latest(ev).get("verdict")
        tail = f" Latest policy verdict: {lv['verdict']} — {'; '.join(lv['reasons'])}." if lv else ""
        return _ok(f"{key}: " + " | ".join(parts) + "." + tail, {"name": "event_history", "args": {"event_key": key}})
    # 4) events_within
    mk = re.search(r"(\d+(?:\.\d+)?)\s*km", low)
    mh = re.search(r"(\d+)\s*h(?:ours?)?\b", low)
    if mk or "within" in low or "dangerous distance" in low or "close" in low:
        km = float(mk.group(1)) if mk else 1.0
        hours = int(mh.group(1)) if mh else 24
        hits = []
        for ev in sim.events.values():
            r = _latest(ev)
            if r and ev.outcome is None and timedelta(0) <= ev.tca - sim.sim_time <= timedelta(hours=hours) \
                    and r["miss_m"] is not None and r["miss_m"] < km * 1000:
                hits.append(f"{ev.key} (miss {fmt_miss(r['miss_m'])}, TCA in {fmt_ttca((ev.tca - sim.sim_time).total_seconds() / 86400)})")
        return _ok(f"{len(hits)} event(s) within {km:g} km in the next {hours} h" + (": " + "; ".join(hits) if hits else "."),
                   {"name": "events_within", "args": {"km": km, "hours": hours}})
    # 5) events_by_rule
    rule = None
    rm = re.search(r"\b([RQT]-[A-Z-]+)\b", q.upper())
    if rm:
        rule = rm.group(1)
    elif "covariance" in low:
        rule = "Q-COV-DEFAULT"
    elif "disagree" in low:
        rule = "Q-PHYS-DISAGREE"
    tier = next((t for t in ("CRITICAL", "ELEVATED", "SAFE") if t.lower() in low), None)
    if "watch" in low:
        tier = "WATCH"
    if rule or tier:
        hits = []
        for ev in sim.events.values():
            r = _latest(ev)
            if not r:
                continue
            if rule and rule in r["tier"]["rule_ids"]:
                hits.append(ev.key)
            elif tier == "WATCH" and ev.last_forecast and ev.last_forecast.get("watch"):
                hits.append(ev.key)
            elif tier and tier == r["tier"]["tier"]:
                hits.append(ev.key)
        what = rule or tier
        return _ok(f"{len(hits)} event(s) match {what} on their latest CDM" + (": " + ", ".join(hits) if hits else "."),
                   {"name": "events_by_rule", "args": {"rule": rule, "tier": tier}})
    return _refuse(OUT_OF_SCOPE, "out_of_scope")


def _ok(text: str, query: dict, extra: dict | None = None) -> dict:
    return {"answer": text, "refusal": None, "query": query, "source": "template", "verified": True, **(extra or {})}


def _refuse(reason: str, kind: str, query: dict | None = None) -> dict:
    return {"answer": None, "refusal": {"reason": reason, "kind": kind}, "query": query, "source": "template",
            "verified": True}
