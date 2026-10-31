"""-> shared/scenarios/<id>.json (the 9 scenarios of design 7.6 / CONTRACT.md).

SIM timing: CDM k of a cast event is released at tca_sim - time_to_tca_k (days). CDMs released before
preload_until_sim are preloaded history. Outcome revealed at tca_sim. All times are SIM (synthetic).
"""
try:
    from . import _boot  # noqa: F401
except ImportError:
    import _boot  # noqa: F401
import datetime as dt
import json

import numpy as np
import pandas as pd

from riskengine import paths

D0 = dt.datetime(2026, 10, 31, 6, 0, tzinfo=dt.timezone.utc)
SEED = 1729


def iso(t: dt.datetime) -> str:
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def at(days: float) -> str:
    return iso(D0 + dt.timedelta(days=float(days)))


def member(key, role, tca_days, **extra):
    m = {"event_key": key, "role": role, "tca_sim": at(tca_days)}
    m.update(extra)
    return m


def scenario(id_, title, tag, teaches, order, cast, **extra):
    s = {"id": id_, "title": title, "tag": tag, "teaches": teaches, "order": order, "d0_sim": iso(D0), "seed": SEED,
         "cast": cast, "preload_until_sim": iso(D0), "window_h": 24}
    s.update(extra)
    return s


def pick_missed_critical(oof: pd.DataFrame) -> str:
    """Final-critical, persistence not critical, lowest OOF (by event) P(crit); prefer Pc-critical-at-end cases."""
    m = oof[(oof.y_tier == "CRITICAL") & (oof.persistence_tier != "CRITICAL")].copy()
    m["pc_crit_final"] = m.y_log10_risk >= -4
    m = m.sort_values(["pc_crit_final", "p_critical_event"], ascending=[False, True])
    m = m[m.p_critical_event < 0.05]
    return m.event_key.iloc[0]


def main():
    ev = pd.read_parquet(paths.EVENTS).set_index("event_key")
    oof = pd.read_parquet(paths.current_forecaster_dir() / "oof.parquet")
    rng = np.random.default_rng(SEED)
    out = []

    # 1 ops day, mission 15
    stories = {"train:2563": 14.5 / 24, "train:8767": 18 / 24, "train:2115": 1.6}
    pool = ev[(ev.mission_id == 15) & (ev.split == "train") & ev.eligible & (ev.n_cdm >= 3)
              & ~ev.index.isin(list(stories))]
    elev = pool[pool.final_tier == "ELEVATED"]
    safe = pool[pool.final_tier == "SAFE"]
    elev_k = rng.choice(elev.index.to_numpy(), 8, replace=False)
    safe_k = rng.choice(safe.index.to_numpy(), 22, replace=False)
    cast = [member(k, "story", t) for k, t in stories.items()]
    elev_t = np.sort(rng.uniform(3 / 24, 30 / 24, len(elev_k)))
    cast += [member(k, "elevated", t) for k, t in zip(elev_k, elev_t)]
    safe_t = np.concatenate([rng.uniform(2 / 24, 1.5, 11), rng.uniform(1.5, 6.5, 11)])
    cast += [member(k, "routine", t) for k, t in zip(safe_k, safe_t)]
    cast.sort(key=lambda m: m["tca_sim"])
    out.append(scenario("ops-day", "Ops day, mission 15", "TRIAGE",
                        "A 24 h duty shift: ~33 live conjunctions for one satellite, a CDM drop every ~8 h per event. "
                        "Rank the queue, watch the 24 h danger window, approve or reject suggested actions.", 1, cast,
                        operator={"mission_id": 15}, focus="train:2563"))

    # 2 late CDM
    out.append(scenario("late-cdm", "The late CDM", "LATE",
                        "Safe at the 2-day decision point, critical by miss 10.6 h before TCA: nothing at T-2 d foresaw it "
                        "(10 of 6,040 safe-at-T-2 d events). Physics and rules catch it after the commitment point.", 2,
                        [member("train:2563", "focal", 2.45)], focus="train:2563", alternates=["train:5253"]))
    # 3 steady climb
    out.append(scenario("steady-climb", "Steady climb", "ESCALATE",
                        "Pc climbs from 1e-9 to 2e-3 while the miss shrinks to 40 m: the full pipeline, an early "
                        "PREPARE suggestion and the dive.", 3, [member("train:8767", "focal", 3.7)], focus="train:8767"))
    # 4 scary then nothing
    out.append(scenario("scary-then-nothing", "Scary, then nothing", "WAIT",
                        "The highest Pc in the training set (3.6e-2 at T-5.2 d) collapses to nothing by T-2.8 d. "
                        "Commit early or wait for the next CDM?", 4,
                        [member("train:2427", "focal", 5.3), member("train:7465", "contrast", 2.3)], focus="train:2427"))
    # 5 rules disagree
    out.append(scenario("rules-disagree", "Rules disagree", "RULES",
                        "Miss 76 m but Pc 6e-19 (critical by the miss screen, safe by probability) next to Pc 1.2e-4 at "
                        "3.9 km (critical by Pc only). Flip the tier-mode knob.", 5,
                        [member("train:8756", "focal", 1.0), member("train:9361", "contrast", 1.2)], focus="train:8756"))
    # 6 covariance unknown
    out.append(scenario("cov-unknown", "Covariance unknown", "REFUSAL",
                        "13 CDMs carry a default covariance (10 Earth radii): the system refuses to compute a Pc, "
                        "floors the tier at ELEVATED and blocks automation until a real covariance arrives.", 6,
                        [member("train:2299", "focal", 2.0), member("train:2759", "contrast", 1.5)], focus="train:2299"))
    # 7 failure
    missed = pick_missed_critical(oof)
    out.append(scenario("failure", "Where we fail", "FAILURE CASE",
                        "train:4200: ESA reports Pc 1e-5 at 57 km, our physics says 1e-478 (we cannot reproduce ESA's "
                        f"covariance scaling). {missed}: the forecaster gave it under 1 % at T-2 d and it ended CRITICAL. "
                        "How the analyst knows, and why the ML never suppresses a rule tier.", 7,
                        [member("train:4200", "focal", 6.95), member(missed, "missed_critical", 3.1)],
                        focus="train:4200", missed_critical=missed))
    # 8 blind test
    blind = ["test:891", "test:1568", "test:1448", "test:2059", "test:2091"]
    cast = [member(k, "blind", float(ev.loc[k, "last_ttca"]) + (2 + 2 * i) / 24) for i, k in enumerate(blind)]
    out.append(scenario("blind-test", "Blind test", "BLIND",
                        "Five unseen test events with the highest latest reported risk. Truth is hidden: forecasts only.",
                        8, cast, truth_hidden=True, focus="test:891"))
    # 9 free play
    fp = ev[(ev.split == "train") & ev.eligible & (ev.n_cdm >= 3)]
    keys = rng.choice(fp.index.to_numpy(), 12, replace=False)
    cast = sorted([member(k, "random", t) for k, t in zip(keys, rng.uniform(3 / 24, 3.0, 12))], key=lambda m: m["tca_sim"])
    out.append(scenario("free-play", "Free play", "FREE",
                        "A seeded random cast from any mission, plus your own CDMs via POST /api/assess.", 9, cast,
                        generator={"kind": "random_train_eligible", "n": 12, "seed": SEED}))

    paths.SCENARIOS.mkdir(parents=True, exist_ok=True)
    for s in out:
        for m in s["cast"]:
            assert m["event_key"] in ev.index, m
        (paths.SCENARIOS / f"{s['id']}.json").write_text(json.dumps(s, indent=1) + "\n")
    print(f"[scenarios] wrote {len(out)} scenarios -> {paths.SCENARIOS}; failure missed critical = {missed}")


if __name__ == "__main__":
    main()
