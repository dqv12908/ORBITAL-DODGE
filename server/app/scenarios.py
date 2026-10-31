"""Scenario catalogue: shared/scenarios/*.json (engine-built) with a built-in fallback so the server always runs."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import structlog

from app.util import days, iso, parse_ts

log = structlog.get_logger("scenarios")

D0 = "2026-10-31T06:00:00Z"

BUILTIN = [
    {"id": "ops-day", "title": "Ops day, mission 15", "tag": "TRIAGE", "order": 1,
     "teaches": "24 h danger window, queue ranking, approvals",
     "cast": [{"event_key": "train:8767", "role": "focal"}, {"event_key": "train:2563", "role": "focal"},
              {"event_key": "train:2115", "role": "focal"}]},
    {"id": "late-cdm", "title": "The late CDM", "tag": "LATE", "order": 2,
     "teaches": "Nothing at T−2 d foresaw it; physics catches it after the commitment point",
     "cast": [{"event_key": "train:2563", "role": "focal"}]},
    {"id": "steady-climb", "title": "Steady climb", "tag": "ESCALATE", "order": 3,
     "teaches": "The full pipeline, early PREPARE, the dive", "cast": [{"event_key": "train:8767", "role": "focal"}]},
    {"id": "scary-then-nothing", "title": "Scary, then nothing", "tag": "WAIT", "order": 4,
     "teaches": "Commit early vs wait: the risk collapses",
     "cast": [{"event_key": "train:2427", "role": "focal"}, {"event_key": "train:7465", "role": "contrast"}]},
    {"id": "rules-disagree", "title": "Rules disagree", "tag": "RULES", "order": 5,
     "teaches": "Miss screen vs Pc: the tier-mode knob",
     "cast": [{"event_key": "train:8756", "role": "focal"}, {"event_key": "train:9361", "role": "contrast"}]},
    {"id": "cov-unknown", "title": "Covariance unknown", "tag": "REFUSAL", "order": 6,
     "teaches": "Refusal, then resolution when real covariance arrives",
     "cast": [{"event_key": "train:2299", "role": "focal"}, {"event_key": "train:2759", "role": "contrast"}]},
    {"id": "failure", "title": "Where we fail", "tag": "FAILURE CASE", "order": 7,
     "teaches": "How the user knows when physics cannot reproduce the reported risk",
     "cast": [{"event_key": "train:4200", "role": "focal"}]},
    {"id": "blind-test", "title": "Blind test", "tag": "BLIND", "order": 8,
     "teaches": "Forecasts on unseen data; truth hidden",
     "cast": [{"event_key": f"test:{i}", "role": "focal"} for i in (891, 1568, 1448, 2059, 2091)]},
    {"id": "free-play", "title": "Free play", "tag": "FREE", "order": 9,
     "teaches": "Seeded random cast: input beyond the script", "cast": []},
]


def load_catalogue(scenarios_dir: Path) -> tuple[list[dict], str]:
    found: dict[str, dict] = {}
    if scenarios_dir.exists():
        for p in sorted(scenarios_dir.glob("*.json")):
            try:
                d = json.loads(p.read_text())
                if isinstance(d, dict) and d.get("id"):
                    found[d["id"]] = d
            except Exception as e:
                log.warning("scenario_unreadable", file=p.name, error=str(e))
    source = "shared/scenarios" if found else "built-in fallback"
    for b in BUILTIN:  # fill in any scenario the engine has not produced yet
        found.setdefault(b["id"], {**b, "_builtin": True})
    cat = sorted(found.values(), key=lambda d: (d.get("order", 99), d["id"]))
    return cat, source


def card(s: dict) -> dict:
    return {"id": s["id"], "title": s.get("title", s["id"]), "tag": s.get("tag", ""), "teaches": s.get("teaches", ""),
            "order": s.get("order", 99), "event_keys": [c["event_key"] for c in s.get("cast", [])],
            "seed": s.get("seed"), "source": "built-in" if s.get("_builtin") else "shared/scenarios"}


def resolve(s: dict, engine, seed: int) -> dict:
    """Fill in d0, seed, cast tca_sim (when absent) and the free-play cast. Deterministic given (scenario, seed)."""
    s = json.loads(json.dumps({k: v for k, v in s.items()}))
    d0 = parse_ts(s.get("d0_sim") or D0)
    cast = [c for c in s.get("cast", []) if engine.has_event(c["event_key"])]
    missing = [c["event_key"] for c in s.get("cast", []) if not engine.has_event(c["event_key"])]
    if not s.get("cast"):
        cast = [{"event_key": k, "role": "focal"} for k in engine.free_play_candidates(6, seed)]
    out = []
    for i, c in enumerate(cast):
        c = dict(c)
        if not c.get("tca_sim"):
            rows = engine.event_rows(c["event_key"])
            first = float(rows[0]["time_to_tca"]) if rows else 1.0
            c["tca_sim"] = iso(d0 + days(first) + timedelta(minutes=30 + 90 * i))
        out.append(c)
    s["cast"] = out
    s["d0_sim"] = iso(d0)
    s["seed"] = seed
    s.setdefault("preload_until_sim", iso(d0))
    s["missing_events"] = missing
    return s
