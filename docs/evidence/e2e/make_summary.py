"""Build docs/evidence/e2e/summary.json from the raw E2E working folder (var/e2e/, git-ignored: SQLite ledgers,
full SSE dumps and logs are too large and machine-specific to commit). Run from the repo root:

    python3 docs/evidence/e2e/make_summary.py

Writes: summary.json (aggregate counts), stories_report.json (verbatim copy of var/e2e/stories/report.json),
ops_day_dashboard_check.json (dashboard tiles recomputed from the saved /api/state snapshot) and ui/*.png
(the headless UI runner's screenshots) and assess_{csv,mp}.json (POST /api/assess responses). No keys or secrets are read: the backend logs are only scanned for the
`ready decisions=... llm=...` mode line and HTTP status codes.
"""
import json
import re
import shutil
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "var/e2e"
OUT = Path(__file__).resolve().parent


def mode_line(log: Path) -> str | None:
    for line in log.read_text(errors="ignore").splitlines():
        if " ready " in line and "decisions=" in line:
            return re.sub(r"\s+", " ", line.split("ready", 1)[1]).strip()
    return None


def http_codes(log: Path) -> dict:
    return dict(Counter(re.findall(r'HTTP/1\.1" (\d{3})', log.read_text(errors="ignore"))))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.loads((SRC / "stories/report.json").read_text())
    shutil.copy(SRC / "stories/report.json", OUT / "stories_report.json")
    scen = {k: v for k, v in rep.items() if not k.startswith("_")}
    stories = {
        "scenarios": sorted(scen),
        "sse_events": sum(v["n_events"] for v in scen.values()),
        "sse_event_types": len(set().union(*[set(v["types"]) for v in scen.values()])),
        "policy_verdicts": dict(sum((Counter(v["kpis"]["decisions"]) for v in scen.values()), Counter())),
        "decision_calls": sum(v["kpis"]["ai_calls"] for v in scen.values()),
        "fallbacks": sum(v["kpis"]["fallbacks"] for v in scen.values()),
        "focal_endpoint_status": sorted({s for v in scen.values() for p in v["per"].values() for s in p.values()}),
        "runner_http_errors": rep.get("_http_errors"),
        "ledger_verify": rep.get("_verify"),
        "backend_mode": mode_line(SRC / "stories/backend8031.log"),
        "backend_http_status_counts": http_codes(SRC / "stories/backend8031.log"),
    }
    stories["policy_verdicts_total"] = sum(stories["policy_verdicts"].values())

    ev = json.loads((SRC / "stories/events_late-cdm.json").read_text())
    late = {
        "sse_events": len(ev),
        "types": dict(Counter(e["type"] for e in ev)),
        "verdicts": [(e["data"]["verdict"], e["data"]["action"]) for e in ev if e["type"] == "policy.verdict"],
        "decision_providers": dict(Counter(e["data"].get("provider") for e in ev if e["type"] == "decision.made")),
    }

    snap = json.loads((SRC / "ops/dash_ops-day.json").read_text())
    dash, st = snap["dash"], snap["state"]
    P = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
    now = P(dash["sim_time"])
    act = [c for c in st["cast"] if c.get("latest") and c.get("outcome") is None and c["latest"].get("tier")]
    win = [c for c in act if timedelta(0) <= P(c["tca_sim"]) - now <= timedelta(hours=24)]
    raw = {
        "within_1km": sum(1 for c in win if c["latest"]["miss_m"] is not None and c["latest"]["miss_m"] < 1000),
        "within_5km": sum(1 for c in win if c["latest"]["miss_m"] is not None and c["latest"]["miss_m"] < 5000),
        "critical": sum(1 for c in act if c["latest"]["tier"] == "CRITICAL"),
        "elevated": sum(1 for c in act if c["latest"]["tier"] == "ELEVATED"),
        "awaiting": len(st["approvals_open"]),
    }
    tiles = {t["id"]: t["value"] for t in dash["tiles"]}
    check = {
        "method": "Ops day run at 8x on our own backend (port 8032), auto-approving, then paused; dashboard tiles "
                  "compared with counts recomputed from the saved /api/state cast (latest CDM per event)",
        "sim_time": dash["sim_time"], "cast": len(st["cast"]), "active": len(act), "in_24h_window": len(win),
        "tiles": {k: {"dashboard": tiles[k], "recomputed": v, "match": tiles[k] == v} for k, v in raw.items()},
        "verdicts": tiles.get("decisions"), "decision_calls": tiles.get("ai", {}).get("calls"),
        "mode": st.get("mode"), "insights": [i["text"] for i in dash["insights"]],
        "backend_mode": mode_line(SRC / "ops/backend.log"),
        "backend_http_status_counts": http_codes(SRC / "ops/backend.log"),
        "note_409": "the 409 Conflict responses are the runner re-posting approvals that were already resolved",
    }
    (OUT / "ops_day_dashboard_check.json").write_text(json.dumps(check, indent=1, ensure_ascii=False))

    for f in ("assess_csv.json", "assess_mp.json"):  # POST /api/assess responses for 5 ad-hoc CDM rows
        shutil.copy(SRC / "ops" / f, OUT / f)
    shots = sorted((SRC / "stories/shots").glob("*.png"))
    (OUT / "ui").mkdir(exist_ok=True)
    for s in shots:
        shutil.copy(s, OUT / "ui" / s.name)
    ui = {"runner": "var/e2e/stories/ui_stories.py (Chrome DevTools Protocol)", "scenarios_run": ["late-cdm"],
          "screenshots": [s.name for s in shots],
          "log_bytes": (SRC / "stories/ui_run1.log").stat().st_size,
          "note": "the runner's console-error and HTTP>=400 lists were printed, not saved; the log file is empty, "
                  "so no console or network numbers are claimed"}

    summary = {"run_date": "2026-10-31", "machine": "MacBook M2 8 GB",
               "ports": {"stories_backend": 8031, "ops_backend": 8032, "vite": 5191},
               "story_runner": stories, "late_cdm": late, "ops_day_dashboard_check": "ops_day_dashboard_check.json",
               "ui_runner": ui,
               "scenarios_not_driven": ["blind-test", "free-play"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
