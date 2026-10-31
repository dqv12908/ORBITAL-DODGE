"""SSE envelopes validate against the pydantic union and the exported shared/schema/events.schema.json."""
import json

import jsonschema

from app.schema import json_schema, validate
from app.settings import get_settings
from backend_fixtures import run_scenario

SCHEMA_FILE = get_settings().schema_dir / "events.schema.json"


def test_schema_file_up_to_date():
    assert SCHEMA_FILE.exists(), "run: cd server && .venv/bin/python -m app.schema"
    assert json.loads(SCHEMA_FILE.read_text()) == json.loads(json.dumps(json_schema(), sort_keys=True))


def test_scenario_events_validate(tmp_path):
    sim, envs, _ = run_scenario(tmp_path, "cov-unknown")
    _, envs2, _ = run_scenario(tmp_path, "late-cdm", db_name="b.sqlite")
    envs = envs + envs2 + [sim.bus.snapshot_envelope()]
    schema = json.loads(SCHEMA_FILE.read_text())
    validator = jsonschema.Draft202012Validator(schema)
    seen = set()
    for e in envs:
        e = json.loads(json.dumps(e))
        validate(e)
        errs = list(validator.iter_errors(e))
        assert not errs, (e["type"], errs[0].message)
        seen.add(e["type"])
    assert {"risk.assessed", "decision.made", "policy.verdict", "approval.required", "explanation.ready",
            "log.entry", "outcome.revealed", "ledger.appended"} <= seen


def test_cov_unknown_scenario_blocks(tmp_path):
    sim, envs, _ = run_scenario(tmp_path, "cov-unknown")
    verdicts = [e["data"]["verdict"] for e in envs if e["type"] == "policy.verdict"]
    assert "BLOCK" in verdicts
    for e in envs:
        if e["type"] == "risk.assessed" and e["data"]["pc"]["status"] == "COV_UNKNOWN":
            assert e["data"]["pc"]["log10"] is None and e["data"]["reported_log10"] is None


def test_late_cdm_critical_escalates(tmp_path):
    sim, envs, _ = run_scenario(tmp_path, "late-cdm")
    crit = [e for e in envs if e["type"] == "risk.assessed" and e["data"]["tier"]["tier"] == "CRITICAL"]
    assert crit
    v = [e for e in envs if e["type"] == "policy.verdict" and e["data"]["cdm_id"] == crit[-1]["data"]["cdm_id"]]
    assert v and v[0]["data"]["verdict"] != "AUTO"
