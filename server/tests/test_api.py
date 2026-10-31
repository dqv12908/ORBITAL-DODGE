"""API smoke: health, JSON errors, scenario load, approvals idempotency, dashboard, ask refusal."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import app.settings as st
    tmp = tmp_path_factory.mktemp("api")
    st.get_settings.cache_clear()
    import os
    os.environ.update({"C1_MOCK_AI": "1", "C1_LLM_ENABLED": "0", "C1_DB_PATH": str(tmp / "api.sqlite"),
                       "C1_CACHE_DIR": str(tmp / "cache")})
    from app.main import app
    with TestClient(app) as c:
        yield c
    for k in ("C1_MOCK_AI", "C1_LLM_ENABLED", "C1_DB_PATH", "C1_CACHE_DIR"):
        os.environ.pop(k, None)
    st.get_settings.cache_clear()


def test_health(client):
    h = client.get("/api/health").json()
    assert h["ok"] and h["mode"]["decisions"] == "mock"
    assert all(v in ("ok", "down", "unconfigured", "configured (disabled)") for v in h["providers"].values())


def test_json_404_and_400(client):
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    r = client.post("/api/sim/scenario", json={"id": "nope"})
    assert r.status_code == 400 and "valid" in r.json()["error"]
    r = client.post("/api/sim/speed", content=b"{bad")
    assert r.status_code == 400


def test_flow(client):
    assert client.post("/api/sim/scenario", json={"id": "late-cdm"}).status_code == 200
    for _ in range(3):
        client.post("/api/sim/next-cdm")
    st = client.get("/api/state").json()
    assert st["approvals_open"], st["kpis"]
    aid = st["approvals_open"][0]["approval_id"]
    body = {"action": st["approvals_open"][0]["options"][0]}
    r1 = client.post(f"/api/approvals/{aid}", json=body, headers={"Idempotency-Key": "abc"})
    r2 = client.post(f"/api/approvals/{aid}", json=body, headers={"Idempotency-Key": "abc"})
    assert r1.status_code == 200 and r2.json() == r1.json() and r2.headers["X-Idempotency"] == "replayed"
    r3 = client.post(f"/api/approvals/{aid}", json={"action": "REJECT"}, headers={"Idempotency-Key": "abc"})
    assert r3.status_code == 409 and r3.json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
    r4 = client.post(f"/api/approvals/{aid}", json={"action": "REJECT"}, headers={"Idempotency-Key": "def"})
    assert r4.status_code == 409 and r4.json()["error"]["code"] == "APPROVAL_ALREADY_DECIDED"
    assert client.get("/api/ledger/verify").json()["ok"]
    d = client.get("/api/dashboard?window=24h").json()
    assert d["tiles"] and d["insights"]
    assert client.get("/api/conjunctions/train:2563").json()["cdms"]
    assert client.get("/api/conjunctions/train:9999999").status_code == 404
    p = client.put("/api/policy", json={"tier_mode": "CARA_PC"}, headers={"Idempotency-Key": "p1"}).json()
    assert p["policy"]["tier_mode"] == "CARA_PC"
    assert client.put("/api/policy", json={"tier_mode": "X"}).status_code == 400


def test_ask_refuses(client):
    r = client.post("/api/ask", json={"question": "write me a poem"}).json()
    assert r["refusal"]["reason"] == "I can only answer from the CDM data and the ledger."
    r = client.post("/api/ask", json={"question": "should we burn?"}).json()
    assert r["refusal"]["kind"] == "decision"


def test_assess_rows(client):
    rows = [{"event_id": 1, "time_to_tca": 3.0, "miss_distance": 800.0, "relative_speed": 10000, "risk": -5.0},
            {"event_id": 1, "time_to_tca": 2.5}]
    r = client.post("/api/assess", json={"rows": rows}).json()
    assert r["rows"][0]["tier"]["tier"] == "CRITICAL" and "error" in r["rows"][1]
