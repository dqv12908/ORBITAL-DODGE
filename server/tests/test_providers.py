"""Decision provider chain: timeouts/errors fall back to mock with degraded events; Jev vote shares; cache."""
import asyncio

from backend_fixtures import mock_settings
from decisions.jev import JevProvider
from decisions.provider import DecisionService, ProviderError, build_request

FACTS = {"event_key": "train:1", "mission_id": 1, "chaser_type": "DEBRIS", "cdm_number": 1, "n_cdm_known": 1,
         "ttca": 3.0, "miss_m": 900.0, "pc": 1e-5, "pc_status": "OK", "tier": "CRITICAL", "pc_tier": "ELEVATED",
         "miss_tier": "CRITICAL", "tier_mode": "BRIEF", "reasons": ["Miss 900 m < 1 km (miss-distance screen)"],
         "rule_ids": ["R-MISS-CRIT"], "history": [], "forecast_valid": False, "mcp_open": False}
ALLOWED = ["REQUEST_TRACKING", "PREPARE_MANOEUVRE"]
BASE = {"action": "PREPARE_MANOEUVRE", "p_critical": 0.8, "expected_loss": {}}
LOSS = {"MONITOR": {"critical": 100, "not_critical": 0}, "REQUEST_TRACKING": {"critical": 30, "not_critical": 1},
        "PREPARE_MANOEUVRE": {"critical": 5, "not_critical": 8}}


class Slow:
    name, model = "openai-decisions", "gpt-6-luna"

    async def decide(self, req, timeout):
        await asyncio.sleep(10)


class Broken:
    name, model = "jev-router", "typesafe/jev-router"

    async def decide(self, req, timeout):
        raise ProviderError("error", "HTTP 500")


class Refuser:
    name, model = "openai-decisions", "gpt-6-luna"

    async def decide(self, req, timeout):
        return {"refused": True}


def svc(tmp_path, providers, **kw):
    s = mock_settings(tmp_path).model_copy(update={"mock_ai": False, "decisions_timeout_s": 0.2, **kw})
    object.__setattr__(s, "openrouter_api_key", "x")  # mode -> live
    return DecisionService(s, providers=providers)


def test_timeout_and_error_fall_back_to_mock(tmp_path):
    d = svc(tmp_path, [Slow(), Broken()])
    prep = d.prepare(FACTS, ALLOWED, "2026-10-31T06:00:00Z")
    r = asyncio.run(d.decide(FACTS, ALLOWED, BASE, LOSS, 1, prep))
    assert r["provider"] == "mock" and r["answer"] in ALLOWED
    assert [a["outcome"] for a in r["attempts"]] == ["timeout", "error", "ok"]
    assert len(r["degraded"]) == 2


def test_all_refuse_is_refusal_not_mock(tmp_path):
    d = svc(tmp_path, [Refuser()])
    prep = d.prepare(FACTS, ALLOWED, "t")
    r = asyncio.run(d.decide(FACTS, ALLOWED, BASE, LOSS, 1, prep))
    assert r["refused"] is True and r["answer"] is None


def test_jev_vote_share_and_routing(tmp_path, monkeypatch):
    s = mock_settings(tmp_path)
    object.__setattr__(s, "openrouter_api_key", type("K", (), {"get_secret_value": lambda self: "k"})())
    p = JevProvider(s)
    answers = iter(["PREPARE_MANOEUVRE", "PREPARE_MANOEUVRE", "REQUEST_TRACKING", "PREPARE_MANOEUVRE", "BOGUS"])

    async def one(client, req, seed):
        assert [c["action"] for c in req["questions"][0]["choices"]] == ALLOWED
        return {"action": next(answers), "evidence_sufficient": seed % 2 == 0,
                "model": "deepseek/deepseek-v4.1-flash-1017", "provider": "Crusoe"}
    monkeypatch.setattr(p, "_one", one)
    req, mapping = build_request(FACTS, ALLOWED, "t", p.model)
    raw = asyncio.run(p.decide(dict(req, _mapping=mapping), 5))
    assert raw["choice"] == "PREPARE_MANOEUVRE"
    assert raw["probabilities"] == {"REQUEST_TRACKING": 0.25, "PREPARE_MANOEUVRE": 0.75}
    assert raw["probability_kind"] == "vote share (n=4 of 5)"
    assert raw["provider_label"] == "jev-router→deepseek-v4.1-flash-1017"


def test_jev_too_few_votes_caps_confidence(tmp_path, monkeypatch):
    s = mock_settings(tmp_path)
    object.__setattr__(s, "openrouter_api_key", type("K", (), {"get_secret_value": lambda self: "k"})())
    p = JevProvider(s)
    answers = iter(["PREPARE_MANOEUVRE", "PREPARE_MANOEUVRE", "BOGUS", "BOGUS", "BOGUS"])

    async def one(client, req, seed):
        return {"action": next(answers), "evidence_sufficient": True, "model": "m", "provider": "p"}
    monkeypatch.setattr(p, "_one", one)
    req, mapping = build_request(FACTS, ALLOWED, "t", p.model)
    raw = asyncio.run(p.decide(dict(req, _mapping=mapping), 5))
    assert raw["probability_kind"] == "vote share (n=2 of 5)"
    assert raw["confidence"] <= 0.5  # a 2-vote answer must not clear the confidence bar


def test_cache_serves_second_call(tmp_path):
    calls = []

    class Once:
        name, model = "jev-router", "typesafe/jev-router"

        async def decide(self, req, timeout):
            calls.append(1)
            return {"choice": "PREPARE_MANOEUVRE", "probabilities": {"PREPARE_MANOEUVRE": 1.0}, "confidence": 1.0,
                    "provider_label": "jev-router→x", "probability_kind": "vote share (n=5)"}
    d = svc(tmp_path, [Once()])
    prep = d.prepare(FACTS, ALLOWED, "t")
    r1 = asyncio.run(d.decide(FACTS, ALLOWED, BASE, LOSS, 1, prep))
    r2 = asyncio.run(d.decide(FACTS, ALLOWED, BASE, LOSS, 1, prep))
    assert len(calls) == 1 and r2["cached"] and r1["answer"] == r2["answer"] == "PREPARE_MANOEUVRE"
    assert r2["provider"] == "jev-router→x"


def test_typesafe_parse_and_body():
    from decisions.typesafe import TypeSafeJev, parse
    body = {"model": "jev-1.14.0", "answers": {
        "next_action": {"type": "choice", "choice": "PREPARE_MANOEUVRE", "confidence": 0.86,
                        "probabilities": {"REQUEST_TRACKING": 0.07, "PREPARE_MANOEUVRE": 0.93}},
        "evidence_sufficient": {"type": "noul", "noul": 0.53}}, "usage": {"input_tokens": 521}}
    out = parse(body, ["REQUEST_TRACKING", "PREPARE_MANOEUVRE"])
    assert out["choice"] == "PREPARE_MANOEUVRE" and out["probabilities"]["PREPARE_MANOEUVRE"] == 0.93
    assert out["evidence_sufficient_p"] == 0.53 and out["model_version"] == "jev-1.14.0"
    s = type("S", (), {"typesafe_model": "jev-latest", "typesafe_url": "u"})()
    req, mapping = build_request(FACTS, ALLOWED, "t", "jev-latest")
    b, allowed = TypeSafeJev(s).body(dict(req, _mapping=mapping))
    assert allowed == ALLOWED and set(b["questions"]["next_action"]["criteria"]) == set(ALLOWED)
    assert b["questions"]["evidence_sufficient"]["type"] == "noul" and "R = 10" in b["state"]
