"""TypeSafe Jev, official System One API (verified with a live call, 31 Oct 2026).

POST https://api.typesafe.ai/v1/systemone
  {model: "jev-latest", state: text,
   questions: {next_action: {type: "choice", instructions, criteria: {ACTION: description}},
               evidence_sufficient: {type: "noul", instructions}}}
-> {model: "jev-1.14.0", answers: {next_action: {type: "choice", choice, probabilities: {ACTION: p}, confidence},
                                    evidence_sufficient: {type: "noul", noul: p}}, usage}

Unlike the OpenRouter `jev-router` (which forwards to other chat models and returns no probabilities), this is Jev
itself and returns a calibrated probability per allowed action. These are decision-model probabilities over actions,
never collision probabilities.
"""
from __future__ import annotations

import time

import httpx

from decisions.provider import ProviderError

POLICY = ("OPERATING POLICY (the operator's, fixed): a missed critical event costs about 10x a false alarm "
          "(cost ratio R = 10). Action costs in false-alarm units: MONITOR 0; REQUEST_TRACKING 0.3 (only helps if it "
          "arrives at least ~1 day before TCA); PREPARE_MANOEUVRE 1.0; an unmitigated event that ends CRITICAL costs 10. "
          "A human analyst approves every action.")


def parse(body: dict, allowed: list[str]) -> dict:
    answers = body.get("answers") or {}
    if isinstance(answers, list):  # tolerate a list of named answers
        answers = {a.get("name"): a for a in answers if isinstance(a, dict)}
    if any(isinstance(a, dict) and a.get("type") == "refusal" for a in answers.values()):
        return {"refused": True}
    ch = answers.get("next_action")
    if not isinstance(ch, dict) or ch.get("type") != "choice":
        raise ProviderError("error", "no choice answer in response")
    probs = {a: float((ch.get("probabilities") or {}).get(a, 0.0)) for a in allowed}
    ev = answers.get("evidence_sufficient")
    ev_p = ev.get("noul") if isinstance(ev, dict) else None
    return {"choice": ch.get("choice"), "probabilities": probs, "confidence": ch.get("confidence"),
            "evidence_sufficient_p": ev_p, "usage": body.get("usage"), "model_version": body.get("model")}


class TypeSafeJev:
    name = "typesafe-jev"

    def __init__(self, settings):
        self.s = settings
        self.model = settings.typesafe_model
        self.url = settings.typesafe_url

    def body(self, request: dict) -> tuple[dict, list[str]]:
        mapping = request.get("_mapping") or {}
        q = request["questions"][0]
        criteria = {mapping.get(c["value"], c["value"]): c["description"] for c in q["choices"]}
        ev_q = next((x for x in request["questions"] if x.get("name") == "evidence_sufficient"), None)
        questions = {"next_action": {"type": "choice", "instructions": q["instructions"], "criteria": criteria}}
        if ev_q:
            questions["evidence_sufficient"] = {"type": "noul", "instructions": ev_q["instructions"]}
        return {"model": self.model, "state": f"{request['input']}\n{POLICY}", "questions": questions}, list(criteria)

    async def decide(self, request: dict, timeout: float) -> dict:
        t0 = time.monotonic()
        body, allowed = self.body(request)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(self.url, json=body, headers={
                "Authorization": f"Bearer {self.s.typesafe_api_key.get_secret_value()}",
                "Content-Type": "application/json", "X-Title": "ORBITAL DODGE"})
        if r.status_code >= 400:
            raise ProviderError("error", f"HTTP {r.status_code}: {r.text[:120]}")
        out = parse(r.json(), allowed)
        if out.get("refused"):
            return out
        version = out.pop("model_version", None) or self.model
        return {**out, "probability_kind": "model probabilities (Jev)", "routed_model": version,
                "provider_label": f"typesafe-jev ({version})", "latency_ms": int((time.monotonic() - t0) * 1000)}
