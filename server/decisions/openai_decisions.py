"""OpenAI Decisions API provider (public beta; shape verified 9 Oct 2026, see docs/design/00-context.md).

POST https://api.openai.com/v1/decisions {model, input, questions[]}
-> {answers:[{type:'choice', name, choice, probabilities, confidence} | {type:'predicate', name, probability} | {type:'refusal'}], usage}
"""
from __future__ import annotations

import time

import httpx

from decisions.provider import ProviderError


def _probs(p) -> dict:
    if isinstance(p, dict):
        return p
    if isinstance(p, list):  # tolerate [{value, probability}] shape
        out = {}
        for it in p:
            if isinstance(it, dict):
                k = it.get("value") or it.get("choice") or it.get("label")
                out[k] = it.get("probability", it.get("p", 0.0))
        return out
    return {}


def parse(body: dict) -> dict:
    answers = body.get("answers") or []
    if any(isinstance(a, dict) and a.get("type") == "refusal" for a in answers):
        return {"refused": True}
    choice = next((a for a in answers if a.get("type") == "choice" or a.get("name") == "next_action"), None)
    pred = next((a for a in answers if a.get("type") == "predicate" or a.get("name") == "evidence_sufficient"), None)
    if choice is None:
        raise ProviderError("error", "no choice answer in response")
    return {"choice": choice.get("choice") or choice.get("answer") or choice.get("value"),
            "probabilities": _probs(choice.get("probabilities")),
            "confidence": choice.get("confidence"),
            "evidence_sufficient_p": None if pred is None else pred.get("probability"),
            "usage": body.get("usage")}


class OpenAIDecisions:
    name = "openai-decisions"

    def __init__(self, settings):
        self.s = settings
        self.model = settings.openai_decisions_model
        self.url = settings.openai_decisions_url

    async def decide(self, request: dict, timeout: float) -> dict:
        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=timeout) as client:
            body = {k: v for k, v in request.items() if not k.startswith("_")}
            r = await client.post(self.url, json=body, headers={
                "Authorization": f"Bearer {self.s.openai_api_key.get_secret_value()}",
                "Content-Type": "application/json"})
        if r.status_code >= 400:
            raise ProviderError("error", f"HTTP {r.status_code}: {r.text[:120]}")
        out = parse(r.json())
        out["latency_ms"] = int((time.monotonic() - t0) * 1000)
        return out
