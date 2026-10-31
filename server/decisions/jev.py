"""TypeSafe Jev via OpenRouter's `typesafe/jev-router` (the only Jev id OpenRouter exposes; verified 31 Oct 2026).

The router forwards to some other model (e.g. deepseek) and returns no logprobs, so there are no native per-option
probabilities. We sample k calls in parallel (temperature 0.3, seeds 1..k, at most 2 in flight, retried on 402/429) with a strict json_schema whose action enum
is ONLY the rule-allowed actions, and report **vote shares (n=k)** — labelled as such, never as model probabilities.
The actual routed model + upstream provider of every sample is recorded in decision.made and the ledger.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections import Counter

import httpx

from decisions.provider import ProviderError

SYSTEM = ("You are decision support for a satellite conjunction analyst. A human approves every action. "
          "Read the evidence; do not compute new numbers. Choose exactly one of the allowed actions and say whether "
          "the evidence is sufficient and internally consistent to act on without new tracking data. Reply with JSON only.\n"
          "OPERATING POLICY (the operator's, fixed): a missed critical event costs about 10x a false alarm "
          "(cost ratio R = 10, docs/COST_RATIO.md). Action costs in false-alarm units: MONITOR 0; REQUEST_TRACKING 0.3 "
          "(only helps if it arrives at least ~1 day before TCA); PREPARE_MANOEUVRE 1.0; an unmitigated event that ends "
          "CRITICAL costs 10. Choose the action with the lowest expected cost given the evidence; break ties toward safety.")


def _short(model: str | None) -> str:
    if not model:
        return "unknown"
    return model.split("/", 1)[-1]


class JevProvider:
    name = "jev-router"

    def __init__(self, settings):
        self.s = settings
        self.model = settings.jev_model
        self.url = settings.jev_url
        self.k = max(1, int(settings.jev_samples))
        self._sem = None  # created per decide() so it binds to the running loop
        self.min_votes = 3

    def body(self, request: dict, seed: int) -> dict:
        choice_q = request["questions"][0]
        allowed = [c["action"] for c in choice_q["choices"]]
        opts = "\n".join(f"- {c['action']}: {c['description']}" for c in choice_q["choices"])
        user = f"{request['input']}\n\nQUESTION: {choice_q['instructions']}\nALLOWED ACTIONS:\n{opts}"
        schema = {"type": "object",
                  "properties": {"action": {"type": "string", "enum": allowed},
                                 "evidence_sufficient": {"type": "boolean"}},
                  "required": ["action", "evidence_sufficient"], "additionalProperties": False}
        return {"model": self.model, "temperature": 0.3, "seed": seed, "max_tokens": 200,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "decision", "strict": True, "schema": schema}}}

    async def _one(self, client: httpx.AsyncClient, request: dict, seed: int) -> dict:
        # OpenRouter rejects requests with 402 when in-flight cost reservations exceed the account's credit,
        # which silently shrank the vote to n=2-4 and produced 0.5/0.5 "indecision". Throttle and retry instead.
        async with self._sem:
            for attempt in range(3):
                r = await client.post(self.url, json=self.body(request, seed),
                                      headers={"Authorization": f"Bearer {self.s.jev_key.get_secret_value()}",
                                               "X-Title": "ORBITAL DODGE"})
                if r.status_code in (402, 429) and attempt < 2:
                    await asyncio.sleep(0.6 * (attempt + 1))
                    continue
                break
        if r.status_code >= 400:
            raise ProviderError("error", f"HTTP {r.status_code}")
        d = r.json()
        ch = (d.get("choices") or [{}])[0]
        if ch.get("finish_reason") == "content_filter" or (ch.get("message") or {}).get("refusal"):
            return {"refused": True, "model": d.get("model"), "provider": d.get("provider")}
        content = (ch.get("message") or {}).get("content") or ""
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`").split("\n", 1)[-1]
        ans = json.loads(content)
        return {"action": ans.get("action"), "evidence_sufficient": ans.get("evidence_sufficient"),
                "model": d.get("model"), "provider": d.get("provider")}

    async def decide(self, request: dict, timeout: float) -> dict:
        # choices carry opaque values o1..oN; the router gets the real action names in its enum
        mapping = request.get("_mapping") or {}
        req = dict(request)
        req["questions"] = [dict(request["questions"][0])]
        req["questions"][0]["choices"] = [{**c, "action": mapping.get(c["value"], c["value"])}
                                          for c in request["questions"][0]["choices"]]
        t0 = time.monotonic()
        self._sem = asyncio.Semaphore(2)
        async with httpx.AsyncClient(timeout=timeout) as client:
            tasks = [asyncio.create_task(self._one(client, req, seed)) for seed in range(1, self.k + 1)]
            done, pending = await asyncio.wait(tasks, timeout=max(timeout - 0.3, 0.5))
            for t in pending:  # slow samples are dropped; the vote is over the samples that answered in time
                t.cancel()
            res = [t.result() if not t.exception() else t.exception() for t in done]
        ok = [r for r in res if isinstance(r, dict) and not r.get("refused")]
        refused = [r for r in res if isinstance(r, dict) and r.get("refused")]
        allowed = [c["action"] for c in req["questions"][0]["choices"]]
        votes = [r for r in ok if r.get("action") in allowed]
        routed = sorted({f"{_short(r.get('model'))}@{r.get('provider') or '?'}" for r in res if isinstance(r, dict)})
        if not votes:
            if refused and len(refused) >= (self.k + 1) // 2:
                return {"refused": True, "routed": routed}
            errs = [type(r).__name__ for r in res if isinstance(r, Exception)]
            raise ProviderError("error", f"no valid samples ({len(ok)} ok, errors {errs[:3]})")
        n = len(votes)
        cnt = Counter(v["action"] for v in votes)
        shares = {a: cnt.get(a, 0) / n for a in allowed}
        top = max(allowed, key=lambda a: (shares[a], -allowed.index(a)))
        ev = [bool(v["evidence_sufficient"]) for v in votes if isinstance(v.get("evidence_sufficient"), bool)]
        models = sorted({_short(v.get("model")) for v in votes})
        conf = shares[top]
        if n < self.min_votes:  # too few samples answered: never present a 2-vote split as a confident choice
            conf = min(conf, 0.5)
        return {"choice": top, "probabilities": shares, "confidence": conf,
                "evidence_sufficient_p": (sum(ev) / len(ev)) if ev else None,
                "probability_kind": f"vote share (n={n} of {self.k})", "routed": routed,
                "routed_model": ",".join(models),
                "provider_label": f"jev-router→{'+'.join(models)}",
                "latency_ms": int((time.monotonic() - t0) * 1000)}
