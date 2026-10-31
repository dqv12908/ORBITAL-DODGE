"""Deterministic mock decision model (design §4.8): softmax(-E[loss]/2) over the allowed actions.

It agrees with the expected-cost baseline by construction and is labelled `DECISIONS: MOCK (cost rule)`.
evidence_sufficient_p is a plausible, seeded value derived from the fact sheet's data-quality flags.
"""
from __future__ import annotations

import math
import random

from decisions.expected_cost import expected_losses


def decide(facts: dict, allowed: list[str], baseline: dict, loss: dict, seed: int, request_hash: str) -> dict:
    p = baseline.get("p_critical", 0.05)
    el = expected_losses(p, allowed, loss)
    m = min(el.values()) if el else 0
    w = {a: math.exp(-(el[a] - m) / 2.0) for a in allowed}
    s = sum(w.values()) or 1.0
    probs = {a: w[a] / s for a in allowed}
    ans = min(allowed, key=lambda a: (el[a], allowed.index(a))) if allowed else None
    rng = random.Random(f"{seed}:{request_hash}")
    ev = 0.82
    rid = set(facts.get("rule_ids") or [])
    if facts.get("pc_status") == "COV_UNKNOWN":
        ev = 0.18
    elif "Q-PHYS-DISAGREE" in rid or "Q-LOW-VREL" in rid:
        ev = 0.35
    elif "Q-DILUTION" in rid:
        ev = 0.55
    ev = min(max(ev + rng.uniform(-0.05, 0.05), 0.01), 0.99)
    return {"choice": ans, "probabilities": probs, "confidence": probs.get(ans), "evidence_sufficient_p": ev}
