"""Deterministic advisor note (design §5.6). Always rendered first; the LLM may replace it only if verified.

The note reads like a colleague's advice, not a fact list: what is going on, what to do next and why, and why it is
(or is not) on the analyst's desk. 2-3 sentences, <= 60 words, no semicolons, no rule IDs, at most 3 numbers, never
contradicting the policy verdict, and the human always decides.
"""
from __future__ import annotations

from llm.factsheet import render

MAX_WORDS = 60

# rule id behind a primary reason -> why it is on the analyst's desk, in plain words
_WHY_RULE = {
    "Q-COV-DEFAULT": "the position uncertainty is a placeholder",
    "Q-PHYS-DISAGREE": "our risk estimate and the reported one disagree",
    "Q-LOW-VREL": "the objects drift past each other too slowly for the usual risk maths",
    "R-PC-CRIT": "the collision probability is above the critical line",
    "R-MISS-CRIT": "the miss distance is below the critical line",
    "T-MCP": "we are inside the last day before closest approach",
}
# policy reasons that carry no rule id (matched on their fixed wording)
_WHY_TEXT = [
    ("Covariance is a default", "the position uncertainty is a placeholder"),
    ("PREPARE not offered", "the position uncertainty is a placeholder"),
    ("Decision model only", "the decision model is not confident in its pick"),
    ("Decision model declined", "the decision model declined to choose"),
    ("disallowed action", "the decision model picked a step the rules do not allow"),
    ("evidence is thin", "the decision model says the evidence is thin"),
    ("cost rule disagree", "the decision model and the cost rule prefer different steps"),
    ("PREPARE is never automatic", "manoeuvre options always need a human go/no-go"),
    ("CRITICAL events always", "critical events always come to you"),
    ("WATCH", "the forecast puts it on watch"),
]


def _why(reason: str, facts: dict) -> str | None:
    rid = dict(zip(facts.get("reasons") or [], facts.get("rule_ids") or [])).get(reason)
    if rid in _WHY_RULE:
        return _WHY_RULE[rid]
    for key, phrase in _WHY_TEXT:
        if key in reason:
            return phrase
    return None


def _has(facts: dict, rid: str) -> bool:
    return rid in (facts.get("rule_ids") or [])


def situation(facts: dict) -> str:
    """Sentence 1: what is going on, in everyday words (at most 2 numbers)."""
    tier = facts.get("tier")
    miss = render(facts, "miss_m")
    when = render(facts, "ttca_words")
    if facts.get("pc_status") == "COV_UNKNOWN":
        return (f"We cannot size this risk yet: the objects pass {miss} apart in {when}, but one position "
                "uncertainty is a placeholder")
    if tier == "CRITICAL":
        s = f"This one is serious: the two objects are expected to pass {miss} apart in {when}"
    elif tier == "ELEVATED":
        s = f"This one needs a look: the two objects are expected to pass {miss} apart in {when}"
    elif facts.get("forecast_valid") and facts.get("p_critical") is not None:
        return (f"This looks quiet today, but the forecast gives it a {render(facts, 'p_critical')} chance of "
                "becoming critical")
    else:
        return f"This one looks quiet: the two objects pass {miss} apart in {when}"
    if _has(facts, "Q-PHYS-DISAGREE"):
        s += ", and our two risk estimates disagree a lot"
    elif _has(facts, "Q-LOW-VREL"):
        s += ", drifting past each other slowly"
    return s


def recommendation(facts: dict) -> str:
    """Sentence 2: what I'd do next and why (consistent with the policy's suggested action)."""
    act = facts.get("verdict_action")
    mcp = bool(facts.get("mcp_open"))
    if act == "PREPARE_MANOEUVRE":
        why = ("because waiting is no longer free this close in" if mcp
               else "while there is still time for a cheap, small burn")
        return f"I'd ask flight dynamics to prepare manoeuvre options now, {why}"
    if act == "REQUEST_TRACKING":
        if facts.get("pc_status") == "COV_UNKNOWN":
            why = "so we get a real position uncertainty before any manoeuvre planning"
        elif _has(facts, "Q-PHYS-DISAGREE"):
            why = "to settle which risk estimate is right"
        elif (facts.get("ttca") or 0) < 1:
            why = "though with under a day left it may land too late to change much"
        else:
            why = "while fresh data can still arrive in time to sharpen the picture"
        return f"I'd request extra tracking now, {why}"
    if facts.get("baseline_track_too_late"):
        why = "extra tracking can't arrive in time to help now"
    elif facts.get("tier") in ("ELEVATED", "CRITICAL"):
        why = "the numbers are low enough to wait for one more update"
    else:
        why = "nothing here needs a step yet"
    return f"I'd keep monitoring and check the next update in about {render(facts, 'next_cdm_h')}, since {why}"


def closing(facts: dict) -> str:
    """Sentence 3: why it is on the desk (or that it was logged), and that the analyst decides."""
    v = facts.get("verdict")
    if v == "AUTO":
        return "It is within policy, so I've logged it, and you can still override"
    prim = list(facts.get("verdict_primary_reasons") or facts.get("verdict_reasons") or [])
    whys = []
    for r in prim:
        w = _why(str(r), facts)
        if w and w not in whys:
            whys.append(w)
    if v == "BLOCK":
        return "Automation is blocked until better data arrives, so the call is yours"
    if whys:
        return f"It's your call: it came to you because {whys[0]}"
    return "It's your call"


def explain_template(facts: dict) -> dict:
    tier = facts.get("tier")
    act = facts.get("verdict_action")
    short = {"MONITOR": "keep monitoring", "REQUEST_TRACKING": "request tracking",
             "PREPARE_MANOEUVRE": "prepare options"}.get(act or "", "")
    headline = f"{tier} · {facts['event_key']}" + (f" · {short}" if short else "")
    sentences = [situation(facts)]
    if facts.get("verdict"):
        sentences += [recommendation(facts), closing(facts)]
    text = " ".join(s.rstrip(".") + "." for s in sentences)
    if len(text.split()) > MAX_WORDS and len(sentences) == 3:  # keep the advice, shorten the closing
        sentences[2] = "It's your call"
        text = " ".join(s.rstrip(".") + "." for s in sentences)
    return {"headline": headline, "body": text, "uncertainty": "", "text": text}
