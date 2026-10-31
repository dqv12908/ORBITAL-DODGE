"""J1 event explanation: template first (always), then optional Claude with numbers as {{placeholders}} -> verifier."""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path

import structlog

from app.util import canonical, dumps, sha256
from llm.factsheet import history_line, render
from llm.templates import explain_template
from llm.verify import PLACEHOLDER, extract, verify

log = structlog.get_logger("llm")
PROMPT_VERSION = "explain-v2-advisor"

SYSTEM = """You are a calm, experienced colleague advising the duty analyst at a satellite operator about one possible collision. You explain and advise; you do not compute, decide or approve anything. The analyst decides.

RULES
- Use only FACTS. Write every number as a placeholder {{key}} where key is a FACTS key whose value you mean; never type a digit yourself.
- A placeholder renders WITH its unit (e.g. {{miss_m}} -> "136 m", {{ttca_words}} -> "11 hours"); never write a unit word after a placeholder.
- Do not round, convert or compare numbers yourself; the comparisons are already made in FACTS.reasons and FACTS.verdict_primary_reasons. Retell them in plain words.
- Never call a collision probability (Pc) a percentage. Forecast and decision-confidence values are percentages and are NOT collision probabilities.
- The policy verdict and the allowed actions are fixed by code; advise within them, never change them. The analyst makes the decision.
- If the FACTS do not support an answer, set refusal to {"reason": "..."} and leave the other fields short.
- Put every FACTS key you used in `cited`.
- FACTS.baseline_track_too_late = yes means extra tracking cannot arrive in time to help (under a day to closest approach).

GLOSSARY
- Pc: probability of collision from a 2D encounter-plane integral of the combined position covariance (physics).
- TCA: time of closest approach. MCP: manoeuvre commitment point, 24 h before TCA; inside it, waiting is no longer free.
- Miss-distance screen: the brief's rule that a miss under 1 km is CRITICAL regardless of covariance. It can over-flag.
- Default covariance: a sentinel value of 10 Earth radii meaning the covariance is unknown; Pc cannot be computed.
- Dilution: a low Pc caused by a covariance larger than the worst case; low Pc is then not reassurance.
- Forecast: calibrated statistical chance that the event ends CRITICAL at its final CDM; valid only 2 to 7 days before TCA.
- Decision model confidence: how sure the decision model is about its chosen action; not a collision probability.

TIER TABLE (brief, NASA CARA / ESA practice)
- CRITICAL: Pc >= 1e-4 or miss < 1 km. ELEVATED: 1e-6 <= Pc < 1e-4 or miss in [1, 5] km. SAFE: Pc < 1e-6 and miss > 5 km.
- Tier mode BRIEF uses the higher of the Pc and miss sub-tiers; CARA_PC uses the Pc sub-tier only.

OUTPUT (an advisor's note, not a fact list)
- headline: at most 8 plain words.
- advice: 2 or 3 short sentences (never 4), aim for 40 to 50 words and never more than 60, in plain English a
  non-specialist understands.
  1) What is going on, in everyday words (how close, how soon, and the one thing that matters most).
  2) What you recommend next and why. Your recommendation MUST be FACTS.verdict_action in plain words
     (MONITOR = keep watching and look again at the next update; REQUEST_TRACKING = ask for fresh tracking;
     PREPARE_MANOEUVRE = have flight dynamics prepare manoeuvre options for a go/no-go).
  3) Optional, one sentence: why it came to the analyst (the first of FACTS.verdict_primary_reasons, in your own plain
     words) or, if the verdict is AUTO, that it was logged within policy, ending with the fact that the analyst decides.
- Speak in the first person ("I'd ..."). Plain words over jargon: "closest approach" not "TCA", "collision
  probability" not "Pc", "position uncertainty" not "covariance". Prefer {{miss_m}} and {{ttca_words}}.
- At most 3 numbers in total (each placeholder counts as one). No semicolons, no lists, no rule IDs (like
  R-MISS-CRIT or T-MCP), no quoting of reason strings, no percentages of the decision model unless they explain the
  recommendation.
- Never say the decision is final or made, never contradict the verdict, never tell the analyst to burn or
  manoeuvre now, dismiss or ignore the event.

EXAMPLE (style only; use the real FACTS): "This one is close: the two objects are expected to pass {{miss_m}} apart in
{{ttca_words}}, and our two risk estimates disagree a lot. I'd ask fresh tracking to settle which one is right, and it
can still arrive in time. It's your call: it came to you because the estimates disagree."
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "advice": {"type": "string"},
        "cited": {"type": "array", "items": {"type": "string"}},
        "refusal": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {"reason": {"type": "string"}},
                                                  "required": ["reason"], "additionalProperties": False}]},
    },
    "required": ["headline", "advice", "cited", "refusal"],
    "additionalProperties": False,
}


def render_placeholders(text: str, facts: dict) -> tuple[str, list[str]]:
    unknown: list[str] = []

    def repl(m):
        k = m.group(1)
        if k not in facts:
            unknown.append(k)
            return m.group(0)
        return render(facts, k)
    return PLACEHOLDER.sub(repl, text), unknown


MAX_WORDS = 60
MAX_NUMBERS = 3
MAX_SENTENCES = 3
_SENT_END = re.compile(r"[.!?](?=\s|$)")
_RULE_ID = re.compile(r"\b(?:[RQT]-[A-Z0-9-]+|MISS-SCREEN)\b")


def style_check(text: str) -> tuple[bool, str | None]:
    """The advisor-note style: sentences, not a fact list. No ';', <= 60 words, <= 3 sentences, <= 3 numbers, no rule IDs."""
    if ";" in text:
        return False, "style: semicolon list"
    n_words = len(text.split())
    if n_words > MAX_WORDS:
        return False, f"style: {n_words} words > {MAX_WORDS}"
    n_sent = len(_SENT_END.findall(text.strip()))
    if n_sent > MAX_SENTENCES:
        return False, f"style: {n_sent} sentences > {MAX_SENTENCES}"
    if _RULE_ID.search(text):
        return False, f"style: rule id {_RULE_ID.search(text).group(0)}"
    n_num = len(extract(text))
    if n_num > MAX_NUMBERS:
        return False, f"style: {n_num} numbers > {MAX_NUMBERS}"
    return True, None


def check_llm_output(out: dict, facts: dict) -> dict:
    """Render + verify a structured LLM output. Returns {ok, text, reason}. `text` is the advice only (the headline is
    separate); numbers and action phrases are verified over headline + advice together."""
    if out.get("refusal"):
        return {"ok": False, "text": None, "reason": f"refused: {out['refusal'].get('reason', '')}"}
    unknown_all: list[str] = []
    headline, unknown = render_placeholders(str(out.get("headline") or ""), facts)
    unknown_all += unknown
    raw_advice = out.get("advice")
    if raw_advice is None:  # older shape: body + uncertainty
        raw_advice = f"{out.get('body') or ''} {out.get('uncertainty') or ''}".strip()
    advice, unknown = render_placeholders(str(raw_advice), facts)
    unknown_all += unknown
    advice = advice.strip()
    ok, reason = verify(f"{headline}. {advice}", facts, cited=out.get("cited") or [],
                        verdict_action=facts.get("verdict_action"), unknown_placeholders=unknown_all)
    if ok:
        ok, reason = style_check(advice)
    return {"ok": ok, "text": advice if ok else None, "reason": reason, "headline": headline, "advice": advice}


class Explainer:
    def __init__(self, settings):
        self.s = settings
        self.client = None
        self.stats = {"calls": 0, "verified": 0, "rejected": 0, "errors": 0, "cache_hits": 0}
        self.status = "unconfigured"
        if settings.llm_mode == "live":
            try:
                import anthropic
                self.client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value(),
                                                       timeout=settings.llm_timeout_s, max_retries=1)
                self.status = "ok"
            except Exception as e:
                log.warning("anthropic_init_failed", error=str(e))
                self.status = "down"
        self.cache_dir: Path = settings.cache_root / "llm"
        self.sem = asyncio.Semaphore(2)  # bounded concurrency for outbound LLM calls

    def template(self, facts: dict) -> dict:
        t = explain_template(facts)
        ok, reason = verify(f"{t['headline']}. {t['text']}", facts, verdict_action=facts.get("verdict_action"))
        if ok:
            ok, reason = style_check(t["text"])
        return {**t, "self_check": ok, "self_check_reason": reason}

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    async def llm(self, facts: dict, task: str = "explain_flag") -> dict:
        """Returns {status: ok|rejected|error|disabled, text?, reason?, latency_ms}."""
        shown = {k: render(facts, k) for k in facts if k != "history"}
        shown["history"] = history_line(facts)
        user = ("FACTS (key -> exactly how {{key}} renders on screen, units included) = "
                + canonical(shown).decode() + f"\nTASK = {task}")
        key = sha256(canonical({"model": self.s.llm_model, "v": PROMPT_VERSION, "system": SYSTEM, "user": user}))
        cp = self._cache_path(key)
        t0 = time.monotonic()
        out = None
        if cp.exists():
            try:
                out = json.loads(cp.read_text())
                self.stats["cache_hits"] += 1
            except Exception:
                out = None
        if out is None:
            if self.client is None:
                return {"status": "disabled", "reason": "LLM not configured (template only)"}
            self.stats["calls"] += 1
            try:
                async with self.sem:
                    resp = await asyncio.wait_for(self.client.messages.create(
                        model=self.s.llm_model,
                        max_tokens=2000,
                        system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
                        messages=[{"role": "user", "content": user}],
                    ), timeout=self.s.llm_timeout_s + 1)
                if resp.stop_reason == "refusal":
                    return {"status": "rejected", "reason": "model refusal", "latency_ms": int((time.monotonic() - t0) * 1000)}
                txt = next((b.text for b in resp.content if b.type == "text"), "")
                out = json.loads(txt)
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    cp.write_text(dumps(out))
                except Exception:
                    pass
                self.status = "ok"
            except Exception as e:
                self.stats["errors"] += 1
                self.status = "down"
                return {"status": "error", "reason": f"{type(e).__name__}: {str(e)[:160]}",
                        "latency_ms": int((time.monotonic() - t0) * 1000)}
        res = check_llm_output(out, facts)
        lat = int((time.monotonic() - t0) * 1000)
        if res["ok"]:
            self.stats["verified"] += 1
            return {"status": "ok", "text": res["text"], "headline": res["headline"], "latency_ms": lat, "prompt_hash": key}
        self.stats["rejected"] += 1
        return {"status": "rejected", "reason": res["reason"], "latency_ms": lat, "prompt_hash": key}
