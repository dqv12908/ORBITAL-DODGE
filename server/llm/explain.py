"""J1 event explanation: template first (always), then optional Claude with numbers as {{placeholders}} -> verifier."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import structlog

from app.util import canonical, dumps, sha256
from llm.factsheet import history_line, render
from llm.templates import explain_template
from llm.verify import PLACEHOLDER, verify

log = structlog.get_logger("llm")
PROMPT_VERSION = "explain-v1"

SYSTEM = """You explain satellite conjunction assessments to a duty analyst at a satellite operator. You do not compute, decide or approve anything.

RULES
- Use only FACTS. Write every number as a placeholder {{key}} where key is a FACTS key whose value you mean; never type a digit yourself.
- A placeholder renders WITH its unit (e.g. {{miss_m}} -> "136 m", {{ttca}} -> "10.5 h"); never write a unit word after a placeholder.
- Do not round, convert or compare numbers yourself; comparisons are already in FACTS.reasons and FACTS.verdict_reasons, which you may quote verbatim.
- Never call a collision probability (Pc) a percentage. Forecast and decision-confidence values are percentages and are NOT collision probabilities.
- The policy verdict and the allowed actions are final; explain them, never change them. Never tell the analyst to burn, manoeuvre now, dismiss or ignore the event. A human decides.
- If the FACTS do not support an answer, set refusal to {"reason": "..."} and leave the other fields short.
- Put every FACTS key you used in `cited`.

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

OUTPUT
- headline: one short line. body: 2 to 4 sentences: why the event is (or is not) flagged, what changed since the last CDM, what the policy verdict means. uncertainty: one sentence on what is uncertain.
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "body": {"type": "string"},
        "uncertainty": {"type": "string"},
        "cited": {"type": "array", "items": {"type": "string"}},
        "refusal": {"anyOf": [{"type": "null"}, {"type": "object", "properties": {"reason": {"type": "string"}},
                                                  "required": ["reason"], "additionalProperties": False}]},
    },
    "required": ["headline", "body", "uncertainty", "cited", "refusal"],
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


def check_llm_output(out: dict, facts: dict) -> dict:
    """Render + verify a structured LLM output. Returns {ok, text, reason}."""
    if out.get("refusal"):
        return {"ok": False, "text": None, "reason": f"refused: {out['refusal'].get('reason', '')}"}
    pieces = []
    unknown_all: list[str] = []
    for k in ("headline", "body", "uncertainty"):
        t, unknown = render_placeholders(str(out.get(k) or ""), facts)
        unknown_all += unknown
        pieces.append(t)
    headline, body, unc = pieces
    text = f"{headline}. {body} {unc}".strip()
    ok, reason = verify(text, facts, cited=out.get("cited") or [], verdict_action=facts.get("verdict_action"),
                        unknown_placeholders=unknown_all)
    return {"ok": ok, "text": text if ok else None, "reason": reason, "headline": headline, "body": body,
            "uncertainty": unc}


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

    def template(self, facts: dict) -> dict:
        t = explain_template(facts)
        ok, reason = verify(t["text"], facts, verdict_action=facts.get("verdict_action"))
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
