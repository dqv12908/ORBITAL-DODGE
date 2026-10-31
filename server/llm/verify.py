"""Number-verification guardrail (design §5.3).

Every numeral in generated text must equal a fact value *as rendered* (or the same value in another unit within 1 %).
A Pc written as a percent, a forecast written in scientific notation, an unknown placeholder or cited key, or an action
phrase that contradicts the policy verdict -> reject.
"""
from __future__ import annotations

import re

from llm.factsheet import PC_KEYS, PCT_KEYS, history_line, render

SUPMAP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹−", "-0123456789-")
PLACEHOLDER = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}")

_SCI_X = re.compile(r"(\d+(?:\.\d+)?)\s*[×x\*]\s*10\s*\^?\s*([-−⁻]?\s*[0-9⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_SCI_E = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)[eE]\s*([-−+]?\d+)\b")
_SCI_10 = re.compile(r"(?<![\d.])10\s*(?:\^\s*([-−]?\d+)|([⁻][⁰¹²³⁴⁵⁶⁷⁸⁹]+))")
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)")
_SPEED = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(?:m/s|km/s)")
_LEN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(km|m|metres|meters|kilometres|kilometers)\b")
_TIME = re.compile(r"(\d+(?:\.\d+)?)\s*(h|hrs?|hours?|d|days?|min|minutes?)\b")
_PLAIN = re.compile(r"\d+(?:[.,]\d+)?")

ALWAYS_FORBIDDEN = [
    r"\bmanoeuvre now\b", r"\bmaneuver now\b", r"\bburn now\b", r"\bexecute (?:the |a )?(?:manoeuvre|maneuver|burn)\b",
    r"\bdismiss\b", r"\bsafe to ignore\b", r"\bignore (?:this|the) (?:event|alert|conjunction)\b",
    r"\bi (?:approve|have approved)\b", r"\bapproved\b",
]
PREPARE_PHRASES = [r"\b(?:prepare|plan|perform|execute|start|begin|do)\w* (?:a |the )?(?:collision[- ]avoidance )?(?:manoeuvre|maneuver|burn)\b",
                   r"\bmanoeuvre (?:is )?(?:needed|required|recommended)\b", r"\bmaneuver (?:is )?(?:needed|required|recommended)\b"]
MONITOR_PHRASES = [r"\bno action (?:is )?(?:needed|required)\b", r"\bjust (?:keep )?monitor", r"\bnothing to do\b"]


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def extract(text: str) -> list[tuple[str, float, str]]:
    """Return [(dimension, value_in_base_unit, token)] for every numeral in text, consuming spans pass by pass."""
    out: list[tuple[str, float, str]] = []
    buf = text

    def consume(rx, fn):
        nonlocal buf
        def repl(m):
            r = fn(m)
            if r:
                out.append((*r, m.group(0)))
            return " " * len(m.group(0))
        buf = rx.sub(repl, buf)

    consume(_SCI_X, lambda m: ("pc", _num(m.group(1)) * 10 ** int(m.group(2).translate(SUPMAP).replace(" ", ""))))
    consume(_SCI_E, lambda m: ("pc", _num(m.group(1)) * 10 ** int(m.group(2).replace("−", "-"))))
    consume(_SCI_10, lambda m: ("pc", 10 ** int((m.group(1) or m.group(2)).translate(SUPMAP))))
    consume(_PCT, lambda m: ("pct", _num(m.group(1)) / 100))
    consume(_SPEED, lambda m: ("speed", _num(m.group(1)) * (1000 if "km/s" in m.group(0) else 1)))

    def length(m):
        u = m.group(2)
        mult = 1000 if u.startswith("k") else 1
        return ("len", _num(m.group(1)) * mult)
    consume(_LEN, length)

    def tm(m):
        u = m.group(2)
        mult = 24 if u.startswith("d") else (1 / 60 if u.startswith("min") else 1)
        return ("time", _num(m.group(1)) * mult)
    consume(_TIME, tm)
    consume(_PLAIN, lambda m: ("plain", _num(m.group(0))))
    return out


def _close(a: float, b: float, tol: float = 0.01) -> bool:
    if a == b:
        return True
    return abs(a - b) <= tol * max(abs(a), abs(b))


def allowed_strings(facts: dict) -> list[str]:
    out: list[str] = []
    for k, v in facts.items():
        if k == "history":
            continue
        if isinstance(v, (list, tuple)):
            out += [str(x) for x in v if isinstance(x, (str, int, float))]
        out.append(render(facts, k))
    hl = history_line(facts)
    if hl:
        out += hl.split(" | ")
    out.append(facts.get("event_key", ""))
    return sorted({s for s in out if s and s != "n/a"}, key=len, reverse=True)


def allowed_numbers(facts: dict) -> list[tuple[str, float, str]]:
    nums: list[tuple[str, float, str]] = []
    for k in facts:
        if k == "history":
            continue
        for dim, val, tok in extract(render(facts, k)):
            nums.append((dim, val, k))
    for dim, val, tok in extract(history_line(facts)):
        nums.append((dim, val, "history"))
    return nums


def verify(text: str, facts: dict, *, cited: list[str] | None = None, verdict_action: str | None = None,
           unknown_placeholders: list[str] | None = None) -> tuple[bool, str | None]:
    if unknown_placeholders:
        return False, f"unknown fact key {unknown_placeholders[0]}"
    for k in cited or []:
        if k not in facts:
            return False, f"cited unknown fact key {k}"
    if PLACEHOLDER.search(text):
        return False, "unrendered placeholder"
    low = text.lower()
    dup = re.search(r"\d\s*(m|km|h|d|%)\s+(m|km|h|d|days?|hours?|met(?:re|er)s?|kilomet(?:re|er)s?)\b", low)
    if dup:
        return False, f"duplicated unit '{dup.group(0)}'"
    # Pc as a percentage (checked on raw text before stripping)
    if re.search(r"\bpc\b[^.;]{0,40}?\d+(?:\.\d+)?\s*(?:%|percent)", low) or \
            re.search(r"\d+(?:\.\d+)?\s*(?:%|percent)\s*(?:collision probability|probability of collision|pc\b)", low):
        return False, "Pc shown as a percentage"
    # action contradicting the verdict
    for rx in ALWAYS_FORBIDDEN:
        if re.search(rx, low):
            return False, f"forbidden action phrase /{rx}/"
    if verdict_action and verdict_action != "PREPARE_MANOEUVRE":
        for rx in PREPARE_PHRASES:
            m = re.search(rx, low)
            if m and not re.search(r"(?:options|go/no-go|not|no need|without)", low[max(0, m.start() - 40): m.end() + 40]):
                return False, f"action '{m.group(0)}' contradicts verdict action {verdict_action}"
    if verdict_action in ("PREPARE_MANOEUVRE", "REQUEST_TRACKING"):
        for rx in MONITOR_PHRASES:
            m = re.search(rx, low)
            if m:
                return False, f"action '{m.group(0)}' contradicts verdict action {verdict_action}"
    # strip verbatim fact strings, rule ids and harmless tokens, then check every remaining numeral
    stripped = text
    for s in allowed_strings(facts):
        if len(s) < 4 or not re.search(r"[A-Za-z]", s):
            continue  # bare numbers are checked numerically below, never stripped as text
        stripped = re.sub(r"(?<![0-9.,])" + re.escape(s) + r"(?![0-9])", " ", stripped)
    stripped = re.sub(r"\b[A-Z]{1,2}-[A-Z0-9-]+\b", " ", stripped)  # rule ids like R-MISS-CRIT, Q-COV-DEFAULT
    stripped = re.sub(r"\b[23]D\b", " ", stripped)
    stripped = re.sub(r"\b(?:App\.|Vol)\s*[A-Z]?\d*(?:\.\d+)?", " ", stripped)
    stripped = re.sub(r"(?:train|test|adhoc):\d+", " ", stripped)
    allowed = allowed_numbers(facts)
    pc_vals = [v for d, v, k in allowed if d == "pc" and k in PC_KEYS | {"history"}]
    pct_vals = [v for d, v, k in allowed if d == "pct" and k in PCT_KEYS]
    for dim, val, tok in extract(stripped):
        if dim == "pct" and any(_close(val, p) for p in pc_vals) and not any(_close(val, p) for p in pct_vals):
            return False, f"Pc shown as a percentage ({tok.strip()})"
        if dim == "pc" and any(_close(val, p) for p in pct_vals) and not any(_close(val, p) for p in pc_vals):
            return False, f"forecast written in scientific notation ({tok.strip()})"
        ok = False
        for adim, aval, _k in allowed:
            if dim == "plain":
                if _close(val, aval):
                    ok = True
                    break
            elif adim == dim and _close(val, aval):
                ok = True
                break
        if not ok:
            return False, f"number not in facts: {tok.strip()}"
    return True, None
