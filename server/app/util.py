"""Small shared helpers: canonical JSON, hashing, deterministic ids, time formatting."""
from __future__ import annotations

import hashlib
import math
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import orjson

NS = uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/dqv12908/ORBITAL-DODGE")


def _default(o: Any):
    if hasattr(o, "item"):  # numpy scalar
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, datetime):
        return iso(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not serialisable: {type(o)}")


def clean_json(o: Any) -> Any:
    """Replace NaN/inf with None recursively and convert numpy types, so JSON is always valid."""
    if isinstance(o, dict):
        return {str(k): clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if hasattr(o, "tolist") and not isinstance(o, (str, bytes)):
        return clean_json(o.tolist())
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except Exception:
            pass
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def canonical(o: Any) -> bytes:
    return orjson.dumps(clean_json(o), option=orjson.OPT_SORT_KEYS, default=_default)


def dumps(o: Any) -> str:
    return orjson.dumps(clean_json(o), default=_default).decode()


def sha256(b: bytes | str) -> str:
    if isinstance(b, str):
        b = b.encode()
    return hashlib.sha256(b).hexdigest()


def uid(name: str) -> str:
    return str(uuid.uuid5(NS, name))


def parse_ts(s: str | datetime) -> datetime:
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=timezone.utc)
    s = s.replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def iso(d: datetime) -> str:
    d = d.astimezone(timezone.utc)
    if d.microsecond:
        return d.strftime("%Y-%m-%dT%H:%M:%S.") + f"{d.microsecond // 1000:03d}Z"
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def wall_now() -> datetime:
    return datetime.now(timezone.utc)


def days(x: float) -> timedelta:
    return timedelta(seconds=round(float(x) * 86400.0))


def hhmm(d: datetime) -> str:
    return d.astimezone(timezone.utc).strftime("%H:%M")


def fnum(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v
