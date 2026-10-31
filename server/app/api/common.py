from __future__ import annotations

import orjson
from fastapi import Request
from fastapi.responses import Response

from app.errors import ApiError
from app.util import canonical, dumps, sha256


async def json_body(request: Request, required: bool = True) -> dict:
    raw = await request.body()
    if not raw:
        if required:
            raise ApiError(400, "BAD_REQUEST", "JSON body required")
        return {}
    try:
        d = orjson.loads(raw)
    except Exception:
        raise ApiError(400, "BAD_REQUEST", "Invalid JSON body")
    if not isinstance(d, dict):
        raise ApiError(400, "BAD_REQUEST", "JSON body must be an object")
    return d


def jresp(obj, status: int = 200, headers: dict | None = None) -> Response:
    return Response(dumps(obj), status_code=status, media_type="application/json", headers=headers)


def idempotent(ledger, request: Request, scope: str, body: dict, fn):
    """Run fn() once per Idempotency-Key. Same key + same body -> stored response; other body -> 409.

    A missing key is tolerated (processed normally, header X-Idempotency: absent) to keep the demo robust.
    """
    key = request.headers.get("Idempotency-Key")
    if not key:
        status, obj = fn()
        return jresp(obj, status, {"X-Idempotency": "absent"})
    rh = sha256(canonical({"scope": scope, "body": body}))
    row = ledger.idem_get(key)
    if row is not None:
        if row["request_hash"] != rh:
            raise ApiError(409, "IDEMPOTENCY_CONFLICT", "Idempotency-Key reused with a different request",
                           "use a new key for a new request")
        return Response(row["response_json"], status_code=row["status"], media_type="application/json",
                        headers={"X-Idempotency": "replayed"})
    status, obj = fn()
    if status < 500:
        ledger.idem_put(key, scope, rh, dumps(obj), status)
    return jresp(obj, status, {"X-Idempotency": "stored"})
