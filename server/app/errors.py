"""JSON error envelope (design §6.7): {error: {code, message, hint, retryable}}."""
from __future__ import annotations

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

log = structlog.get_logger("api")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, hint: str | None = None, retryable: bool = False,
                 extra: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.hint, self.retryable, self.extra = status, code, message, hint, retryable, extra or {}


def err_body(code: str, message: str, hint: str | None = None, retryable: bool = False, **extra) -> dict:
    return {"error": {"code": code, "message": message, "hint": hint, "retryable": retryable, **extra}}


def install(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api(request: Request, exc: ApiError):
        return JSONResponse(err_body(exc.code, exc.message, exc.hint, exc.retryable, **exc.extra), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def _val(request: Request, exc: RequestValidationError):
        errs = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse(err_body("BAD_REQUEST", "invalid request", "see details", False, details=errs), status_code=400)

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(err_body(code, str(exc.detail), "see GET /api/health for the endpoint list" if exc.status_code == 404 else None),
                            status_code=exc.status_code)

    @app.exception_handler(Exception)
    async def _any(request: Request, exc: Exception):
        log.exception("unhandled", path=request.url.path, error=str(exc))
        return JSONResponse(err_body("INTERNAL", f"{type(exc).__name__}: {str(exc)[:200]}", None, True), status_code=500)
