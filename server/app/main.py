"""ORBITAL DODGE backend: FastAPI app, one process (--workers 1).

Run:  cd server && .venv/bin/uvicorn app.main:app --port 8000
"""
from __future__ import annotations

import asyncio
import contextlib
import logging

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import errors
from app.api import data, ops, sim
from app.settings import get_settings
from app.state import build_ctx

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
log = structlog.get_logger("main")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    ctx = await asyncio.to_thread(build_ctx, settings)
    app.state.ctx = ctx
    try:
        await ctx.sim.load_scenario(settings.boot_scenario if any(
            s["id"] == settings.boot_scenario for s in ctx.sim.catalogue) else ctx.sim.catalogue[0]["id"])
    except Exception as e:
        log.exception("boot_scenario_failed", error=str(e))
    try:
        await asyncio.to_thread(ctx.ledger.verify)
    except Exception:
        pass
    task = asyncio.create_task(ctx.sim.run(), name="simloop")
    log.info("ready", decisions=settings.decisions_mode, llm=settings.llm_mode, engine=ctx.engine.data_source)
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        ctx.ledger.close()


app = FastAPI(title="ORBITAL DODGE", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Idempotency"])
errors.install(app)
app.include_router(ops.router)
app.include_router(sim.router)
app.include_router(data.router)
from app.api import curves as _curves  # noqa: E402  (ggplot-style stats curves, additive)
app.include_router(_curves.router)


@app.api_route("/api/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"], include_in_schema=False)
async def api_404(rest: str):
    raise errors.ApiError(404, "NOT_FOUND", f"no endpoint /api/{rest}", "see GET /api/health for the endpoint list")


def _mount_game() -> None:
    from pathlib import Path

    from fastapi.staticfiles import StaticFiles
    dist = get_settings().repo_dir / "game" / "dist"
    if Path(dist).exists():
        app.mount("/", StaticFiles(directory=str(dist), html=True), name="game")


_mount_game()
