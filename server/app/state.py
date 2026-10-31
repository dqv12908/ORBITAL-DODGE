"""Process-wide context (one process, --workers 1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request


@dataclass
class Ctx:
    settings: Any
    bus: Any
    ledger: Any
    engine: Any
    decisions: Any
    explainer: Any
    sim: Any
    started_wall: str = ""


def ctx(request: Request) -> Ctx:
    return request.app.state.ctx


def build_ctx(settings, *, db_path=None) -> Ctx:
    from app.engine_adapter import Engine
    from app.simloop import Sim
    from app.sse import Bus
    from app.util import canonical, iso, sha256, wall_now
    from decisions.provider import DecisionService
    from ledger.ledger import Ledger
    from llm.explain import Explainer

    engine = Engine(settings)
    engine.load()
    config_hash = sha256(canonical({"mode": settings.decisions_mode, "seed": settings.seed, "llm": settings.llm_mode}))[:16]
    ledger = Ledger(db_path or settings.sqlite_path, config_hash=config_hash, model_sha=engine.forecaster_version or "")
    bus = Bus(ledger)
    dec = DecisionService(settings)
    exp = Explainer(settings)
    sim = Sim(settings, bus, engine, dec, exp, ledger)
    sim.reload_catalogue()
    return Ctx(settings, bus, ledger, engine, dec, exp, sim, iso(wall_now()))
