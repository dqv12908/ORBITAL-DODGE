"""Shared helpers for backend tests (kept out of conftest.py so the engine's tests stay independent)."""
from __future__ import annotations

import asyncio
import functools

from app.settings import Settings


def mock_settings(tmp_path) -> Settings:
    return Settings(mock_ai=True, llm_enabled=False, cache_dir=tmp_path / "cache", db_path=tmp_path / "t.sqlite")


@functools.lru_cache(maxsize=1)
def shared_engine():
    from app.engine_adapter import Engine
    from app.settings import get_settings
    e = Engine(get_settings())
    e.load()
    return e


def make_sim(tmp_path, db_name="t.sqlite"):
    from app.simloop import Sim
    from app.sse import Bus
    from decisions.provider import DecisionService
    from ledger.ledger import Ledger
    from llm.explain import Explainer
    s = mock_settings(tmp_path)
    led = Ledger(tmp_path / db_name, config_hash="test", model_sha="test")
    bus = Bus(led)
    sim = Sim(s, bus, shared_engine(), DecisionService(s, providers=[]), Explainer(s), led)
    sim.reload_catalogue()
    return sim, bus, led


def run(coro):
    return asyncio.run(coro)


def run_scenario(tmp_path, sid, seed=1729, db_name="t.sqlite"):
    sim, bus, led = make_sim(tmp_path, db_name)
    envs = []
    bus.listeners.append(envs.append)

    async def go():
        await sim.load_scenario(sid, seed)
        await sim.run_headless()
    run(go())
    return sim, envs, led
