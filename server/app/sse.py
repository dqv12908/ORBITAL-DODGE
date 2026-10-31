"""SSE event bus: envelope v1, monotonic seq, ring buffer for Last-Event-ID, 15 s ping, snapshot on connect.

The ledger is built by observing this bus (reference pattern): every event type with a ledger kind is appended to
the hash chain *before* it is broadcast, and the envelope carries the resulting `ledger_seq`.
"""
from __future__ import annotations

import asyncio
import collections
from typing import Any, Callable

import structlog

from app.util import dumps, iso, wall_now

log = structlog.get_logger("sse")

EVENT_TYPES = [
    "state.snapshot", "clock.tick", "scenario.loaded", "cdm.received", "risk.assessed", "forecast.updated",
    "policy.gated", "decision.requested", "decision.made", "policy.verdict", "approval.required",
    "human.action", "explanation.ready", "log.entry", "outcome.revealed", "policy.updated",
    "ledger.appended", "system.degraded",
]

LEDGER_KIND = {
    "scenario.loaded": "system",
    "cdm.received": "signal",
    "risk.assessed": "physics",
    "forecast.updated": "ml",
    "policy.gated": "rule",
    "decision.requested": "decision",
    "decision.made": "decision",
    "policy.verdict": "rule",
    "approval.required": "system",
    "human.action": "human",
    "explanation.ready": "llm",
    "outcome.revealed": "outcome",
    "policy.updated": "human",
    "system.degraded": "system",
}

RING = 2000
PING_S = 15.0


class Bus:
    def __init__(self, ledger=None):
        self.ledger = ledger
        self.seq = 0
        self.ring: collections.deque[dict] = collections.deque(maxlen=RING)
        self.clients: set[asyncio.Queue] = set()
        self.snapshot_fn: Callable[[], dict] | None = None
        self.sim_time_fn: Callable[[], str] = lambda: iso(wall_now())
        self.listeners: list[Callable[[dict], None]] = []
        self.ledger_refs_fn: Callable[[str, dict], dict] = lambda t, d: {}
        self.emit_ledger_appended = True

    # ------------------------------------------------------------------
    def publish(self, type_: str, data: dict, *, event_key: str | None = None, sim_time: str | None = None,
                actor: str | None = None, refs: dict | None = None, ledger: bool = True) -> dict:
        assert type_ in EVENT_TYPES, type_
        sim_time = sim_time or self.sim_time_fn()
        ledger_seq = None
        led = None
        kind = LEDGER_KIND.get(type_)
        if ledger and kind and self.ledger is not None:
            try:
                r = dict(self.ledger_refs_fn(type_, data))
                r.update(refs or {})
                led = self.ledger.append(kind, {"type": type_, "data": data}, event_key=event_key,
                                         actor=actor or "system", sim_time=sim_time, refs=r, type_=type_)
                ledger_seq = led["seq"]
            except Exception as e:  # ledger must never take the pipeline down
                log.error("ledger_append_failed", error=str(e), type=type_)
        env = self._envelope(type_, data, event_key, sim_time, ledger_seq)
        self._broadcast(env)
        if led and self.emit_ledger_appended:
            self._broadcast(self._envelope("ledger.appended", {"seq": led["seq"], "hash": led["hash"], "kind": led["kind"],
                                                                "type": type_}, event_key, sim_time, None))
        return env

    def _envelope(self, type_, data, event_key, sim_time, ledger_seq) -> dict:
        self.seq += 1
        return {"v": 1, "seq": self.seq, "type": type_, "at": iso(wall_now()), "sim_time": sim_time,
                "event_key": event_key, "ledger_seq": ledger_seq, "data": data}

    def _broadcast(self, env: dict) -> None:
        self.ring.append(env)
        for fn in list(self.listeners):
            try:
                fn(env)
            except Exception as e:
                log.warning("listener_failed", error=str(e))
        msg = self.format(env)
        for q in list(self.clients):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # slow client: drop it; it reconnects with Last-Event-ID
                self.clients.discard(q)

    @staticmethod
    def format(env: dict) -> str:
        return f"id: {env['seq']}\ndata: {dumps(env)}\n\n"

    def snapshot_envelope(self) -> dict:
        state = self.snapshot_fn() if self.snapshot_fn else {}
        return {"v": 1, "seq": max(self.seq, 1), "type": "state.snapshot", "at": iso(wall_now()),
                "sim_time": self.sim_time_fn(), "event_key": None, "ledger_seq": None, "data": state}

    def since(self, last_id: int) -> list[dict] | None:
        """Events after last_id, or None if the buffer no longer covers that point (client must resync)."""
        if last_id >= self.seq:
            return []
        if not self.ring or self.ring[0]["seq"] > last_id + 1:
            return None
        return [e for e in self.ring if e["seq"] > last_id]

    async def stream(self, last_event_id: str | None, is_disconnected: Callable[[], Any]):
        q: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self.clients.add(q)
        try:
            yield "retry: 2000\n\n"
            replay = None
            if last_event_id:
                try:
                    replay = self.since(int(last_event_id))
                except ValueError:
                    replay = None
            if replay is None:
                yield self.format(self.snapshot_envelope())
            else:
                for env in replay:
                    yield self.format(env)
            while True:
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=PING_S)
                    yield msg
                except asyncio.TimeoutError:
                    if await is_disconnected():
                        break
                    yield ": ping\n\n"
        finally:
            self.clients.discard(q)
