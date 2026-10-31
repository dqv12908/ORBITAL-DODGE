from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.common import json_body, jresp
from app.errors import ApiError
from app.simloop import SPEEDS, SimBusy
from app.state import ctx

router = APIRouter(prefix="/api")


@router.get("/state")
async def get_state(request: Request):
    return jresp(ctx(request).sim.state())


@router.get("/scenarios")
async def scenarios(request: Request):
    sim = ctx(request).sim
    sim.reload_catalogue()
    return jresp({"scenarios": sim.cards(), "source": sim.catalogue_source,
                  "current": (sim.scenario or {}).get("id")})


@router.post("/sim/scenario")
async def load_scenario(request: Request):
    b = await json_body(request)
    sim = ctx(request).sim
    sid = b.get("id")
    seed = b.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ApiError(400, "BAD_REQUEST", "seed must be an integer")
    sim.reload_catalogue()
    valid = [s["id"] for s in sim.catalogue]
    if sid not in valid:
        raise ApiError(400, "BAD_REQUEST", f"unknown scenario {sid!r}", f"valid: {valid}", extra={"valid": valid})
    try:
        st = await sim.load_scenario(sid, seed)
    except SimBusy as e:
        raise ApiError(409, "SIM_BUSY", str(e), "retry in a few seconds", True)
    return jresp(st)


@router.post("/sim/speed")
async def speed(request: Request):
    b = await json_body(request)
    try:
        return jresp(ctx(request).sim.set_speed(b.get("speed")))
    except ValueError:
        raise ApiError(400, "BAD_REQUEST", f"speed must be one of {list(SPEEDS)}", extra={"valid": list(SPEEDS)})


@router.post("/sim/pause")
async def pause(request: Request):
    return jresp(ctx(request).sim.set_paused(True))


@router.post("/sim/resume")
async def resume(request: Request):
    return jresp(ctx(request).sim.set_paused(False))


@router.post("/sim/next-cdm")
async def next_cdm(request: Request):
    sim = ctx(request).sim
    if sim.scenario is None:
        raise ApiError(409, "NO_SCENARIO", "load a scenario first", "POST /api/sim/scenario")
    return jresp(await sim.next_cdm())
