import sys
sys.path.append("..") 

from fastapi import APIRouter, WebSocket
import asyncio
import jax
import jax.numpy as jnp
from calculations import post
from vortex_induced_vibration import VortexSimulation
import json
from .DTOs.CalculationParamsRequest import CalculationParamsRequest 

router = APIRouter(prefix="/stream")

@router.websocket("/calculate")
async def stream(ws: WebSocket):
    await ws.accept()

    # 1) grab the init message
    init_msg = await ws.receive_json()
    if init_msg.get("type") != "init_params":
        await ws.close(code=1003)
        return

    p0 = CalculationParamsRequest(**init_msg["body"])
    sim = VortexSimulation(
        D   = p0.cylinderDiameter,
        U0  = p0.windSpeed,
        RE  = p0.reynoldsNumber,
        UR  = p0.reducedVelocity,
        MR  = p0.massRatio,
        DR  = p0.dampingRatio,
        NX  = p0.nx,
        NY  = p0.ny,
    )

    # 2) set up a queue and background task to catch all update_params
    update_q: asyncio.Queue[dict] = asyncio.Queue()

    async def listener():
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "update_params":
                    await update_q.put(msg["body"])
        except Exception:
            pass  # socket closed or error

    listener_task = asyncio.create_task(listener())

    try:
        # 3) main simulation loop
        while True:
            # drain *all* pending updates before stepping
            while True:
                try:
                    upd_body = update_q.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    p = CalculationParamsRequest(**upd_body)
                    sim = VortexSimulation(
                        D   = p.cylinderDiameter or sim.D,
                        U0  = p.windSpeed        or sim.U0,
                        RE  = p.reynoldsNumber   or sim.RE,
                        UR  = p.reducedVelocity  or sim.UR,
                        MR  = p.massRatio        or sim.MR,
                        DR  = p.dampingRatio     or sim.DR,
                        NX  = p.nx               or sim.NX,
                        NY  = p.ny               or sim.NY,
                    )

            # now step once with the (possibly new) sim
            f, rho, u, d, v, a, h = sim.step()
            d0 = jax.device_get(d[0] / sim.D).astype(jnp.float32)
            d1 = jax.device_get(d[1] / sim.D).astype(jnp.float32)
            curlT = post.calculate_curl(u).T
            u_host = jax.device_get(curlT)

            payload = d0.tobytes() + d1.tobytes() + u_host.tobytes()
            await ws.send_bytes(payload)

            # give the event loop a breather
            await asyncio.sleep(0.001)

    finally:
        listener_task.cancel()
        await ws.close()
