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

    try:
        sim: VortexSimulation | None = None

        while True:
            message = await ws.receive_text()
            data = json.loads(message)

            if data["type"] == "init_params":
                params = CalculationParamsRequest(**data["body"])
                sim = VortexSimulation(D = params.cylinderDiameter,
                                       U0 = params.windSpeed,
                                       RE = params.reynoldsNumber,
                                       UR = params.reducedVelocity,
                                       MR = params.massRatio,
                                       DR = params.dampingRatio,
                                       NX=params.nx,
                                       NY=params.ny)
                break


        while True:
            f, rho, u, d, v, a, h = sim.step()
            
            d0 = jax.device_get(d[0] / sim.D).astype(jnp.float32)
            d1 = jax.device_get(d[1] / sim.D).astype(jnp.float32)

            curlT = post.calculate_curl(u).T
            u_host = jax.device_get(curlT)

            payload = d0.tobytes() + d1.tobytes() + u_host.tobytes()
            await ws.send_bytes(payload)
            await asyncio.sleep(0.001)

    except Exception as e:
        print(e)
    finally:
        if ws.client_state.name == "DISCONNECTED":
            await ws.close()