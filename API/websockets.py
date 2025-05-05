from fastapi import APIRouter, WebSocket
import asyncio
import jax
import jax.numpy as jnp
from calculations import post
import vortex_induced_vibration

router = APIRouter(prefix="/stream")

@router.websocket("/calculate")
async def stream(ws: WebSocket):
    await ws.accept()
    
    try:
        while True:
            f, rho, u, d, v, a, h = vortex_induced_vibration.updatePublic()
            
            d0 = jax.device_get(d[0] / vortex_induced_vibration.D).astype(jnp.float32)
            d1 = jax.device_get(d[1] / vortex_induced_vibration.D).astype(jnp.float32)

            curlT = post.calculate_curl(u).T
            u_host = jax.device_get(curlT)

            payload = d0.tobytes() + d1.tobytes() + u_host.tobytes()
            await ws.send_bytes(payload)
            await asyncio.sleep(0.001)
    except Exception:
        await ws.close()