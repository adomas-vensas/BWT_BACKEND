from fastapi import APIRouter, WebSocket
import vortex_induced_vibration


router = APIRouter()

@router.get("/params")
async def get_params():
    return {
        "NX": vortex_induced_vibration.NX,
        "NY": vortex_induced_vibration.NY,

        "RE": vortex_induced_vibration.RE,
        "UR": vortex_induced_vibration.UR,
        "MR": vortex_induced_vibration.MR,
        "DR": vortex_induced_vibration.DR,
        "D_PHYSICAL": vortex_induced_vibration.D_PHYSICAL
    }
