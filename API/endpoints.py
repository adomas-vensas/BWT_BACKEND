from fastapi import APIRouter, status
from .DTOs.CalculationParamsRequest import CalculationParamsRequest
import vortex_induced_vibration

import sys
sys.path.append('../')

router = APIRouter()

@router.get("/params")
async def get_params():
    return {
        "NX": vortex_induced_vibration.NX,
        "NY": vortex_induced_vibration.NY,
        "maxDiameter": vortex_induced_vibration.D_MAX_PHYSICAL,
        "maxWindSpeed": vortex_induced_vibration.U_MAX_PHYSICAL
    }


@router.post("/params", status_code=status.HTTP_202_ACCEPTED, response_model=None)
async def set_params(params: CalculationParamsRequest):
    return
    # if params.reynoldsNumber is not None:
    #     vortex_induced_vibration.RE = params.reynoldsNumber
    
    # if params.reducedVelocity is not None:
    #     vortex_induced_vibration.UR = params.reducedVelocity

    # if params.dampingRatio is not None:
    #     vortex_induced_vibration.DR = params.dampingRatio

    # if params.windSpeed is not None:
    #     vortex_induced_vibration.U_PHYSICAL = params.windSpeed
    
    # if params.cylinderDiameter is not None:
    #     vortex_induced_vibration.D_PHYSICAL = params.cylinderDiameter
    
    # if params.massRatio is not None:
    #     vortex_induced_vibration.MR = params.massRatio
