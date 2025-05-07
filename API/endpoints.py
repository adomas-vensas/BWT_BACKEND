import sys
sys.path.append("..") 


from fastapi import APIRouter, status
from .DTOs.CalculationParamsRequest import CalculationParamsRequest
from vortex_induced_vibration import VortexSimulation

router = APIRouter()

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
