import sys
sys.path.append("..") 


from fastapi import APIRouter, status
from .DTOs.CalculationParamsRequest import CalculationParamsRequest
from vortex_induced_vibration import VortexSimulation
from . import websockets

router = APIRouter()