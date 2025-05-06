from pydantic import BaseModel

class CalculationParamsRequest(BaseModel):
    reynoldsNumber: float | None = None
    reducedVelocity: float | None = None
    dampingRatio: float | None = None
    windSpeed: float | None = None
    cylinderDiameter: float | None = None

    

