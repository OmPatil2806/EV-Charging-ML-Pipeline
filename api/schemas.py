"""Pydantic request/response models for the inference API."""

from datetime import datetime

from pydantic import BaseModel, Field


class SessionInput(BaseModel):
    vehicle_model: str = Field(..., examples=["Tesla Model 3"])
    battery_capacity_kwh: float = Field(..., gt=0, examples=[60.0])
    charger_type: str = Field(..., examples=["Level 2"])
    charging_start_time: datetime = Field(..., examples=["2024-06-15T18:30:00"])
    charging_duration_hours: float = Field(..., gt=0, examples=[2.5])
    charging_rate_kw: float = Field(..., gt=0, examples=[11.0])
    charging_cost_usd: float = Field(..., ge=0, examples=[8.5])
    soc_start_pct: float = Field(..., ge=0, le=100, examples=[25.0])
    soc_end_pct: float = Field(..., ge=0, le=100, examples=[80.0])
    distance_since_last_charge_km: float = Field(..., ge=0, examples=[180.0])
    temperature_c: float = Field(..., examples=[18.0])
    vehicle_age_years: float = Field(..., ge=0, examples=[2.0])
    user_type: str = Field(..., examples=["Commuter"])


class PredictionResponse(BaseModel):
    predicted_energy_kwh: float
    model_used: str
