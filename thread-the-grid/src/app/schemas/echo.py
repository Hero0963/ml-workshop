# src/app/schemas/echo.py
from pydantic import BaseModel


class EchoRequest(BaseModel):
    """Schema for the echo request body."""

    message: str


class EchoResponse(BaseModel):
    """Schema for the echo response body."""

    response: str


class HealthResponse(BaseModel):
    """Schema for the health check response."""

    status: str = "ok"
