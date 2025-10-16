# src/app/schemas/echo.py
from pydantic import BaseModel


class EchoRequest(BaseModel):
    """Schema for the echo request body."""

    message: str
