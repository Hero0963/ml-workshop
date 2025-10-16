# src/app/routers/echo.py
from fastapi import APIRouter

from src.app.schemas.echo import EchoRequest

# Create a new router object
router = APIRouter()


@router.post("/echo")
def echo(request: EchoRequest):
    """Echoes the received message back to the client."""
    return {"response": f"Echo: {request.message}"}
