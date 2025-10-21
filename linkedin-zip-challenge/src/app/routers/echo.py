# src/app/routers/echo.py
from fastapi import APIRouter
from src.app.schemas.echo import EchoRequest, EchoResponse, HealthResponse

router = APIRouter(
    prefix="/echo",
    tags=["echo"],
)


@router.post("/", response_model=EchoResponse)
async def post_echo(request: EchoRequest) -> EchoResponse:
    """Echoes the received message back to the client.

    - **request**: The input message.
    - **return**: The echoed message, prefixed with "Echo: ".
    """
    return EchoResponse(response=f"Echo: {request.message}")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check if the service is running."""
    return HealthResponse(status="ok")
