# src/app/main.py
import gradio as gr
import uvicorn
from fastapi import FastAPI

from src.app.routers import echo, solver
from src.settings import get_settings
from src.ui.gradio_app import demo as gradio_app

# Create FastAPI instance
app = FastAPI(
    title="Zip Puzzle Solver API",
    description="An API to solve Zip puzzles, with a Gradio UI.",
    version="0.1.0",
)

# Include API routers
app.include_router(echo.router, prefix="/api", tags=["Echo"])
app.include_router(solver.router, prefix="/api/solver", tags=["Solver"])


# Mount the Gradio UI to the root path
# This makes the Gradio interface available at the root URL
app = gr.mount_gradio_app(app, gradio_app, path="/")

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=True,
    )
