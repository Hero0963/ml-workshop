# src/app/main.py
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.routers import echo, solver
from src.ui.gradio_app import demo as gradio_app

# Create FastAPI instance
app = FastAPI(
    title="Zip Puzzle Solver API + Gradio UI",
    description="An API to solve Zip puzzles, with both a Gradio UI and support for a separate frontend client.",
    version="2.0.0",
)

# Configure CORS (Cross-Origin Resource Sharing)
# This allows the new standalone frontend (running on a different port)
# to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, for development.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods.
    allow_headers=["*"],  # Allows all headers.
)


# Include API routers
app.include_router(echo.router, prefix="/api", tags=["Echo"])
app.include_router(solver.router, prefix="/api/solver", tags=["Solver"])


@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Welcome to the Zip Puzzle Solver API. Visit /docs for API documentation or /ui for the Gradio interface."
    }


# Mount the Gradio UI to the /ui path
# This keeps the original UI accessible while allowing the root path for pure API usage.
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


# The following block allows running the app directly with `python src/app/main.py`
if __name__ == "__main__":
    import uvicorn
    from src.settings import get_settings

    settings = get_settings()
    uvicorn.run(
        "src.app.main:app",
        host="0.0.0.0",
        port=settings.app_port,
        reload=True,
    )
