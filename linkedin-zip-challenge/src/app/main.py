# src/app/main.py
import gradio as gr
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
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


# --- Mount Static Frontend (Svelte UI) ---
def find_project_root_from_main(marker: str = "pyproject.toml") -> Path:
    """Finds the project root by searching upwards from this main.py file."""
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        if (current_path / marker).exists():
            return current_path
        current_path = current_path.parent
    raise FileNotFoundError(
        f"Could not find project root. Marker '{marker}' not found."
    )


try:
    PROJECT_ROOT = find_project_root_from_main()
    SVELTE_UI_DIR = (
        PROJECT_ROOT
        / "src"
        / "custom_components"
        / "puzzle_editor"
        / "frontend"
        / "dist"
    )

    if SVELTE_UI_DIR.exists() and SVELTE_UI_DIR.is_dir():
        app.mount(
            "/svelte-ui",
            StaticFiles(directory=SVELTE_UI_DIR, html=True),
            name="svelte-ui",
        )
        print(f"INFO: Svelte UI mounted successfully from '{SVELTE_UI_DIR}'.")
    else:
        # This is not a runtime error, just a warning for the developer.
        print(
            f"INFO: Svelte UI 'dist' directory not found at '{SVELTE_UI_DIR}'. The UI will not be available at /svelte-ui."
        )
        print(
            "      (This is expected if you haven't run 'npm run build' in the frontend directory.)"
        )

except FileNotFoundError:
    print("WARNING: Could not find project root. Svelte UI will not be mounted.")
# --- End Mount Static Frontend ---


# Include API routers
app.include_router(echo.router, prefix="/api", tags=["Echo"])
app.include_router(solver.router, prefix="/api/solver", tags=["Solver"])


@app.get("/", tags=["Root"])
def read_root():
    return {
        "message": "Welcome to the Zip Puzzle Solver API. Visit /docs for API documentation, /ui for the Gradio interface, or /svelte-ui for the advanced puzzle editor."
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
