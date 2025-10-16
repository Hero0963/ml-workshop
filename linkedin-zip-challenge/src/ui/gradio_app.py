# src/ui/gradio_app.py
import pprint

import gradio as gr
import requests

from src.core.solvers.a_star import solve_puzzle_a_star
from src.core.solvers.cp import solve_puzzle_cp
from src.core.solvers.dfs import solve_puzzle as solve_puzzle_dfs
from src.core.tests.conftest import puzzle_04_data, puzzle_04_layout
from src.settings import get_settings

# --- Settings and API Config ---
settings = get_settings()
API_BASE_URL = f"http://127.0.0.1:{settings.app_port}"

# --- Solver Mapping ---
SOLVERS = {
    "DFS": solve_puzzle_dfs,
    "A* (heapq)": solve_puzzle_a_star,
    "CP-SAT": solve_puzzle_cp,
}

# Use a puzzle from conftest as the default layout
DEFAULT_LAYOUT = pprint.pformat(puzzle_04_layout)
# Extract the corresponding walls and format them as a string
DEFAULT_WALLS = pprint.pformat(puzzle_04_data.get("walls", set()))


# --- UI Backend Functions ---


def echo_from_api(message: str) -> str:
    """Calls the backend's /api/echo endpoint and returns the result."""
    if not message:
        return "(Please enter a message)"
    try:
        response = requests.post(f"{API_BASE_URL}/api/echo", json={"message": message})
        response.raise_for_status()
        return response.json().get("response", "Invalid response format")
    except requests.exceptions.RequestException as e:
        return f"API call failed: {e}"


def solve_puzzle_ui(puzzle_layout_str: str, walls_str: str, solver_name: str):
    """Calls the backend solver API and returns HTML for the GIF and final image."""
    if not puzzle_layout_str.strip():
        error_html = "<p style='color:red;'>Error: Puzzle layout cannot be empty.</p>"
        return error_html, error_html

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/solver/solve",
            json={
                "puzzle_layout_str": puzzle_layout_str,
                "walls_str": walls_str,
                "solver_name": solver_name,
            },
            timeout=120,  # Set a timeout for potentially long-running solver calls
        )
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        data = response.json()
        solution_path_text = data.get("solution_path")
        gif_b64 = data.get("solution_gif_b64")
        final_image_b64 = data.get("solution_final_image_b64")

        gif_html = f"<p>{solution_path_text}</p>"
        final_image_html = f"<p>{solution_path_text}</p>"

        if gif_b64:
            gif_html = f"<img src='data:image/gif;base64,{gif_b64}' alt='Solution Animation' />"
        if final_image_b64:
            final_image_html = f"<img src='data:image/png;base64,{final_image_b64}' alt='Final Solution' />"

        return gif_html, final_image_html

    except requests.exceptions.Timeout:
        error_html = (
            "<p style='color:red;'>Error: The request to the solver API timed out.</p>"
        )
        return error_html, error_html
    except requests.exceptions.RequestException as e:
        error_detail = "Unknown error."
        if e.response is not None:
            try:
                error_detail = e.response.json().get("detail", error_detail)
            except ValueError:
                error_detail = e.response.text
        error_html = f"<p style='color:red;'>API call failed: {error_detail}</p>"
        return error_html, error_html


# --- Gradio App Layout ---
with gr.Blocks() as demo:
    gr.Markdown("## Zip Puzzle Solver UI")

    with gr.Tab("Echo Test"):
        gr.Markdown(
            "A simple interface to test the backend FastAPI's `/api/echo` endpoint."
        )
        with gr.Row():
            echo_input = gr.Textbox(
                label="Input Message", placeholder="Enter any text here..."
            )
            echo_output = gr.Textbox(label="Response from API", interactive=False)
        echo_button = gr.Button("Send to API")

    with gr.Tab("Puzzle Solver naive version"):
        gr.Markdown(
            "Enter the puzzle layout and walls as Python literals and select a solver."
        )
        with gr.Row():
            with gr.Column(scale=1):
                puzzle_input = gr.Textbox(
                    label="Puzzle Layout",
                    lines=10,
                    value=DEFAULT_LAYOUT,
                )
                walls_input = gr.Textbox(
                    label="Walls",
                    lines=5,
                    value=DEFAULT_WALLS,
                )
                solver_dropdown = gr.Dropdown(
                    label="Select Solver", choices=list(SOLVERS.keys()), value="DFS"
                )
                solve_button = gr.Button("Solve Puzzle", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Animated Solution")
                solution_gif_html = gr.HTML()
                gr.Markdown("### Final Result")
                solution_final_html = gr.HTML()

    with gr.Tab("Puzzle Solver interact version"):
        gr.Markdown("This feature is under construction.")

    # --- Event Handlers ---
    solve_button.click(
        fn=solve_puzzle_ui,
        inputs=[puzzle_input, walls_input, solver_dropdown],
        outputs=[solution_gif_html, solution_final_html],
    )
    echo_button.click(fn=echo_from_api, inputs=echo_input, outputs=echo_output)
