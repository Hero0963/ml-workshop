# src/ui/gradio_app.py
import ast
import pprint

import gradio as gr
import numpy as np
import pandas as pd
import requests
from loguru import logger
from PIL import Image, ImageDraw, ImageFont

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
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        solution_path_text = data.get("solution_path")
        gif_b64 = data.get("solution_gif_b64")
        final_image_b64 = data.get("solution_final_image_b64")

        gif_html = f"<p>{solution_path_text}</p>"
        if gif_b64:
            gif_html = f"<img src='data:image/gif;base64,{gif_b64}' alt='Solution Animation' />"

        final_image_html = f"<p>{solution_path_text}</p>"
        if final_image_b64:
            final_image_html = f"<img src='data:image/png;base64,{final_image_b64}' alt='Final Solution' />"

        return gif_html, final_image_html

    except requests.exceptions.RequestException as e:
        error_detail = f"API call failed: {e}"
        try:
            error_detail = e.response.json().get("detail", str(e))
        except (AttributeError, ValueError):
            pass
        error_html = f"<p style='color:red;'>{error_detail}</p>"
        return error_html, error_html


# --- Interactive Tab Functions ---


def _convert_df_to_puzzledata(puzzle_df: pd.DataFrame, walls: set):
    """Helper to convert UI data (DataFrames, sets) to a puzzle data dictionary for drawing."""
    if puzzle_df is None:
        return None

    def _convert_cell_value(cell):
        val = str(cell).strip().lower()
        if val == "x":
            return "xx"
        if val == "":
            return "  "
        return str(cell).strip()

    grid_layout = [
        [_convert_cell_value(cell) for cell in row] for row in puzzle_df.values.tolist()
    ]

    height, width = len(grid_layout), len(grid_layout[0]) if grid_layout else 0
    grid = [[0] * width for _ in range(height)]
    num_map = {}
    blocked_cells = set()

    for r, row_val in enumerate(grid_layout):
        for c, item in enumerate(row_val):
            if item.isdigit():
                grid[r][c] = int(item)
                if int(item) > 0:
                    num_map[int(item)] = (r, c)
            elif item == "xx":
                blocked_cells.add((r, c))

    return {
        "grid_size": (height, width),
        "grid": grid,
        "blocked_cells": blocked_cells,
        "walls": walls,
    }


def generate_preview_image(puzzle_df: pd.DataFrame, walls: set):
    """Draws the current state of the puzzle as a PIL Image."""
    puzzle_data = _convert_df_to_puzzledata(puzzle_df, walls)
    if not puzzle_data:
        return None

    height, width = puzzle_data["grid_size"]
    grid = puzzle_data["grid"]
    blocked_cells = puzzle_data["blocked_cells"]

    cell_size, margin = 50, 10
    img_width = width * cell_size + 2 * margin
    img_height = height * cell_size + 2 * margin

    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    img = Image.new("RGB", (img_width, img_height), "white")
    draw = ImageDraw.Draw(img)

    for r in range(height):
        for c in range(width):
            x0, y0 = margin + c * cell_size, margin + r * cell_size
            rect = [(x0, y0), (x0 + cell_size, y0 + cell_size)]

            if (r, c) in blocked_cells:
                draw.rectangle(rect, fill="black")
            else:
                draw.rectangle(rect, outline="black", fill="white")
                content = str(grid[r][c]) if grid[r][c] > 0 else ""
                if content:
                    draw.text((x0 + 20, y0 + 15), content, fill="black", font=font)

    for wall in walls:
        (r1, c1), (r2, c2) = wall
        if r1 == r2:  # Vertical wall
            line_x = margin + max(c1, c2) * cell_size
            line_y0, line_y1 = margin + r1 * cell_size, margin + (r1 + 1) * cell_size
            draw.line([(line_x, line_y0), (line_x, line_y1)], fill="red", width=3)
        else:  # Horizontal wall
            line_x0, line_x1 = margin + c1 * cell_size, margin + (c1 + 1) * cell_size
            line_y = margin + max(r1, r2) * cell_size
            draw.line([(line_x0, line_y), (line_x1, line_y)], fill="red", width=3)

    return img


def create_interactive_grids(m: int, n: int):
    m, n = int(m), int(n)
    if m <= 0 or n <= 0:
        return None, None
    puzzle_df = pd.DataFrame(np.full((m, n), ""), index=range(m), columns=range(n))
    initial_image = generate_preview_image(puzzle_df, set())
    return puzzle_df, initial_image


def solve_from_interactive_ui(puzzle_df: pd.DataFrame, walls: set, solver_name: str):
    def _convert_cell_value(cell):
        val = str(cell).strip().lower()
        if val == "x":
            return "xx"
        if val == "":
            return "  "
        return str(cell).strip()

    puzzle_list = [
        [_convert_cell_value(cell) for cell in row] for row in puzzle_df.values.tolist()
    ]
    puzzle_layout_str = pprint.pformat(puzzle_list)
    walls_str = pprint.pformat(walls) if walls else "set()"

    logger.debug("--- Interactive Solve Payload ---")
    logger.debug(f"Solver: {solver_name}")
    logger.debug(f"Puzzle Layout:\n{puzzle_layout_str}")
    logger.debug(f"Walls:\n{walls_str}")
    logger.debug("---------------------------------")

    return solve_puzzle_ui(puzzle_layout_str, walls_str, solver_name)


def _walls_set_to_df(walls: set) -> pd.DataFrame:
    if not walls:
        return pd.DataFrame(columns=["Wall"])
    sorted_walls = sorted(list(walls), key=lambda x: (x[0], x[1]))
    return pd.DataFrame([str(w) for w in sorted_walls], columns=["Wall"])


def add_wall(current_walls: set, r1, c1, r2, c2):
    try:
        c1_tuple, c2_tuple = (int(r1), int(c1)), (int(r2), int(c2))
        manhattan_distance = abs(c1_tuple[0] - c2_tuple[0]) + abs(
            c1_tuple[1] - c2_tuple[1]
        )
        if manhattan_distance != 1:
            gr.Warning("Wall coordinates must be adjacent.")
            return current_walls, _walls_set_to_df(current_walls)

        wall = tuple(sorted((c1_tuple, c2_tuple)))
        current_walls.add(wall)
        return current_walls, _walls_set_to_df(current_walls)
    except (ValueError, TypeError, AttributeError) as e:
        gr.Warning(f"Invalid coordinate. Please use numbers. Error: {e}")
        return current_walls, _walls_set_to_df(current_walls)


def store_selection(evt: gr.SelectData):
    return evt.index[0]


def delete_wall(current_walls: set, wall_list_df: pd.DataFrame, selected_index: int):
    if selected_index is None or selected_index >= len(wall_list_df):
        gr.Warning("Please select a wall from the list to delete.")
        return current_walls, wall_list_df, None

    wall_to_delete_str = wall_list_df.iloc[selected_index, 0]
    try:
        wall_to_delete = ast.literal_eval(wall_to_delete_str)
        current_walls.discard(wall_to_delete)
        return current_walls, _walls_set_to_df(current_walls), None
    except (ValueError, SyntaxError):
        gr.Warning("Could not parse the selected wall to delete.")
        return current_walls, wall_list_df, None


def reset_all():
    return {
        m_input: 6,
        n_input: 6,
        puzzle_grid: None,
        preview_image: None,
        solution_gif_html_interactive: None,
        solution_final_html_interactive: None,
        walls_state: set(),
        wall_list_df: _walls_set_to_df(set()),
        r1_in: None,
        c1_in: None,
        r2_in: None,
        c2_in: None,
        selected_wall_index: None,
    }


# --- Gradio App Layout ---
with gr.Blocks() as demo:
    gr.Markdown("## Zip Puzzle Solver UI")

    with gr.Tab("Echo Test"):
        gr.Markdown(
            "A simple interface to test the backend FastAPI's /api/echo endpoint."
        )
        with gr.Row():
            echo_input = gr.Textbox(
                label="Input Message", placeholder="Enter any text here..."
            )
            echo_output = gr.Textbox(label="Response from API", interactive=False)
        echo_button = gr.Button("Send to API")

    with gr.Tab("Puzzle Solver (Naive)"):
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
                solver_dropdown_naive = gr.Dropdown(
                    label="Select Solver", choices=list(SOLVERS.keys()), value="DFS"
                )
                solve_button_naive = gr.Button("Solve Puzzle", variant="primary")

            with gr.Column(scale=2):
                gr.Markdown("### Animated Solution")
                solution_gif_html_naive = gr.HTML()
                gr.Markdown("### Final Result")
                solution_final_html_naive = gr.HTML()

    with gr.Tab("Puzzle Solver (Interactive)"):
        walls_state = gr.State(set())
        selected_wall_index = gr.State(None)

        gr.Markdown("Create a puzzle grid interactively.")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Define Grid & Puzzle")
                with gr.Row():
                    m_input = gr.Number(label="Rows (m)", value=6, minimum=1, step=1)
                    n_input = gr.Number(label="Columns (n)", value=6, minimum=1, step=1)
                with gr.Row():
                    create_grid_btn = gr.Button("Create Grid")
                    reset_btn = gr.Button("重設 (Reset)")
                puzzle_grid = gr.Dataframe(
                    label="Puzzle Cells (Enter numbers or 'x')",
                    interactive=True,
                    datatype=["str"],
                )

                gr.Markdown("### 2. Define Walls")
                with gr.Row():
                    r1_in = gr.Number(label="R1", minimum=0, step=1)
                    c1_in = gr.Number(label="C1", minimum=0, step=1)
                with gr.Row():
                    r2_in = gr.Number(label="R2", minimum=0, step=1)
                    c2_in = gr.Number(label="C2", minimum=0, step=1)
                with gr.Row():
                    add_wall_btn = gr.Button("Add Wall")
                    delete_wall_btn = gr.Button("Delete Selected Wall")
                wall_list_df = gr.Dataframe(
                    headers=["Wall"],
                    datatype=["str"],
                    label="Current Walls",
                    interactive=False,
                )

            with gr.Column(scale=2):
                gr.Markdown("### Live Puzzle Preview")
                preview_image = gr.Image(label="Live Preview", interactive=False)

                gr.Markdown("### 3. Solve")
                solver_dropdown_interactive = gr.Dropdown(
                    label="Select Solver", choices=list(SOLVERS.keys()), value="DFS"
                )
                solve_button_interactive = gr.Button("Solve Puzzle", variant="primary")
                gr.Markdown("### Animated Solution")
                solution_gif_html_interactive = gr.HTML()
                gr.Markdown("### Final Result")
                solution_final_html_interactive = gr.HTML()

    # --- Event Handlers ---
    solve_button_naive.click(
        fn=solve_puzzle_ui,
        inputs=[puzzle_input, walls_input, solver_dropdown_naive],
        outputs=[solution_gif_html_naive, solution_final_html_naive],
    )
    echo_button.click(fn=echo_from_api, inputs=echo_input, outputs=echo_output)

    # Interactive Tab Handlers
    create_grid_btn.click(
        fn=create_interactive_grids,
        inputs=[m_input, n_input],
        outputs=[puzzle_grid, preview_image],
    )
    solve_button_interactive.click(
        fn=solve_from_interactive_ui,
        inputs=[puzzle_grid, walls_state, solver_dropdown_interactive],
        outputs=[solution_gif_html_interactive, solution_final_html_interactive],
    )
    reset_btn.click(
        fn=reset_all,
        inputs=[],
        outputs=[
            m_input,
            n_input,
            puzzle_grid,
            preview_image,
            solution_gif_html_interactive,
            solution_final_html_interactive,
            walls_state,
            wall_list_df,
            r1_in,
            c1_in,
            r2_in,
            c2_in,
            selected_wall_index,
        ],
    )
    add_wall_btn.click(
        fn=add_wall,
        inputs=[walls_state, r1_in, c1_in, r2_in, c2_in],
        outputs=[walls_state, wall_list_df],
    ).then(
        fn=generate_preview_image,
        inputs=[puzzle_grid, walls_state],
        outputs=[preview_image],
    )
    wall_list_df.select(
        fn=store_selection,
        inputs=[],
        outputs=[selected_wall_index],
    )
    delete_wall_btn.click(
        fn=delete_wall,
        inputs=[walls_state, wall_list_df, selected_wall_index],
        outputs=[walls_state, wall_list_df, selected_wall_index],
    ).then(
        fn=generate_preview_image,
        inputs=[puzzle_grid, walls_state],
        outputs=[preview_image],
    )
    puzzle_grid.input(
        fn=generate_preview_image,
        inputs=[puzzle_grid, walls_state],
        outputs=[preview_image],
    )
