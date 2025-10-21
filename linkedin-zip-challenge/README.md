# Zip Puzzle Solver Challenge

This project is dedicated to exploring, developing, and comparing various algorithmic approaches to solve the "Zip" puzzle game. It includes a suite of solvers, procedural puzzle generation, and a modern web interface for interaction.

## About The "Zip" Puzzle

The game's objective is to draw a single, continuous path that visits every empty cell on a grid exactly once.

### Rules
*   The path must cover all visitable cells, and no cell can be visited more than once.
*   The path must connect all numbered waypoints in ascending order (1 → 2 → 3 ...).
*   The path cannot cross walls, which are marked by `|` or `—`.

## Features

*   **Interactive Web UI**: Built with Gradio, allowing users to easily input puzzles and visualize solutions.
*   **RESTful API**: A backend powered by FastAPI, providing a programmatic interface to the solvers.
*   **Multiple Solver Algorithms**: A wide variety of solvers, from exact algorithms to metaheuristics.
*   **Procedural Puzzle Generation**: A powerful script to generate vast datasets of new puzzles, located in `src/core/puzzle_generation/`.
*   **Rich Visualization**: Generates detailed animated GIFs and static images of the solution process.

### Solver Algorithms
- **Exact Solvers**: DFS, A* Search, Constraint Programming (CP-SAT).
- **Metaheuristic Solvers**: Monte Carlo, Simulated Annealing, Genetic Algorithm, Tabu Search, PSO, and more.

## Project Structure

```
linkedin-zip-challenge/
├── src/
│   ├── app/                   # FastAPI backend application
│   │   ├── routers/           # API endpoint routers (e.g., solver, echo)
│   │   ├── schemas/           # Pydantic data models
│   │   └── main.py            # Main FastAPI app definition and startup
│   ├── core/                  # Core puzzle logic and solvers
│   │   ├── puzzle_generation/ # Scripts for procedural puzzle generation
│   │   ├── solvers/           # All solver algorithm implementations
│   │   ├── tests/             # Pytest unit tests for all components
│   │   └── utils.py           # Shared utilities (parsing, visualization)
│   ├── ui/                    # Gradio UI application
│   │   └── gradio_app.py      # Gradio interface definition
│   └── settings.py            # Global application settings
├── .env                       # Environment variables (e.g., APP_PORT)
├── pyproject.toml             # Project dependencies for `uv`
└── README.md                  # This file
```

## Getting Started

### Running with Docker (Recommended)

For the most streamlined setup, you can run the entire development environment using Docker. This method automatically builds the necessary container images, starts the backend and frontend services, and handles all internal networking.

**Prerequisites:**
*   [Docker](https://www.docker.com/get-started)
*   [Docker Compose](https://docs.docker.com/compose/install/)

**Launch with one command:**

Simply run the provided Python script from the project root:

```bash
python run_docker_dev.py
```

This script will:
1.  Stop any old containers.
2.  Build and start the FastAPI backend and Svelte frontend services.
3.  Automatically start the main application process inside the backend container.

Once the script is finished, the Gradio UI will be accessible at `http://127.0.0.1:7440` and the Svelte UI at `http://localhost:5173` (or your configured ports).

### Manual Setup

If you prefer to run the application without Docker, follow these steps.

### Prerequisites
*   Python 3.11
*   [uv](https://github.com/astral-sh/uv) (for package management)

### Installation

1.  **Clone the repository**

2.  **Create a virtual environment and install dependencies:**
    ```bash
    # This will install all dependencies from pyproject.toml and uv.lock
    uv sync
    ```

### Running the Web Application

1.  **(Optional) Configure the port:**
    Create a `.env` file in the project root and add the following line to change the default port (8008):
    ```
    APP_PORT=7440
    ```

2.  **Launch the server:**
    Run the following command from the project root directory:
    ```bash
    python -m src.app.main
    ```

3.  **Access the UI:**
    Open your web browser and navigate to `http://127.0.0.1:7440` (or your configured port).

### Alternative Svelte Frontend (Advanced)

In addition to the Gradio UI, a more advanced, experimental frontend is available, built with Svelte. It offers a richer, canvas-based WYSIWYG editor for creating puzzles.

**Note:** This frontend is a pure client and requires the main FastAPI backend to be running simultaneously.

1.  **Navigate to the frontend directory:**
    ```powershell
    cd src\custom_components\puzzle_editor\frontend
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```

3.  **Run the Svelte development server:**
    ```bash
    npm run dev
    ```

4.  **Access the Svelte UI:**
    The server will typically be available at `http://localhost:5173`. You can use this interface to create puzzles and solve them using the running backend.

### Using the Interface

The application provides three tabs. The most powerful one is the **"Puzzle Solver (Interactive)"** tab.

#### How to Use the Interactive Solver:

1.  **Define Grid Size**:
    -   Set the desired number of `Rows (m)` and `Columns (n)`.
    -   Click the **"Create Grid"** button. An empty grid editor will appear below.

2.  **Edit Puzzle Cells**:
    -   In the "Puzzle Cells" grid, click on cells to input values:
        -   **Numbers** (e.g., `1`, `2`, `12`): Define the waypoints the path must follow in order.
        -   **Obstacles** (e.g., `x`): Define blocked cells the path cannot enter.
    -   As you edit, the **"Live Puzzle Preview"** on the right will update in real-time.

3.  **Define Walls**:
    -   In the "Define Walls" section, specify the coordinates of two **adjacent** cells to place a wall between them.
    -   For example, to place a wall between `(0,0)` and `(0,1)`, enter `0`, `0` in the first row of boxes and `0`, `1` in the second.
    -   Click **"Add Wall"**. The wall will be added to the "Current Walls" list and will appear as a red line on the live preview.

4.  **Delete Walls**:
    -   In the "Current Walls" list, **click on the row** corresponding to the wall you wish to remove.
    -   Click the **"Delete Selected Wall"** button.

5.  **Solve the Puzzle**:
    -   Select your desired algorithm from the **"Select Solver"** dropdown.
    -   Click the **"Solve Puzzle"** button.
    -   The animated solution and final result will be displayed on the right.

6.  **Reset**:
    -   Click the **"重設 (Reset)"** button at any time to clear all inputs and start over.

## Development

### Running Tests
To run the entire test suite and generate a report:
```powershell
.\run_tests.bat
```
Test results and detailed logs will be saved in the `src/core/tests/reports/` directory.

### Development Log
For a detailed, chronological history of the project, please see the [Development Log](./dev_log.md).
