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
*   **Procedural Puzzle Generation**: A powerful script to generate vast datasets of new puzzles.
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
    Create a `.env` file in the project root and add the following line to change the default port (8000):
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

## Development

### Running Tests
To run the entire test suite and generate a report:
```powershell
.\run_tests.bat
```
Test results and detailed logs will be saved in the `src/core/tests/reports/` directory.

### Development Log
For a detailed, chronological history of the project, please see the [Development Log](./dev_log.md).