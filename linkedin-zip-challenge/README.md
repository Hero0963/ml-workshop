# LinkedIn Zip Puzzle Solver Challenge

This project is dedicated to exploring, developing, and comparing various algorithmic approaches to solve the "LinkedIn Zip" puzzle game. It includes a suite of solvers, procedural puzzle generation, and a modern web interface for interaction.

## About The "Zip" Puzzle

The game's objective is to draw a single, continuous path that visits every empty cell on a grid exactly once.

### Rules
*   The path must cover all visitable cells, and no cell can be visited more than once.
*   The path must connect all numbered waypoints in ascending order (1 → 2 → 3 ...).
*   The path cannot cross walls, which are marked by `|` or `—`.

## Features

*   **Multiple User Interfaces**:
    *   **Gradio UI**: A comprehensive interface for solving puzzles, generating new ones, and testing the API.
    *   **Svelte UI**: A modern frontend interface, with a Canvas-based "What You See Is What You Get" (WYSIWYG) editor at its core. Users can directly click cells on the canvas to edit, achieving a smooth and intuitive puzzle creation and modification workflow.
*   **RESTful API**: A backend powered by FastAPI, providing a programmatic interface to the solvers.
*   **Multiple Solver Algorithms**: A wide variety of solvers, from exact algorithms to metaheuristics.
*   **Procedural Puzzle Generation**: A powerful script and UI to generate vast datasets of new puzzles.
*   **Rich Visualization**: Generates detailed animated GIFs and static images of the solution process.


## Project Structure

The project structure has been organized to separate concerns between the core logic, the API application, and the user interfaces.

```
linkedin-zip-challenge/
├── src/
│   ├── app/                # FastAPI backend application
│   │   ├── routers/        # API endpoint definitions
│   │   └── main.py         # Main FastAPI app definition and startup
│   ├── core/               # Core puzzle logic and solvers
│   │   ├── puzzle_generation/ # Scripts for procedural puzzle generation
│   │   ├── solvers/        # All solver algorithm implementations
│   │   └── utils.py        # Shared utilities (parsing, visualization)
│   ├── custom_components/  # Contains the source for the Svelte UI
│   │   └── puzzle_editor/
│   │       └── frontend/   # Svelte source code
│   ├── ui/                 # Gradio UI application
│   │   └── gradio_app.py   # Gradio interface definition
│   └── settings.py         # Global application settings
├── .devcontainer/
│   ├── Dockerfile          # Multi-stage Dockerfile for PRODUCTION
│   └── Dockerfile.dev      # Dockerfile for DEVELOPMENT
├── .env                    # Environment variables (user-created)
├── docker-compose.yml      # Docker Compose file for PRODUCTION
├── docker-compose.dev.yml  # Docker Compose file for DEVELOPMENT
├── run_docker_dev.py       # Automation script for launching the dev environment
├── pyproject.toml          # Project dependencies for `uv`
└── README.md               # This file
```

## Getting Started

This project supports two primary workflows: a Docker-based environment (recommended for ease of use and consistency) and a manual local setup.

### Workflow 1: Docker Environment (Recommended)

The Docker setup has been designed to support both rapid development (with hot-reloading) and production-grade builds.

#### For Development (with Hot-Reloading)

This is the **recommended workflow for active development**. It uses `docker-compose.dev.yml` to launch two containers (backend and frontend dev server) with volume mounts, enabling instant code changes.

**Prerequisites:**
*   [Docker](https://www.docker.com/get-started) & [Docker Compose](https://docs.docker.com/compose/install/)

**Steps:**

1.  **Configure Environment:**
    Create a file named `.env` in the project root with the following content.   
    Details in  `.env.example`.  

2.  **Launch with One Command:**
    Run the provided automation script. It handles everything for you.
    ```bash
    python run_docker_dev.py
    ```

3.  **Access the Services:**
    After the application starts, you can access the following interfaces uniformly provided by the main service (port `7440`) through your browser:

    *   **Gradio Console**: `http://localhost:7440/ui`
        *   A comprehensive interface for solving puzzles, generating new ones, and testing the API.
    *   **Svelte Interactive Editor**: `http://localhost:7440/svelte-ui`
        *   A modern "What You See Is What You Get" editor, providing a smooth puzzle creation experience.

    ---
    **Tip for Developers:**
    When using development mode (`run_docker_dev.py`), the Svelte frontend will have an independent development server. To get instant **Hot-Reload** effects, please visit `http://localhost:5173` directly. In this mode, when backend Python code changes, the FastAPI service will also automatically reload.

#### For Production Simulation

This workflow builds a single, optimized, self-contained Docker image, just as you would for a real deployment.

**Steps:**

1.  **Use the Production Compose File:**
    Run the following command from the project root:
    ```bash
    # -f points to the production config file
    # --build forces a new build, running the multi-stage Dockerfile
    docker compose -f docker-compose.yml up --build -d
    ```

2.  **Access the Service:**
    Everything is served from a single port.
    *   **Gradio UI**: `http://localhost:7440/ui`
    *   **Svelte UI**: `http://localhost:7440/svelte-ui`

### Workflow 2: Manual Local Setup

This method runs the services directly on your machine without Docker.

**Prerequisites:**
*   Python 3.11 & `uv`
*   Node.js & `npm`

**Steps:**

1.  **Install Python Dependencies:**
    ```bash
    uv sync
    ```

2.  **Build Frontend:**
    The Svelte UI must be compiled into static files first.
    ```bash
    cd src/custom_components/puzzle_editor/frontend
    npm install
    npm run build
    cd ../../../../  # Return to project root
    ```

3.  **Configure Environment:**
    Create a `.env` file in the project root.  
    Details in  `.env.example`.

4.  **Run the Application:**
    Use `uv run` to execute the application within the virtual environment.
    ```bash
    uv run python -m src.app.main
    ```

5.  **Access the UIs:**
    *   **Gradio UI**: `http://localhost:7440/ui`
    *   **Svelte UI**: `http://localhost:7440/svelte-ui`

## Using the Interface

For more details and UI screenshots, please refer to the `illustrations/` directory.

The main interface is the Gradio UI, accessible at `/ui`. It provides several tabs:

*   **Generate Puzzle**: Create a new, random 6x6 puzzle. You can select the number of blocked cells.
*   **Puzzle Solver (Naive)**: Paste a puzzle layout and walls as text to solve it.
*   **Puzzle Solver (Interactive)**: A powerful WYSIWYG editor to create or edit puzzles by clicking on a grid, adding waypoints, obstacles, and walls.
*   **Echo Test**: A simple utility to confirm the backend API is responsive.

## Future Work

The following areas are planned for future development and are currently not integrated into the main application:

*   **Reinforcement Learning (RL) Solvers**: The code under `src/core/rl/` is an experimental framework for training RL agents to solve puzzles.
*   **Vision Language (VL) Models**: The code under `src/core/vl_models/` is an experimental area for parsing puzzles from images using multi-modal AI models.

## Development

### Running Tests
To run the entire test suite and generate a report:
```powershell
.
un_tests.bat
```
Test results and detailed logs will be saved in the `src/core/tests/reports/` directory.

### Development Log
For a detailed, chronological history of the project, please see the [Development Log](./dev_log.md).