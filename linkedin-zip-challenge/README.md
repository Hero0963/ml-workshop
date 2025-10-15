# Zip Puzzle Solver Challenge

This project is dedicated to exploring, developing, and comparing various algorithmic approaches to solve the "Zip" puzzle game. It includes a suite of solvers ranging from exact algorithms to metaheuristics and a framework for procedural puzzle generation.

The project is currently pivoting towards a service-oriented architecture to provide a user-facing application for puzzle solving.

## About The "Zip" Puzzle

The game's objective is to draw a single, continuous path that visits every empty cell on a grid exactly once.

### Rules
*   The path must cover all visitable cells, and no cell can be visited more than once.
*   The path must connect all numbered waypoints in ascending order (1 → 2 → 3 ...).
*   The path cannot cross walls, which are marked by `|` or `—`.

## Features

### Multiple Solver Algorithms
The project implements a wide variety of solvers, categorized as follows:

#### 1. Exact Solvers
These algorithms guarantee finding the optimal (and only) solution, though they may be computationally expensive for large or complex puzzles.
*   **Depth-First Search (DFS)**: A backtracking-based solver that exhaustively explores all possible paths.
*   **A* Search**: A more efficient graph traversal algorithm that uses a heuristic (Manhattan distance) to guide its search.
*   **Constraint Programming (CP-SAT)**: Models the puzzle as a constraint satisfaction problem using Google's OR-Tools to find a valid solution.

#### 2. Metaheuristic Solvers
These algorithms use probabilistic and optimization techniques to find high-quality solutions quickly, which is especially useful for very large or complex puzzles where an exact solution is infeasible to find in a reasonable amount of time.
*   **Monte Carlo Search**
*   **Simulated Annealing (SA)**
*   **Genetic Algorithm (GA)**
*   **Tabu Search (TS)**
*   **Particle Swarm Optimization (PSO)**

### Procedural Puzzle Generation
A powerful, multiprocessing-capable script (`src/core/puzzle_generator.py`) can generate vast datasets of new puzzles with guaranteed solutions.

### Solution Visualization
The project includes utilities to visualize solutions in various ways:
*   Static console-based grid printing.
*   Step-by-step console-based animation.
*   Exporting solution animations as high-quality GIFs.

## Current Status & Next Steps

The project's immediate focus is shifting from pure algorithmic development to building a user-facing service.

*   **New Direction**: To build a web service using **FastAPI** for the backend logic and **Gradio** for an interactive user interface. This will allow users to upload their own puzzles and receive solutions.
*   **Paused Development**: An exploration into using **Reinforcement Learning (RL)** to solve the puzzles has been conducted. This work is currently **on hold** pending further research. The detailed progress and challenges are documented in the development log.

## Project Structure
```
linkedin-zip-challenge/
├── src/
│   ├── core/
│   │   ├── solvers/         # All solver algorithm implementations (DFS, A*, GA, etc.)
│   │   ├── rl/              # (Paused) Reinforcement Learning framework
│   │   ├── tests/           # Pytest unit tests for all components
│   │   ├── puzzle_generator.py # Procedural puzzle generator
│   │   └── utils.py         # Shared utilities (parsing, visualization, etc.)
├── logs/                    # Directory for runtime logs
├── models/                  # Saved models (e.g., for RL)
├── puzzle_dataset/          # Generated puzzle datasets
├── .venv/                   # Project-specific virtual environment
├── pyproject.toml           # Project dependencies for `uv`
├── dev_log.md               # Detailed development and decision log
├── run_tests.bat            # Script to run the test suite
└── README.md                # This file
```

## Getting Started

### Prerequisites
*   Python 3.11
*   [uv](https://github.com/astral-sh/uv) (for package management)

### Installation
Clone the repository and sync the environment using `uv`:
```bash
# This will install all dependencies from pyproject.toml and uv.lock
uv sync
```

### Running Tests
To run the entire test suite and generate a report:
```powershell
.\run_tests.bat
```
Test results and detailed logs will be saved in the `src/core/tests/reports/` directory.

## Development Log

For a detailed, chronological history of the project's development, including technical challenges, experiments, and architectural decisions, please see the [Development Log](./dev_log.md).