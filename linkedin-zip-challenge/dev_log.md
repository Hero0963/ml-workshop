# Development Log

## 2025-10-04

### Expansion of Metaheuristic Solver Suite

-   **Simulated Annealing (SA) Solver:** Implemented `solve_puzzle_simulated_annealing` in a new `simulated_annealing.py` module. The development process uncovered a critical bug in the initial neighbor generation logic:
    -   An initial `2-opt` swap strategy, common in TSP-like problems, was found to produce non-contiguous paths (i.e., "jumps") on a grid. This bug was identified thanks to user feedback.
    -   The logic was corrected by replacing `2-opt` with a robust "truncate and regrow" strategy in the `_generate_neighbor_path` helper function, which guarantees path contiguity.
-   **Genetic Algorithm (GA) Solver:** Implemented `solve_puzzle_genetic_algorithm` in `genetic_algorithm.py`.
    -   To avoid the path contiguity issues inherent in traditional crossover operations, a pragmatic "no-crossover" variant was designed. 
    -   The implemented GA relies on elitism (carrying over the best solutions) and mutation (using the new `generate_neighbor_path` function) for reproduction and population evolution.
-   **Tabu Search (TS) Solver:** Implemented `solve_puzzle_tabu_search` in `tabu_search.py`.
    -   The solver uses a `collections.deque` with a fixed `maxlen` as an efficient short-term memory (the "tabu list").
    -   To save memory, hashes of path tuples (`hash(tuple(path))`) are stored in the tabu list instead of the paths themselves.
    -   An aspiration criterion is included. The logic for this criterion was significantly refined based on user feedback:
        -   A critical logical flaw in the initial implementation (`score > best_score`), where the condition would never be met for a tabu item, was identified by the user.
        -   The final, more flexible and effective implementation (`score >= aspiration_threshold * best_score`) was also proposed by the user, and the `aspiration_threshold` parameter was added accordingly.
-   **Particle Swarm Optimization (PSO) Solver:** Implemented a discrete adaptation of PSO in `particle_swarm_optimization.py`.
    -   A particle's "position" is defined as a path, and its "velocity" is defined as a list of swap operations.
    -   Discrete analogues for velocity and position updates were implemented. This approach relies on the fitness function's heavy penalty for non-contiguous "jumps" to guide the swarm toward valid paths.
    -   During a detailed review, the user correctly pointed out that the sequential application of swap operations (the "velocity") causes "distortion," as the effect of a later swap is dependent on the state change from an earlier swap. It was clarified that this is an accepted and inherent characteristic of this discrete PSO adaptation, providing a form of stochastic perturbation that aids in exploration, with the fitness function acting as the ultimate arbiter of path quality.

### Major Refactoring and Code Quality Enhancements

-   **Centralized Path Utilities:** To eliminate code duplication across solvers, the common helper functions `generate_random_path` and `generate_neighbor_path` were moved from individual solver files into the shared `src/core/utils.py` module. `monte_carlo.py` and `simulated_annealing.py` were refactored to use these new shared utilities.
-   **Fitness Function Hardening:** The `calculate_fitness_score` function in `utils.py` was made more robust. A Manhattan distance check was added to penalize non-contiguous path "jumps", which was a weakness identified during the SA implementation.
-   **Increased Test Coverage:** 
    -   Added smoke tests for all new metaheuristic solvers (SA, GA, TS, PSO) to ensure they run and produce correctly formatted output.
    -   Added new, dedicated unit tests to `test_utils.py` for the shared `generate_random_path` and `generate_neighbor_path` functions to validate their core logic (e.g., path contiguity, no duplicates, correct start point).
-   **Code Style and Linting:** Fixed several `pre-commit` errors reported by `ruff`, including an `F821 Undefined name` error from a missing `import` and an `E402 Module level import not at top of file` style violation.

## 2025-10-01

### Codebase Modernization and Toolchain Overhaul

-   **Path Handling Refactoring:** Replaced all instances of `os.path` with the modern `pathlib` library across the test suite (`conftest.py`, `test_dfs.py`). This improves path manipulation logic, making it more readable, consistent, and object-oriented.
-   **Alternative A* Solver Implementation:** Implemented a new A* solver variant, `solve_puzzle_a_star_sortedlist`, which leverages `sortedcontainers.SortedList` as its priority queue instead of the standard `heapq`. A corresponding parametrized unit test was added to `test_a_star.py` to ensure its correctness against the full puzzle suite.
-   **Pre-Commit and CI/CD Pipeline Refinement:**
    -   **Test Pathing Resolution:** Resolved a critical `ModuleNotFoundError` during test collection by migrating the Python path configuration from a `sys.path` manipulation in `conftest.py` to a centralized `pythonpath` setting in `pytest.ini`. This aligns with `pytest` best practices.
    -   **Toolchain Consolidation:** Diagnosed and fixed a persistent formatting conflict loop between `black`, `isort`, and `ruff`. The pre-commit configuration was completely refactored to use `ruff` exclusively for all linting, import sorting, and code formatting, removing `isort` and `black` for a faster and simpler CI pipeline.

### Metaheuristic Search Framework and Baseline Implementation

-   **Fitness Function Design & Implementation:**
    -   Designed and implemented a comprehensive `calculate_fitness_score` function in `utils.py`. This function establishes the core evaluation metric for all metaheuristic solvers, incorporating a system of penalties and rewards (for path length, waypoint sequencing, etc.).
    -   The function was enhanced to return both the path's current score and the puzzle's theoretical perfect score, providing a clear benchmark for solution quality.
    -   Added a full suite of unit tests in `test_utils.py` to validate the fitness function's behavior.

-   **Monte Carlo Solver:**
    -   Implemented the first metaheuristic solver, `solve_puzzle_monte_carlo`, as a baseline for performance comparison. The solver generates a specified number of random paths and returns the one with the highest fitness score.
    -   The solver's logging was integrated with the new fitness function output to display comparative scores (e.g., `Best score: 420200/1720360`).
    -   A unit test was created to verify the integrity of the Monte Carlo solver, ensuring it produces valid paths.

### Code Quality and Refactoring

-   **DRY Principle Refactoring:** Refactored all existing exact solvers (`dfs.py`, `a_star.py`, `cp.py`) to consume the `num_map` from the puzzle dictionary, eliminating redundant code.
-   **Bug Fixes:** Diagnosed and resolved multiple `NameError` exceptions in `a_star.py` and `test_utils.py` that were introduced during refactoring, ensuring the entire test suite passes.

## 2025-09-25

### Advanced Solver Implementation and Analysis

-   **A* Solver:** Implemented a complete A* solver (`a_star.py`) using a priority queue (`heapq`) and a Manhattan distance heuristic. Iteratively debugged the implementation, correcting a critical flaw in the `closed_set` logic to ensure proper state tracking, which resulted in all test cases passing.
-   **CP-SAT Solver:** Developed a solver using Google's OR-Tools (`cp.py`). Modeled the puzzle as a Constraint Satisfaction Problem, and after multiple iterations, resolved an `INFEASIBLE` status by re-modeling the problem. The final, successful implementation uses the "dummy node" technique to correctly represent a Hamiltonian path with an `AddCircuit` constraint.
-   **Algorithm Analysis:** Performed a detailed theoretical analysis of the Time and Space Complexity (TC/SC) for the DFS, A*, and CP-SAT solvers. Compared their trade-offs in terms of memory usage, practical speed, and implementation paradigm.

### Major Project Structure Refactoring

-   Relocated all solver implementations (`dfs.py`, `a_star.py`, `cp.py`) into a new, dedicated `src/core/solvers/` directory to improve modularity and separation of concerns.
-   Mirrored the source code structure within the test directory by creating `src/core/tests/solvers/` and moving the corresponding test files. This refactoring enhances test organization and future scalability.
-   Updated all relevant `import` statements across the test suite to reflect the new file locations, ensuring all 19 tests pass after the refactoring.

### To-Do List

-   **Metaheuristic Solvers:** Begin implementation of non-deterministic, metaheuristic algorithms.
    -   Define a robust **fitness/cost function** to score partial or imperfect solutions.
    -   Implement a baseline **Monte Carlo (Random Sampling) Search**.
    -   Implement other metaheuristics such as **Simulated Annealing**, **Genetic Algorithm**, or **Ant Colony Optimization**.
    -   All metaheuristic solvers should accept an `attempts` parameter to control the number of iterations.

## 2025-09-23

### Solver Verification and Visualization Overhaul

-   **DFS Solver Logic Verified:** Through a process of debugging and adding detailed logging, it was determined that the core DFS solver algorithm was logically correct. The previously observed test failures were traced back to incorrect reference solutions in the test data.
-   **Test Data Corrected:** Fixed typos in the ground-truth data within `conftest.py`, leading to all 9 unit tests passing and validating the solver's correctness.
-   **Advanced Visualization Implemented:** Iteratively redesigned and implemented multiple solution-visualization features in `utils.py` based on interactive feedback:
    -   Implemented two distinct console-based styles: a simple `[bracket]` highlighter and a more advanced ANSI background-color highlighter.
    -   Added console-based animation functions (`animate_solution_*`) to display the step-by-step pathfinding process, addressing the need to show path order.
    -   To handle layout "wobbling" during animation, the printing logic was refactored to pre-calculate and enforce a fixed grid size across all animation frames.
-   **GIF Animation Generation:** Implemented a new feature, `save_animation_as_gif`, using the Pillow library to generate and save high-quality, shareable GIF animations of puzzle solutions, complete with wall rendering.

## 2025-09-22

### Input System Refactoring and Test Data Integration

-   **Input Refactoring:** Overhauled the puzzle input system. Puzzles are now defined with a readable, text-based `puzzle_layout`, which is then processed by a dedicated `parser` in `utils.py`.
-   **Utility Functions:** Created `src/core/utils.py` to house shared functions, including the new `parse_puzzle_layout` parser and a `visualize_solution` function for displaying results.
-   **Test Data Enhancement:** Integrated the user-provided, ground-truth solutions for all six puzzles (`puzzle_01` to `puzzle_06`) into the `conftest.py` test suite, enabling strict path verification.

### To-Do List

-   **Unit Testing:** Write and pass unit tests for the new utility functions in `src/core/utils.py`.
-   **Algorithm Validation:** Run the full test suite to verify the DFS solver's correctness against all 6 ground-truth solutions.
-   **Debugging:** Based on test results, debug any discrepancies between the solver's output and the expected solutions.
-   **Visualization Polish:** Re-evaluate and possibly redesign the presentation of the visualized solution for better clarity during debugging.

## 2025-09-21

### Test Suite and Architecture Overhaul

-   **Test Case Expansion:** Transcribed and added puzzles 01 through 06 from image files into the test suite.
-   **Test Architecture Refactoring:** Refactored the entire test workflow to be scalable and reusable. Test data is now centralized in `conftest.py` and dynamically loaded into a single test function in `test_dfs.py` using `pytest.parametrize`.
-   **Input Refactoring:** Enhanced the core solver and input data structure to support "blocked cells" in addition to "walls", making the algorithm more versatile.
-   **Workflow Update:** Updated the internal Gemini README to define collaboration rules regarding package management and test execution.

## 2025-09-20

### Core Solver Implementation

-   Initialized the project structure.
-   Implemented the core puzzle-solving logic in `src/core/dfs.py` using a backtracking Depth-First Search (DFS) algorithm.
-   The solver handles grids with numbered waypoints and walls that blocking paths.

### Testing and Reporting Setup

-   Introduced `pytest` as the testing framework.
-   Created a test suite in `src/core/tests/test_dfs.py` with multiple test cases, including simple solvable puzzles, puzzles with walls, and puzzles designed to be unsolvable.
-   Iteratively refined the "unsolvable" test cases after discovering the solver was more robust than initially anticipated.

### Automation and Workflow Refinement

-   Set up `loguru` to provide detailed, professional-grade logging for test execution.
-   Configured the logger to output to timestamped files (`log_[timestamp].log`) with UTC timestamps in the filename and local timezone information in the log messages.
-   Engineered a system to automatically generate test reports that mirror the console output.
-   After exploring `pytest.ini` and `conftest.py` hooks, finalized the reporting mechanism using a `run_tests.bat` script for maximum reliability and platform consistency. This script redirects all console output to a timestamped `test_report_[timestamp].txt` file.
-   The final workflow is simplified to a single command: `.\run_tests.bat`
