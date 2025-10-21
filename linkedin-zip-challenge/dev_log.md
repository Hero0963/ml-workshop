# Development Log

## 2025-10-21

### Dockerized Development Workflow Automation

To streamline the development process and simplify the startup of the containerized environment, this commit introduces a new automation script and enhances the project's containerization strategy.

-   **Docker Compose Enhancement**:
    -   The `docker-compose.yml` file was updated to define a complete, multi-service development environment, including the FastAPI backend (`zip-challenge-app`) and the Svelte frontend (`svelte-frontend`).
    -   Configuration was refined to ensure proper volume mounting for live code reloading and inter-container communication.

-   **Automated Startup Script**:
    -   Created a new Python script, `run_docker_dev.py`, to provide a one-command solution for launching the entire development stack.
    -   The script automates the following sequence:
        1.  Stops and removes any existing containers (`docker compose down`).
        2.  Builds fresh images and starts all services in the background (`docker compose up --build -d`).
        3.  Waits briefly for the main application container to initialize.
        4.  Executes the command to start the FastAPI server inside the running container, ensuring the virtual environment is activated.
    -   This script eliminates the need for manual `docker exec` commands and simplifies the developer onboarding experience.

-   **Documentation Update**:
    -   Updated the `README.md` and `README_zh-TW.md` files with a new "Running with Docker" section, explaining how to use the `run_docker_dev.py` script.
    -   This ensures that the documentation is synchronized with the latest, most efficient development workflow.

## 2025-10-20 (another commit)

### Svelte Frontend UX and Test Suite Refinements

This commit enhances the Svelte frontend's user experience and ensures the stability of the existing test suite.

-   **Svelte UI Enhancements**:
    -   **In-place Cell Editing**: The cell editing UX was significantly improved by replacing the browser's default `prompt()` dialog. A new, dynamic in-place editing mechanism was implemented. Now, clicking a cell overlays an `<input>` element directly onto the canvas grid, allowing for a more seamless and intuitive editing workflow.
-   **Test Suite Maintenance**:
    -   **Gradio Test Fix**: Corrected a failing test case in `test_gradio_app.py`. The assertion was updated to correctly handle the HTML-formatted error messages now returned by the Gradio UI, bringing the test suite back to a passing state.

## 2025-10-20

### Gradio UI Overhaul and Interactive Solver Implementation

This phase focused on building a highly interactive and user-friendly puzzle editor within the Gradio web UI, moving from a text-based input to a full "What You See Is What You Get" (WYSIWYG) experience.

-   **Interactive Puzzle Editor ("V2")**:
    -   Replaced the initial text-based "naive" solver tab with a new "Interactive" tab.
    -   Implemented a dynamic grid creation system where users can specify puzzle dimensions (`m x n`).
    -   **Refactored Wall Editor**: Based on user feedback regarding the initial confusing checkbox-based UI, the wall editor was completely redesigned.
        -   Users now input wall coordinates using four simple number boxes (`r1, c1, r2, c2`).
        -   A list view displays all current walls, with a proper "select-then-click" button to delete walls.
    -   **Live Image Preview**: Added a new preview panel that generates and displays an image of the puzzle in real-time. The preview automatically updates whenever the user edits the puzzle grid (adding numbers/obstacles) or modifies the wall list.
    -   **New UI Controls**: Implemented a "Reset" button to clear all interactive components to their default state.

-   **Debugging and Stability**:
    -   **Extensive Bug Fixing**: Resolved a long series of bugs discovered during iterative development, including `IndentationError`, `NameError`, `AttributeError`, `UnboundLocalError`, and several data format mismatches between the frontend and backend (e.g., `'x'` vs `'xx'`, `dict` vs `set`).
    -   **Enhanced Logging**: Added detailed `loguru` logging to both the frontend (`gradio_app.py`) and backend (`solver.py`). These logs capture the raw UI payload and the parsed puzzle data, which was critical in diagnosing the data flow issues. Also added logging for temporary file deletion in the backend.
    -   **Code Maintenance**: Fixed a `FutureWarning` from the `pandas` library by migrating from the deprecated `Styler.applymap` to `Styler.map`.

-   **Architectural Refinements**:
    -   The frontend `gradio_app.py` was refactored multiple times to serve as a robust "Adapter", translating intuitive user actions into the precise data formats expected by the backend API.
    -   The core backend logic in `utils.py` and `solver.py` was validated and corrected to ensure it properly handles obstacles and other puzzle constraints.

## 2025-10-16

### Implementation of Service-Oriented Architecture (Phase 1)

Following the pivot from pure algorithmic development, the first phase of the user-facing web service has been implemented. This phase establishes the core architecture and a functional user interface.

-   **Web Service Backend (FastAPI):**
    -   Initialized a FastAPI application (`src/app/main.py`) to serve as the backend.
    -   Implemented a robust, layered configuration system using `pydantic-settings` (`src/settings.py`) that reads from a `.env` file, making settings like port numbers easily configurable.
    -   Refactored the API structure into a scalable `routers` and `schemas` pattern. All API endpoints are now modularly organized (e.g., `src/app/routers/echo.py`, `src/app/routers/solver.py`).
    -   Created a `/api/solver/solve` endpoint that receives puzzle data, calls the appropriate core solver, and returns a JSON response containing the solution path and Base64-encoded images.
    -   Improved code quality by replacing magic numbers for HTTP status codes with `fastapi.status` constants.

-   **Web User Interface (Gradio):**
    -   Developed a multi-tab Gradio interface (`src/ui/gradio_app.py`) for user interaction.
    -   The UI is mounted directly within the FastAPI application, creating a single, unified service.
    -   Implemented a "Puzzle Solver naive version" tab that allows users to paste puzzle layouts and walls, select a solver, and receive a visual solution.
    -   The UI now displays both an animated GIF of the solution process and a static image of the final result.

-   **Visualization Enhancements:**
    -   Created a new `save_detailed_animation_as_gif` function in `utils.py` to generate GIFs with enhanced visuals, including a highlighted path head (blue) and sequential step numbers (green).
    -   Added a `save_solution_as_image` function to generate a static PNG of the final solved puzzle.
    -   The backend now uses these new functions to provide richer visual feedback to the user.

-   **Bug Fixes & Refinements:**
    -   Standardized file path comments in `.py` files to use forward slashes (`/`) for cross-platform consistency.
    -   Resolved a `PermissionError` on Windows related to `tempfile` by implementing a more robust file handling pattern in the solver API.
    -   Corrected multiple `IndentationError` syntax issues that arose during refactoring.
    -   Standardized type hint styles in Pydantic schemas to the modern `|` union operator as per project conventions.

### Quality Assurance and Refactoring

-   **Unit Test Implementation**: Added a comprehensive suite of unit tests for the new service-oriented architecture. This includes API endpoint tests using `TestClient` (`src/app/tests/`), UI logic tests using `unittest.mock` (`src/ui/tests/`), and smoke tests for new visualization utilities in `src/core/tests/`.
-   **Project Structure Refactoring**: To improve modularity, moved `puzzle_generator.py` and `generate_dataset.py` into a new dedicated `src/core/puzzle_generation/` directory and updated all corresponding import paths across the project.
-   **Performance Tuning**: Modified the `generate_dataset.py` script to limit the multiprocessing pool to 75% of available CPU cores, ensuring system responsiveness during heavy computation.

### Next Steps

-   **Interactive UI**: Implement the "Puzzle Solver interact version" tab in the Gradio UI.
-   **Containerization**: Introduce a `Dockerfile` to allow the entire web service to be built and run as a container.

## 2025-10-15

### Reinforcement Learning Development Paused

Due to the inherent challenges in reward function design and overall training complexity, the Reinforcement Learning (RL) development effort is being temporarily paused.

Future work in this area will be resumed after a period of deeper research into advanced RL concepts and architectures. The planned areas of study include:
-   Architectures of seminal models like **AlphaGo** and **AlphaZero**.
-   Reviewing the hands-on examples in the local `more_simple_reinforcement_learning` directory.
-   Studying the "Hands-on Reinforcement Learning" course materials (from `hrl.boyuai.com`).

When RL development resumes, a revised approach will be considered to simplify the problem, such as:
-   Reducing the `map_size` to a smaller dimension.
-   Relaxing the environment's constraints (e.g., allowing the agent to revisit paths, transforming the problem from finding a single Hamiltonian path to a more flexible pathfinding task).

### Project Pivot to Service-Oriented Architecture

The project's immediate focus will shift from algorithmic development to building a user-facing service. The goal is to create an application with a UI that allows users to upload their own puzzles and receive a computed solution.

### New To-Do List

-   **Service Backend:** Implement a web backend using **FastAPI**.
-   **User Interface:** Create an interactive web UI with **Gradio**.
-   **Future Exploration:** Investigate the integration of **MCP (Model-View-Controller Pattern)** and **multi-modal** capabilities.

### Archived Progress 

*This section documents the last active development goal before the pivot.*

The previous focus was on attempting to solve a 6x6 map using an RL approach. The strategy was to first test and solve the problem on a **single map** (i.e., achieve overfitting) as a proof of concept. The successful completion of this step would then serve as a foundation for the ultimate goal of **generalizing** the solution to arbitrary 6x6 maps. The starting point for this development was the implementation of the `src/core/rl/train_single_sb.py` script.

## 2025-10-13

### Deep Dive into Deterministic Loop & Reward Shaping

Following the successful overfitting of the MLP-based model during training and its subsequent failure in deterministic evaluation, a series of experiments were conducted to resolve the underlying "deterministic policy loop" issue with a new CNN-based model.

-   **Problem Persistence & State Representation Fix**: Despite refactoring the environment to use a 6-channel image-like state representation (including separate layers for walls and obstacles) and switching to a `CnnPolicy`, the agent continued to fail during deterministic evaluation. It achieved high rewards during training (with exploration) but fell into inescapable loops when `deterministic=True`. This confirmed the issue was not the agent's "vision" but likely its "motivation".

-   **Hypothesis 1: Insufficient Penalty for Inefficiency.** The first hypothesis was that the `-1.0` time penalty was not enough to discourage looping.
    -   **Experiment:** A "soft constraint" was added to the reward function in `rl_env.py`, applying a `-2.0` penalty for revisiting any cell already in the `path_taken`.
    -   **Result:** **Failure.** The evaluation log (`evaluation_path_2025-10-13_13-37-36.log`) showed that while the agent explored more territory, it ultimately still fell into a tight loop (`(4, 0) <-> (5, 0)`), indicating the revisit penalty was not sufficient to overcome the root cause.

-   **Hypothesis 2: Dense Reward Traps.** The primary suspect shifted to the distance-based reward shaping (`(dist_before - dist_after) * weight`), which could be creating local optima ("reward traps") that are more attractive than exploring a path to the true goal.
    -   **Experiment:** The reward shaping weight was reduced by an order of magnitude, from `0.1` to `0.01`. The parameter was also refactored into the `PuzzleEnv` constructor and the training script's `CONFIG` for easier tuning.
    -   **Result:** **Failure.** The evaluation log (`evaluation_path_2025-10-13_14-05-36.log`) again showed the agent getting stuck in a terminal loop, proving that even a very small positive incentive towards the goal can create a powerful enough trap to derail the deterministic policy.

-   **Final Diagnosis:** The distance-based reward shaping, even with a minimal weight, is fundamentally at odds with the sparse penalty system. It encourages a "greedy" local-optimization behavior that results in policy loops. The agent is unwilling to incur a small penalty (by moving away from the target) to find a path around an obstacle, as the dense reward signal is too dominant.

### To-Do List

-   **[Next Step]** Completely eliminate the dense reward signal by setting `DISTANCE_REWARD_WEIGHT` to `0` in `train_single_cnn_sb.py`.
-   Re-train the model from scratch using the purely sparse reward function (only step/revisit/invalid penalties and waypoint/goal rewards).
-   Perform a deterministic evaluation on the new model to verify if the looping issue is finally resolved.
-   If the issue persists, the final recourse is to escalate the "soft constraint" on revisits to a "hard constraint" by making it an invalid move.

## 2025-10-12

### RL Agent Deep Debugging and Analysis

A deep-dive debugging session was conducted to diagnose why the DQN agent, despite successful training metrics, failed during deterministic evaluation.

-   **Initial State & Problem:** The agent, whether custom-built or using `stable-baselines3`, showed high average rewards during training but consistently failed to complete a puzzle during deterministic evaluation (`epsilon=0`), always timing out at the maximum step limit.

-   **Hypothesis 1: Insufficient Evaluation Steps.** The initial hypothesis was that the evaluation loop's step limit was too low. This was proven false, as increasing the limit in the evaluation script had no effect. The root cause was identified as a hardcoded `_max_steps` limit within the `PuzzleEnv` itself.

-   **Hypothesis 2: Flawed Reward Shaping.** The second hypothesis was that the distance-based reward shaping (`(dist_before - dist_after) * 1.0`) was creating a "reward trap" or local optimum, causing the agent to loop near the goal. An experiment was conducted by reducing the shaping weight to `0.1`. While this produced even better training metrics, the deterministic evaluation still failed in the exact same manner.

-   **Final Diagnosis: Deterministic Policy Loop.** The conclusive diagnosis is that the agent's learned deterministic policy contains an inescapable loop. The successful, shorter-episode training runs were an illusion created by random exploration (`epsilon > 0`) accidentally "bumping" the agent out of its learned loop, allowing it to reach the goal. When this randomness is removed, the policy's fatal flaw is revealed.

-   **Framework Enhancement:** To facilitate debugging, the `PuzzleEnv` was refactored to allow its `max_steps` limit to be configured externally during instantiation. The evaluation scripts (`evaluate_sb.py`) were updated to use this new parameter, providing a more flexible testing environment.

### Reinforcement Learning Framework Q&A

A summary of the RL agent's core mechanics was documented to clarify understanding.

-   **Q1: What are the agent's movement rules?**
    -   The agent has a discrete action space (Up, Down, Left, Right). It is permitted to reverse its direction and revisit cells it has previously occupied. There are no rules preventing revisits.

-   **Q2: What is the agent's goal and behavior?**
    -   **Goal:** To navigate from a starting position, visiting a sequence of numbered waypoints in the correct order, and finally arriving at the last waypoint.
    -   **Behavior:** The agent's behavior is governed by a policy network (an MLP). This network takes the current state (`agent_location`, `next_waypoint_location`) and outputs Q-values for each of the four actions. The agent selects the action with the highest Q-value, which it predicts will lead to the maximum cumulative future reward.

-   **Q3: How does the agent interact with the environment?**
    -   The interaction follows the standard RL loop. The agent submits an `action` to the environment via `env.step(action)`. The environment transitions to a `next_state` and returns a `reward`, a `terminated` flag (for goal completion), a `truncated` flag (for timeouts), and an `info` dictionary. The agent uses this feedback to update its policy.

-   **Q4: What is the reward function?**
    -   The reward function is composed of several components:
        -   `+1000.0` for reaching the final waypoint.
        -   `+200.0` for reaching an intermediate waypoint.
        -   `-10.0` for an invalid move (hitting a wall, obstacle, or boundary).
        -   `-1.0` as a time penalty for every step taken.
        -   `(dist_before - dist_after) * 0.1` as a small, dense reward for reducing the Manhattan distance to the next target.

-   **Q5: What logging is available besides the GIF animation?**
    -   **Console Logs:** Real-time statistical tables from `stable-baselines3` during training.
    -   **File Logs:** Detailed, timestamped logs saved by `loguru` to the `logs/` directory.
    -   **TensorBoard Logs:** The most powerful tool. Detailed, interactive graphs of all training metrics (reward, loss, etc.) are saved to `logs/sb_tensorboard/`. This can be launched via the command `tensorboard --logdir ./logs/sb_tensorboard/`.

### To-Do List

-   Review the visual `evaluation_sb.gif` and TensorBoard logs to pinpoint the exact location and pattern of the agent's deterministic loop.
-   Based on the loop's characteristics, redesign the reward function to specifically penalize or disincentivize the observed looping behavior.
-   If reward redesign is insufficient, consider redesigning the environment's rules of interaction (e.g., adding a penalty for immediately revisiting the previous state).

## 2025-10-12

### Reinforcement Learning (RL) Solver Framework

-   **Architectural Design**: Designed a complete framework to solve puzzles using Deep Reinforcement Learning. The approach is based on a DQN (Deep Q-Network) agent interacting with a custom environment, with a focus on making the training pipeline robust and reproducible.
-   **Custom RL Environment (`rl_env.py`)**: Implemented a `gymnasium.Env`-compatible environment, `PuzzleEnv`, to wrap the puzzle logic.
    -   Features a sophisticated **reward shaping** mechanism to provide dense rewards, guiding the agent by calculating the change in Manhattan distance to the next waypoint.
    -   The state space is defined by the agent's location and the next target waypoint, making the problem tractable for a neural network.
-   **DQN Agent (`dqn_agent.py`)**: Implemented a complete DQN agent, including:
    -   A `DQNModel` (MLP) to approximate the Q-function.
    -   A `ReplayBuffer` for experience storage and sampling.
    -   The core `DQNAgent` class encapsulating the learning logic, epsilon-greedy action selection, and target network updates.
-   **Two-Stage Training Pipeline**: Decoupled data generation from training for better workflow and reproducibility.
    -   **Dataset Generation (`generate_rl_dataset.py`)**: Created a multiprocessing-enabled script to generate and save large puzzle datasets (`6x6` and `7x7`). It outputs both a human-readable log for verification and a `pickle` file for the trainer to consume.
    -   **Training Script (`train.py`)**: Developed the main training script that loads the pre-generated dataset, manages the training loop, logs progress with `tqdm` and `loguru`, and saves the final trained model.

### Code Quality and Bug Fixes

-   **Pathing Logic**: Corrected a path calculation error in `generate_rl_dataset.py` and `train.py` that resulted in an incorrect, duplicated output directory path. The logic for determining the project root was made more robust.
-   **Linter Compliance**: Resolved a `SyntaxError` reported by `ruff` in `dqn_agent.py` by refactoring a multi-line expression to be more robust, ensuring the codebase passes all `pre-commit` checks.
-   **Dependency Management**: Identified and added necessary dependencies (`gymnasium`, `torch`, `tqdm`) for the new RL framework, using the project's `uv add` workflow.

## 2025-10-08

### Puzzle Generation Framework

-   **Procedural Puzzle Generator:** Created a new, sophisticated puzzle generation module (`src/core/puzzle_generator.py`).
    -   The core logic is built upon a **randomized backtracking algorithm** (`_generate_hamiltonian_path`) that generates a guaranteed valid solution path covering all visitable cells.
    -   Introduced a robust generation process with a **retry and decrement** mechanism: if generating a puzzle with `N` obstacles fails, it automatically retries, and if still unsuccessful, it gracefully degrades to attempt generation with `N-1` obstacles.
    -   Implemented a true **internal timeout** within the pathfinding algorithm to terminate and abandon attempts that take too long, preventing the process from hanging and saving CPU resources.
-   **Automated Dataset Creation Script:** Developed a powerful script (`src/core/generate_dataset.py`) to automate the creation of large puzzle datasets.
    -   Leverages the `multiprocessing` module to generate multiple puzzles in **parallel**, significantly speeding up the process.
    -   The script is highly configurable and creates a clean, **timestamped directory structure** for each run, organizing the generated puzzle data (`puzzles.py`) and GIF animations (`gifs/`) separately.
    -   Waypoint count is now **dynamically calculated** based on puzzle size (1/4 to 1/3 of path length) to create more balanced puzzles.

### Code Quality and Refactoring

-   **Improved Type Safety:** Introduced a `Puzzle` `TypedDict` in `utils.py` to provide a strict data contract for puzzle objects, replacing generic dictionaries and improving type safety across the codebase. All relevant functions (`puzzle_generator`, `utils`, etc.) were updated to use this precise type.
-   **DRY Principle Refactoring:** Refactored `puzzle_generator.py` to call the canonical `parse_puzzle_layout` function instead of manually re-implementing the puzzle object construction logic.
-   **Code Style and Conventions:**
    -   Standardized all new modules to use the `pathlib` library for path manipulations, adhering to project conventions.
    -   Updated all new modules to use absolute imports (e.g., `from src.core...`) as per user preference.
    -   Eliminated all "magic numbers" by defining them as named constants at the top of modules (e.g., `MAX_RETRIES_PER_COUNT`).
    -   Updated `gemini_readme_raw.md` to formally document the `pathlib` and absolute import style rules.
-   **Bug Fixes and Linting:**
    -   Fixed a critical `NameError` bug in `puzzle_generator.py` where `logger` was used but not imported.
    -   Fixed a `NameError` in the `generate_dataset.py` multiprocessing worker where `logger` was not available in the child process scope.
    *   Fixed a visual bug in `save_animation_as_gif` where `blocked_cells` were not being rendered; they are now correctly drawn as black squares.
    -   Resolved multiple `ruff` linter errors (`F841`: unused variable) in `utils.py`.

### Testing

-   **Generator Test Suite:** Created a new test file `src/core/tests/test_puzzle_generator.py`.
    -   Added a comprehensive **smoke test** (`test_generate_puzzle_smoke`) that validates the integrity of a complex generated puzzle (with walls and obstacles) and its solution.
    -   Added a dedicated test (`test_generate_puzzle_default_waypoints`) to verify the new **dynamic default waypoint calculation** logic.
-   **Standardized Test Output:** Replaced all `print()` statements in the new test file with `logger` calls to maintain consistency with project standards.

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
-   The final workflow is simplified to a single command: `.
un_tests.bat`
