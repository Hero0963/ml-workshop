# Development Log

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