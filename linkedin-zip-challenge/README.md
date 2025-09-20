# LinkedIn Zip Challenge Solver

## 1. Project Purpose

This project provides an automated solver for the LinkedIn "Zip" path-drawing puzzle game. The solver is designed to find a valid path on a grid that connects numbered waypoints in sequential order, ensuring the path visits every cell exactly once.

The core logic is implemented using a backtracking Depth-First Search (DFS) algorithm.

## 2. Project Structure

```
linkedin-zip-challenge/
├── src/
│   ├── core/
│   │   ├── dfs.py               # Core solver algorithm
│   │   └── tests/
│   │       ├── test_dfs.py      # Pytest unit tests
│   │       └── reports/         # Test reports and logs are generated here
├── illustrations/               # Images of puzzle examples
├── .venv/                       # Project-specific virtual environment
├── pyproject.toml               # Project dependencies and configuration for `uv`
├── run_tests.bat                # Script to run the test suite
└── README.md                    # This file
```

## 3. How to Use

### a. Setup

This project uses `uv` for package management. The required packages are `pytest` and `loguru`.

### b. Running Tests

To run the entire test suite and generate reports:

1.  Navigate to the `linkedin-zip-challenge` directory in your terminal.
2.  Execute the batch script:
    ```powershell
    .\run_tests.bat
    ```

### c. Viewing Results

After running the tests, check the `src/core/tests/reports/` directory. Two new timestamped files will be created for each run:

*   `test_report_[timestamp].txt`: A plain text summary of the test results, identical to the terminal output.
*   `log_[timestamp].log`: A detailed, step-by-step log of the test execution for debugging purposes.
