# src/app/tests/test_solver_api.py
import ast
import pprint

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.app.main import app
from src.core.solvers import registry
from src.core.solvers.verify import is_solution
from src.core.tests.conftest import puzzle_01_data, puzzle_01_layout, solution_01

client = TestClient(app)

#: Enough for a few reruns per heuristic without six of them slowing the suite.
TEST_HEURISTIC_BUDGET_SECONDS = 0.2
GAVE_UP_BUDGET_SECONDS = 0.05


def _payload(solver_name: str) -> dict:
    return {
        "puzzle_layout_str": pprint.pformat(puzzle_01_layout),
        "walls_str": pprint.pformat(puzzle_01_data.get("walls", set())),
        "solver_name": solver_name,
    }


def test_solve_puzzle_happy_path():
    """Test the /api/solver/solve endpoint with a valid request."""
    # Arrange
    layout_str = pprint.pformat(puzzle_01_layout)
    walls_str = pprint.pformat(puzzle_01_data.get("walls", set()))
    request_payload = {
        "puzzle_layout_str": layout_str,
        "walls_str": walls_str,
        "solver_name": "DFS",
    }

    # Act
    response = client.post("/api/solver/solve", json=request_payload)

    # Assert
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "solution_path" in data
    assert "solution_gif_b64" in data
    assert "solution_final_image_b64" in data
    assert data["solution_gif_b64"] is not None
    assert data["solution_final_image_b64"] is not None


@pytest.mark.parametrize("solver_name", list(registry.SOLVERS))
def test_every_registered_solver_answers_and_only_draws_real_solutions(
    solver_name, monkeypatch
):
    """Whatever a solver returns, a drawn answer must be a solution.

    A heuristic may give up (no picture, a "could not find" message), and the RL solver
    is a 503 when no checkpoint is under models/ -- but an exact solver must solve, and
    nothing may be drawn that `is_solution` rejects.
    """
    monkeypatch.setattr(
        registry, "HEURISTIC_TIME_BUDGET_SECONDS", TEST_HEURISTIC_BUDGET_SECONDS
    )

    response = client.post("/api/solver/solve", json=_payload(solver_name))

    if response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE:
        assert solver_name == "RL (behaviour cloning)"
        return
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    if data["solution_gif_b64"] is None:
        assert solver_name not in registry.EXACT_SOLVERS
        assert data["solution_final_image_b64"] is None
        assert "could not find a solution" in data["solution_path"]
        return
    path = [ast.literal_eval(step) for step in data["solution_path"].split(" -> ")]
    assert is_solution(puzzle_01_data, path)


def test_a_heuristic_that_gave_up_is_not_drawn_as_an_answer(monkeypatch):
    """The failure this guards against: a best guess one cell short, drawn as solved."""
    monkeypatch.setattr(
        registry, "HEURISTIC_TIME_BUDGET_SECONDS", GAVE_UP_BUDGET_SECONDS
    )
    monkeypatch.setitem(
        registry.SOLVERS,
        "Monte Carlo",
        registry._until_verified(lambda puzzle: solution_01[:-1]),
    )

    data = client.post("/api/solver/solve", json=_payload("Monte Carlo")).json()

    assert data["solution_gif_b64"] is None
    assert data["solution_final_image_b64"] is None
    assert "could not find a solution" in data["solution_path"]


def test_solve_puzzle_bad_layout():
    """Test the endpoint with a malformed puzzle_layout_str."""
    # Arrange
    request_payload = {
        "puzzle_layout_str": "not a valid list",
        "walls_str": "set()",
        "solver_name": "DFS",
    }

    # Act
    response = client.post("/api/solver/solve", json=request_payload)

    # Assert
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert "Error parsing input" in response.json()["detail"]


def test_solve_puzzle_solver_not_found():
    """Test the endpoint with a non-existent solver name."""
    # Arrange
    layout_str = pprint.pformat(puzzle_01_layout)
    request_payload = {
        "puzzle_layout_str": layout_str,
        "walls_str": "set()",
        "solver_name": "non_existent_solver",
    }

    # Act
    response = client.post("/api/solver/solve", json=request_payload)

    # Assert
    assert response.status_code == status.HTTP_404_NOT_FOUND
    assert "Solver 'non_existent_solver' not found" in response.json()["detail"]
