# src/app/tests/test_solver_api.py
import pprint

from fastapi import status
from fastapi.testclient import TestClient

from src.app.main import app
from src.core.tests.conftest import puzzle_01_data, puzzle_01_layout

client = TestClient(app)


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
