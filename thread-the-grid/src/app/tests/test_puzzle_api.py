# src/app/tests/test_puzzle_api.py
from unittest.mock import patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.app.main import app
from src.app.routers.puzzle import GENERATION_ATTEMPTS
from src.core.solvers.registry import SOLVERS
from src.core.solvers.verify import is_solution
from src.core.utils import parse_puzzle_layout

client = TestClient(app)


@pytest.mark.parametrize("size", [4, 5, 6])
def test_a_generated_board_is_well_formed_and_solvable(size: int) -> None:
    response = client.post("/api/puzzle/generate", json={"size": size})

    assert response.status_code == status.HTTP_200_OK
    body = response.json()
    assert body["grid_size"] == [size, size]
    assert [len(row) for row in body["layout"]] == [size] * size
    cells = [cell for row in body["layout"] for cell in row]
    assert all(len(cell) == 2 for cell in cells)
    assert "xx" not in cells  # the UIs no longer offer blocked cells
    waypoints = sorted(int(cell) for cell in cells if cell.isdigit())
    assert waypoints == list(range(1, len(waypoints) + 1))

    puzzle = parse_puzzle_layout(body["layout"])
    puzzle["walls"] = {
        (tuple(wall["cell1"]), tuple(wall["cell2"])) for wall in body["walls"]
    }
    for first, second in puzzle["walls"]:
        assert abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
    assert is_solution(puzzle, SOLVERS["CP-SAT"](puzzle))


@pytest.mark.parametrize("size", [3, 7])
def test_sizes_the_uis_do_not_offer_are_rejected(size: int) -> None:
    response = client.post("/api/puzzle/generate", json={"size": size})

    assert response.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


def test_a_generator_that_gives_up_once_is_retried() -> None:
    board = parse_puzzle_layout([["01", "  "], ["03", "02"]])
    board["walls"] = set()
    with patch(
        "src.app.routers.puzzle.generate_puzzle", side_effect=[None, (board, [])]
    ) as generator:
        response = client.post("/api/puzzle/generate", json={"size": 4})

    assert response.status_code == status.HTTP_200_OK
    assert generator.call_count == 2


def test_a_generator_that_keeps_giving_up_is_reported() -> None:
    with patch(
        "src.app.routers.puzzle.generate_puzzle", return_value=None
    ) as generator:
        response = client.post("/api/puzzle/generate", json={"size": 5})

    assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "try again" in response.json()["detail"]
    assert generator.call_count == GENERATION_ATTEMPTS
