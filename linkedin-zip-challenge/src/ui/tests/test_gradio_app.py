# src/ui/tests/test_gradio_app.py
import requests
from unittest.mock import Mock, patch


from PIL import Image
from src.ui.gradio_app import generate_puzzle_ui, solve_puzzle_ui


@patch("src.ui.gradio_app.requests.post")
def test_solve_puzzle_ui_success(mock_post):
    """Test the UI logic for a successful API call."""
    # Arrange
    # Configure the mock response for a successful call
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "solution_path": "(0, 0) -> (0, 1)",
        "solution_gif_b64": "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",  # 1x1 transparent GIF
        "solution_final_image_b64": "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
    }
    mock_post.return_value = mock_response

    # Act
    gif_html, final_image_html = solve_puzzle_ui("layout", "walls", "DFS")

    # Assert
    assert (
        "<img src='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'"
        in gif_html
    )
    assert (
        "<img src='data:image/png;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'"
        in final_image_html
    )


@patch("src.ui.gradio_app.requests.post")
def test_solve_puzzle_ui_api_error(mock_post):
    """Test the UI logic for a failed API call."""
    # Arrange
    # Configure the mock response for a 404 error
    mock_response = Mock()
    mock_response.status_code = 404
    mock_response.json.return_value = {"detail": "Solver not found"}

    # Create a mock HTTPError that has the mock response
    mock_http_error = requests.exceptions.HTTPError()
    mock_http_error.response = mock_response
    mock_post.side_effect = mock_http_error

    # Act
    gif_html, final_image_html = solve_puzzle_ui("layout", "walls", "BadSolver")

    # Assert
    assert "Solver not found" in gif_html
    assert "Solver not found" in final_image_html


@patch("src.ui.gradio_app.generate_puzzle")
def test_generate_puzzle_ui_success(mock_generate_puzzle):
    """Test the puzzle generation UI function for a successful case."""
    # Arrange
    sample_puzzle_data = {
        "grid_size": (2, 2),
        "puzzle_layout": [["1 ", "  "], ["xx", "2 "]],
        "walls": {((0, 1), (1, 1))},
        "grid": [[1, 0], [0, 2]],
        "blocked_cells": {(1, 0)},
        "num_map": {1: (0, 0), 2: (1, 1)},
    }
    mock_generate_puzzle.return_value = (sample_puzzle_data, [])

    # Act
    image_result, layout_str, walls_str = generate_puzzle_ui(num_blocks=1)

    # Assert
    mock_generate_puzzle.assert_called_once_with(
        m=6, n=6, has_walls=True, num_blocked_cells=1
    )
    assert isinstance(image_result, Image.Image)
    assert isinstance(layout_str, str)
    assert isinstance(walls_str, str)
    assert "'xx'" in layout_str
    assert "((0, 1), (1, 1))" in walls_str
