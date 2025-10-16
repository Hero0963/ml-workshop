# src/ui/tests/test_gradio_app.py
import requests
from unittest.mock import Mock, patch


from src.ui.gradio_app import solve_puzzle_ui


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
    assert "API call failed: Solver not found" in gif_html
    assert "API call failed: Solver not found" in final_image_html
