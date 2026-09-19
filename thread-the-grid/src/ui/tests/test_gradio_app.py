# src/ui/tests/test_gradio_app.py
import requests
from unittest.mock import Mock, patch

import gradio as gr
import pytest
from PIL import Image
from src.ui.gradio_app import (
    _board_from_literals,
    add_wall,
    generate_puzzle_ui,
    load_board_into_editor,
    solve_from_image_ui,
    solve_puzzle_ui,
    walls_from_labels,
)


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


@patch("src.ui.gradio_app.requests.post")
def test_generate_puzzle_ui_asks_the_api_for_the_chosen_size(mock_post):
    """Generation goes through the API, like solving, so Svelte gets the same boards."""
    # Arrange
    mock_response = Mock()
    mock_response.json.return_value = {
        "grid_size": [2, 2],
        "layout": [["01", "  "], ["03", "02"]],
        "walls": [{"cell1": [0, 1], "cell2": [0, 0]}],
    }
    mock_post.return_value = mock_response

    # Act
    image_result, layout_str, walls_str = generate_puzzle_ui("4x4")

    # Assert
    assert mock_post.call_args.args[0].endswith("/api/puzzle/generate")
    assert mock_post.call_args.kwargs["json"] == {"size": 4}
    assert isinstance(image_result, Image.Image)
    assert "'03'" in layout_str
    assert "((0, 0), (0, 1))" in walls_str


@patch("src.ui.gradio_app.gr.Warning")
@patch("src.ui.gradio_app.requests.post")
def test_generate_puzzle_ui_shows_the_api_error(mock_post, mock_warning):
    error_response = Mock()
    error_response.json.return_value = {"detail": "Could not generate ... try again."}
    mock_post.side_effect = requests.exceptions.HTTPError(response=error_response)

    assert generate_puzzle_ui("5x5") == (None, "", "")
    assert "try again" in mock_warning.call_args.args[0]


def test_a_board_is_opened_in_the_editor_as_the_cells_a_user_would_type():
    """Waypoints lose their zero padding, blocked cells become 'x', empty becomes ''."""
    grid, walls = _board_from_literals(
        "[['01', '  '], ['xx', '12']]", "{((0, 1), (0, 0))}"
    )

    assert grid.values.tolist() == [["1", ""], ["x", "12"]]
    assert walls == {((0, 0), (0, 1))}


def test_a_board_without_walls_opens_too():
    """An empty set is printed as 'set()', which is not a set display."""
    _, walls = _board_from_literals("[['01', '02']]", "set()")

    assert walls == set()


def test_opening_the_editor_without_a_board_says_so():
    with pytest.raises(gr.Error):
        load_board_into_editor("", "")


@patch("src.ui.gradio_app.gr.Warning")
def test_every_added_wall_stays_listed(mock_warning):
    """The list used to show only the first wall however many were added."""
    walls = set()
    for (r1, c1), (r2, c2) in [((1, 1), (2, 1)), ((4, 2), (4, 3)), ((3, 1), (3, 2))]:
        walls, picker = add_wall(walls, r1, c1, r2, c2)

    assert len(walls) == 3
    assert [value for _, value in picker.choices] == picker.value
    assert walls_from_labels(picker.value) == walls
    mock_warning.assert_not_called()


@patch("src.ui.gradio_app.gr.Warning")
def test_a_wall_between_cells_that_do_not_touch_is_refused(mock_warning):
    walls, _ = add_wall(set(), 0, 0, 1, 1)

    assert walls == set()
    mock_warning.assert_called_once()


@patch("src.ui.gradio_app.requests.post")
def test_solve_from_image_ui_reports_warnings_and_hands_back_literals(
    mock_post, tmp_path
):
    """The adapter must surface dropped walls, not quietly show a tidy answer."""
    # Arrange
    image_path = tmp_path / "board.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model_name": "threadgrid-qwen35-4b-p4c:f16",
        "prompt_variant": "finetune",
        "solver_name": "CP-SAT",
        "grid_size": [6, 6],
        "layout": [["01", "  "], ["  ", "02"]],
        "walls": [{"cell1": [0, 0], "cell2": [0, 1]}],
        "warnings": ["Dropped a wall between non-adjacent cells: (0, 0)-(5, 5)."],
        "solvable": True,
        "solution_path": "(0, 0) -> (0, 1)",
        "solution_final_image_b64": "R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7",
        "solution_gif_b64": None,
    }
    mock_post.return_value = mock_response

    # Act
    summary, layout_text, walls_text, solution_html = solve_from_image_ui(
        str(image_path), "CP-SAT", False
    )

    # Assert
    assert "non-adjacent" in summary
    assert "threadgrid-qwen35-4b-p4c:f16" in summary
    assert "'01'" in layout_text
    assert "((0, 0), (0, 1))" in walls_text
    assert "data:image/png;base64," in solution_html


@patch("src.ui.gradio_app.requests.post")
def test_solve_from_image_ui_says_so_when_the_reading_is_unsolvable(
    mock_post, tmp_path
):
    """Unsolvable is proof of a misread, so it must not look like a hard puzzle."""
    # Arrange
    image_path = tmp_path / "board.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model_name": "m",
        "prompt_variant": "finetune",
        "solver_name": "CP-SAT",
        "grid_size": [6, 6],
        "layout": [["01"]],
        "walls": [],
        "warnings": ["The board as read has no solution, ..."],
        "solvable": False,
        "solution_path": None,
        "solution_final_image_b64": None,
        "solution_gif_b64": None,
    }
    mock_post.return_value = mock_response

    # Act
    summary, _, _, solution_html = solve_from_image_ui(str(image_path), "CP-SAT", False)

    # Assert
    assert "**Solvable**: NO" in summary
    assert solution_html == ""


@patch("src.ui.gradio_app.requests.post")
def test_solve_from_image_ui_does_not_blame_the_reading_when_a_solver_gave_up(
    mock_post, tmp_path
):
    """`solvable` is None when a solver that can give up did; that is not a misread."""
    # Arrange
    image_path = tmp_path / "board.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model_name": "m",
        "prompt_variant": "finetune",
        "solver_name": "Monte Carlo",
        "grid_size": [6, 6],
        "layout": [["01"]],
        "walls": [],
        "warnings": ["Monte Carlo is not exact and found no solution ..."],
        "solvable": None,
        "solution_path": None,
        "solution_final_image_b64": None,
        "solution_gif_b64": None,
    }
    mock_post.return_value = mock_response

    # Act
    summary, _, _, solution_html = solve_from_image_ui(
        str(image_path), "Monte Carlo", False
    )

    # Assert
    assert "**Solvable**: unknown" in summary
    assert "misread" not in summary
    assert solution_html == ""


@patch("src.ui.gradio_app.requests.post")
def test_solve_from_image_ui_shows_the_api_error_detail(mock_post, tmp_path):
    """A missing model must reach the user as a message, not an empty panel."""
    # Arrange
    image_path = tmp_path / "board.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    error_response = Mock()
    error_response.json.return_value = {"detail": "ollama is not running"}
    mock_post.side_effect = requests.exceptions.HTTPError(response=error_response)

    # Act
    summary, layout_text, _, _ = solve_from_image_ui(str(image_path), "CP-SAT", False)

    # Assert
    assert "ollama is not running" in summary
    assert layout_text == ""


def test_solve_from_image_ui_asks_for_an_image_before_calling_the_api():
    """No upload means no request; the API would only answer 422 anyway."""
    summary, layout_text, walls_text, solution_html = solve_from_image_ui(
        None, "CP-SAT", False
    )

    assert "Upload a screenshot" in summary
    assert (layout_text, walls_text, solution_html) == ("", "", "")
