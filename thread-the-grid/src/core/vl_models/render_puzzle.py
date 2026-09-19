# src/core/vl_models/render_puzzle.py
"""Draws a Zip puzzle the way the real screenshots look.

Replaces the renderer in ``src/core/puzzle_generation/generate_cod_dataset.py``, which
produced the 2025-10 dataset. Three things in that one make its output the wrong thing
to train on, and each is fixed here:

1.  **Walls and grid lines were the same colour.** It drew cell borders with
    ``outline="black"`` and walls as a ``width=5`` black line, so a wall was "a slightly
    thicker black line among black lines". In the real screenshots the grid is *light
    grey* and walls are *heavy black*: the discriminating cue is contrast, not width.
    Training on the old images teaches a cue that does not exist at inference, which is
    a plausible structural reason wall F1 sits at ~0.31.
2.  **Numbers were bare black text.** Real waypoints are a solid black disc with the
    number knocked out in white.
3.  **``ImageFont.truetype("arial.ttf")`` with a silent fallback.** Linux has no
    ``arial.ttf``, so the same code renders differently on Colab than on Windows and the
    "deterministic from a seed" property quietly breaks across machines. This module
    uses ``ImageFont.load_default(size=...)``, which since Pillow 10.1 returns a
    scalable **Aileron** face bundled inside Pillow itself -- no system font, no vendored
    file, no licensing question, identical on every platform.

Measured wall counts in the six ground-truth screenshots (``src/core/tests/conftest.py``):
6x6 boards carry 0, 4, 4 and 10 walls; 7x7 boards carry 0 and 14. Two of the six have
none at all, which is why the dataset has to include wall-free boards -- a model that
never sees one has no reason to learn restraint, and a hallucinated wall makes a puzzle
unsolvable just as surely as a missed one does.
"""

from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from src.core.utils import Puzzle

DEFAULT_CELL_SIZE = 100
BOARD_PADDING_RATIO = 0.28
BOARD_CORNER_RATIO = 0.22

GRID_LINE_RATIO = 0.02
BOARD_BORDER_RATIO = 0.022
WALL_THICKNESS_RATIO = 0.11
WALL_LENGTH_RATIO = 1.0

WAYPOINT_RADIUS_RATIO = 0.33
WAYPOINT_FONT_RATIO = 0.40
WAYPOINT_BOLD_STROKE_RATIO = 0.022

BUTTON_HEIGHT_RATIO = 0.62
BUTTON_GAP_RATIO = 0.18
BUTTON_FONT_RATIO = 0.30
BUTTON_CORNER_RATIO = 0.31

CURSOR_SIZE_RATIO = 0.16

MIN_LINE_PIXELS = 1


@dataclass(frozen=True)
class RenderTheme:
    """Colours only. Geometry is shared so the two themes stay directly comparable."""

    name: str
    background: tuple[int, int, int]
    board_fill: tuple[int, int, int]
    board_border: tuple[int, int, int]
    grid_line: tuple[int, int, int]
    wall: tuple[int, int, int]
    waypoint_fill: tuple[int, int, int]
    waypoint_text: tuple[int, int, int]
    blocked_fill: tuple[int, int, int]
    button_fill: tuple[int, int, int]
    button_border: tuple[int, int, int]
    button_text: tuple[int, int, int]
    cursor: tuple[int, int, int]


LIGHT_THEME = RenderTheme(
    name="light",
    background=(245, 245, 243),
    board_fill=(252, 252, 250),
    board_border=(196, 196, 194),
    grid_line=(176, 176, 174),
    wall=(17, 17, 17),
    waypoint_fill=(17, 17, 17),
    waypoint_text=(255, 255, 255),
    blocked_fill=(40, 40, 40),
    button_fill=(222, 222, 220),
    button_border=(180, 180, 178),
    button_text=(90, 90, 88),
    cursor=(60, 60, 60),
)

DARK_THEME = RenderTheme(
    name="dark",
    background=(24, 24, 26),
    board_fill=(34, 34, 37),
    board_border=(78, 78, 82),
    grid_line=(92, 92, 96),
    wall=(240, 240, 240),
    waypoint_fill=(240, 240, 240),
    waypoint_text=(20, 20, 22),
    blocked_fill=(120, 120, 124),
    button_fill=(52, 52, 56),
    button_border=(86, 86, 90),
    button_text=(190, 190, 194),
    cursor=(220, 220, 220),
)

THEMES = {theme.name: theme for theme in (LIGHT_THEME, DARK_THEME)}

BUTTON_LABELS = ("Undo", "Hint")


def _px(ratio: float, cell_size: int) -> int:
    return max(MIN_LINE_PIXELS, round(ratio * cell_size))


def render_puzzle(
    puzzle: Puzzle,
    cell_size: int = DEFAULT_CELL_SIZE,
    theme: RenderTheme = LIGHT_THEME,
    show_buttons: bool = True,
    show_cursor: bool = False,
    cursor_cell: tuple[int, int] | None = None,
) -> Image.Image:
    """Renders an unsolved puzzle as a screenshot-like image."""
    height, width = puzzle["grid_size"]
    padding = _px(BOARD_PADDING_RATIO, cell_size)

    board_width = width * cell_size
    board_height = height * cell_size
    image_width = board_width + 2 * padding
    image_height = board_height + 2 * padding
    if show_buttons:
        image_height += _px(BUTTON_HEIGHT_RATIO, cell_size) + _px(
            BUTTON_GAP_RATIO, cell_size
        )

    image = Image.new("RGB", (image_width, image_height), theme.background)
    draw = ImageDraw.Draw(image)

    board_box = (padding, padding, padding + board_width, padding + board_height)
    _draw_board(draw, board_box, cell_size, height, width, theme)
    _draw_blocked_cells(draw, puzzle, board_box, cell_size, theme)
    _draw_walls(draw, puzzle, board_box, cell_size, theme)
    _draw_waypoints(draw, puzzle, board_box, cell_size, theme)

    if show_buttons:
        _draw_buttons(draw, board_box, cell_size, theme)
    if show_cursor:
        _draw_cursor(draw, board_box, cell_size, height, width, theme, cursor_cell)

    return image


def _draw_board(
    draw: ImageDraw.ImageDraw,
    board_box: tuple[int, int, int, int],
    cell_size: int,
    height: int,
    width: int,
    theme: RenderTheme,
) -> None:
    """Rounded outer border, light grid lines -- the contrast walls stand out against."""
    left, top, right, bottom = board_box
    draw.rounded_rectangle(
        board_box,
        radius=_px(BOARD_CORNER_RATIO, cell_size),
        fill=theme.board_fill,
        outline=theme.board_border,
        width=_px(BOARD_BORDER_RATIO, cell_size),
    )

    grid_width = _px(GRID_LINE_RATIO, cell_size)
    for column in range(1, width):
        x = left + column * cell_size
        draw.line([(x, top), (x, bottom)], fill=theme.grid_line, width=grid_width)
    for row in range(1, height):
        y = top + row * cell_size
        draw.line([(left, y), (right, y)], fill=theme.grid_line, width=grid_width)


def _draw_blocked_cells(
    draw: ImageDraw.ImageDraw,
    puzzle: Puzzle,
    board_box: tuple[int, int, int, int],
    cell_size: int,
    theme: RenderTheme,
) -> None:
    left, top, _, _ = board_box
    for row, column in sorted(puzzle["blocked_cells"]):
        x0 = left + column * cell_size
        y0 = top + row * cell_size
        draw.rectangle(
            [(x0, y0), (x0 + cell_size, y0 + cell_size)], fill=theme.blocked_fill
        )


def _draw_walls(
    draw: ImageDraw.ImageDraw,
    puzzle: Puzzle,
    board_box: tuple[int, int, int, int],
    cell_size: int,
    theme: RenderTheme,
) -> None:
    """Heavy bars centred on the shared edge, the way the real UI draws them."""
    left, top, _, _ = board_box
    thickness = _px(WALL_THICKNESS_RATIO, cell_size)
    half = thickness / 2
    length = WALL_LENGTH_RATIO * cell_size
    overhang = (length - cell_size) / 2

    for (row1, column1), (row2, column2) in sorted(puzzle["walls"]):
        if row1 == row2:  # vertical wall on the edge between two side-by-side cells
            x = left + max(column1, column2) * cell_size
            y0 = top + row1 * cell_size - overhang
            draw.rectangle([(x - half, y0), (x + half, y0 + length)], fill=theme.wall)
        else:  # horizontal wall between two stacked cells
            y = top + max(row1, row2) * cell_size
            x0 = left + column1 * cell_size - overhang
            draw.rectangle([(x0, y - half), (x0 + length, y + half)], fill=theme.wall)


def _draw_waypoints(
    draw: ImageDraw.ImageDraw,
    puzzle: Puzzle,
    board_box: tuple[int, int, int, int],
    cell_size: int,
    theme: RenderTheme,
) -> None:
    """Solid disc with the number knocked out in white."""
    left, top, _, _ = board_box
    radius = WAYPOINT_RADIUS_RATIO * cell_size
    font = ImageFont.load_default(size=_px(WAYPOINT_FONT_RATIO, cell_size))
    stroke = _px(WAYPOINT_BOLD_STROKE_RATIO, cell_size)

    for number, (row, column) in sorted(puzzle["num_map"].items()):
        centre_x = left + column * cell_size + cell_size / 2
        centre_y = top + row * cell_size + cell_size / 2
        draw.ellipse(
            [
                (centre_x - radius, centre_y - radius),
                (centre_x + radius, centre_y + radius),
            ],
            fill=theme.waypoint_fill,
        )
        draw.text(
            (centre_x, centre_y),
            str(number),
            font=font,
            fill=theme.waypoint_text,
            anchor="mm",
            stroke_width=stroke,
            stroke_fill=theme.waypoint_text,
        )


def _draw_buttons(
    draw: ImageDraw.ImageDraw,
    board_box: tuple[int, int, int, int],
    cell_size: int,
    theme: RenderTheme,
) -> None:
    left, _, right, bottom = board_box
    button_height = _px(BUTTON_HEIGHT_RATIO, cell_size)
    gap = _px(BUTTON_GAP_RATIO, cell_size)
    top = bottom + gap
    half_width = (right - left - gap) / 2
    font = ImageFont.load_default(size=_px(BUTTON_FONT_RATIO, cell_size))
    radius = _px(BUTTON_CORNER_RATIO, cell_size)

    for index, label in enumerate(BUTTON_LABELS):
        x0 = left + index * (half_width + gap)
        box = (x0, top, x0 + half_width, top + button_height)
        draw.rounded_rectangle(
            box,
            radius=radius,
            fill=theme.button_fill if index == 0 else theme.board_fill,
            outline=theme.button_border,
            width=_px(BOARD_BORDER_RATIO, cell_size),
        )
        draw.text(
            ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
            label,
            font=font,
            fill=theme.button_text,
            anchor="mm",
        )


def _draw_cursor(
    draw: ImageDraw.ImageDraw,
    board_box: tuple[int, int, int, int],
    cell_size: int,
    height: int,
    width: int,
    theme: RenderTheme,
    cursor_cell: tuple[int, int] | None,
) -> None:
    """The mouse pointer artefact that shows up in real screenshots."""
    left, top, _, _ = board_box
    row, column = cursor_cell if cursor_cell else (height - 1, width - 1)
    size = CURSOR_SIZE_RATIO * cell_size
    x = left + column * cell_size + cell_size / 2
    y = top + row * cell_size + cell_size / 2
    draw.polygon(
        [
            (x, y),
            (x, y + size * 1.6),
            (x + size * 0.42, y + size * 1.18),
            (x + size * 0.72, y + size * 1.75),
            (x + size * 0.95, y + size * 1.62),
            (x + size * 0.66, y + size * 1.06),
            (x + size * 1.05, y + size * 0.98),
        ],
        fill=theme.cursor,
    )
