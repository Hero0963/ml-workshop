# src/core/vl_models/prompt_baseline.py
"""The frozen baseline prompt and the few-shot answers it is built from.

Every number in ``ai-collab/reports/2026-08-15_vl-p0-p1-baseline.html`` was measured
against ``build_puzzle_prompt()``. Editing it in place would silently invalidate that
comparison, so it is pinned by ``test_prompt_baseline.py`` -- a hash test fails the
moment the rendered text changes. New ideas belong in ``prompt_variants.py``.

This module used to live inside ``final_puzzle_parser.py``. It was lifted out so the
frozen assets do not depend on a scratchpad script; ``final_puzzle_parser`` re-exports
them and the rendered text is unchanged.

Leakage warning: ``puzzle_01`` .. ``puzzle_03`` are also evaluation images, so their
scores are not evidence of generalisation. See the handover's trap list.
"""

PUZZLE_01_JSON_STR = """{
  "layout": [
    ["  ", "  ", "  ", "  ", "  ", "  "],
    ["  ", "01", "  ", "  ", "02", "  "],
    ["  ", "  ", "03", "04", "  ", "  "],
    ["  ", "  ", "06", "05", "  ", "  "],
    ["  ", "08", "  ", "  ", "07", "  "],
    ["  ", "  ", "  ", "  ", "  ", "  "]
  ],
  "walls": [
    {"cell1": [1, 0], "cell2": [1, 1]},
    {"cell1": [0, 1], "cell2": [1, 1]},
    {"cell1": [2, 1], "cell2": [2, 2]},
    {"cell1": [1, 3], "cell2": [2, 3]},
    {"cell1": [1, 4], "cell2": [2, 4]},
    {"cell1": [3, 3], "cell2": [3, 4]},
    {"cell1": [3, 1], "cell2": [4, 1]},
    {"cell1": [3, 2], "cell2": [4, 2]},
    {"cell1": [4, 4], "cell2": [4, 5]},
    {"cell1": [4, 4], "cell2": [5, 4]}
  ]
}"""

PUZZLE_02_JSON_STR = """{
  "layout": [
    ["  ", "  ", "05", "08", "  ", "  "],
    ["  ", "12", "  ", "  ", "09", "  "],
    ["04", "  ", "  ", "  ", "  ", "01"],
    ["  ", "  ", "06", "07", "  ", "  "],
    ["  ", "11", "  ", "  ", "10", "  "],
    ["03", "  ", "  ", "  ", "  ", "02"]
  ],
  "walls": []
}"""

PUZZLE_03_JSON_STR = """{
  "layout": [
    ["12", "  ", "11", "  ", "09", "  "],
    ["02", "  ", "01", "  ", "10", "  "],
    ["  ", "  ", "  ", "  ", "  ", "  "],
    ["  ", "  ", "  ", "  ", "  ", "  "],
    ["  ", "08", "  ", "07", "  ", "06"],
    ["  ", "03", "  ", "04", "  ", "05"]
  ],
  "walls": [
    {"cell1": [2, 1], "cell2": [3, 1]},
    {"cell1": [2, 2], "cell2": [3, 2]},
    {"cell1": [2, 3], "cell2": [3, 3]},
    {"cell1": [2, 4], "cell2": [3, 4]}
  ]
}"""


def build_puzzle_prompt() -> str:
    """Builds the few-shot prompt to instruct the model to return a JSON string."""
    return f"""
You are an expert Zip puzzle analyzer. Your task is to analyze the provided image and respond with ONLY a single JSON object in a markdown code block.

The JSON object must have two keys: "layout" and "walls".
- "layout": A 2D array of strings representing the grid.
- "walls": A list of objects, where each object represents a wall between two cells.

Here are some examples of the required output format.

--- EXAMPLE 1 ---
```json
{PUZZLE_01_JSON_STR.strip()}
```

--- EXAMPLE 2 ---
```json
{PUZZLE_02_JSON_STR.strip()}
```

--- EXAMPLE 3 ---
```json
{PUZZLE_03_JSON_STR.strip()}
```

--- TASK ---
Now, analyze the new image provided and generate the corresponding JSON object in the exact same format. Do not include any other text, explanations, or apologies in your response.
"""
