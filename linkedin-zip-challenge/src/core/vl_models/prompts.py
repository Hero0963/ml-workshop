# src/core/vl_models/prompts.py

import json

# This data structure now matches the Pydantic models in hf_parser.py
few_shot_examples = [
    {
        "image_path": "illustrations/puzzle_01.png",
        "output": {
            "layout": [
                ["  ", "  ", "  ", "  ", "  ", "  "],
                ["  ", "01", "  ", "  ", "02", "  "],
                ["  ", "  ", "03", "04", "  ", "  "],
                ["  ", "  ", "06", "05", "  ", "  "],
                ["  ", "08", "  ", "  ", "07", "  "],
                ["  ", "  ", "  ", "  ", "  ", "  "],
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
                {"cell1": [4, 4], "cell2": [5, 4]},
            ],
        },
    },
    {
        "image_path": "illustrations/puzzle_02.png",
        "output": {
            "layout": [
                ["  ", "  ", "05", "08", "  ", "  "],
                ["  ", "12", "  ", "  ", "09", "  "],
                ["04", "  ", "  ", "  ", "  ", "01"],
                ["  ", "  ", "06", "07", "  ", "  "],
                ["  ", "11", "  ", "  ", "10", "  "],
                ["03", "  ", "  ", "  ", "  ", "02"],
            ],
            "walls": [],
        },
    },
    {
        "image_path": "illustrations/puzzle_03.png",
        "output": {
            "layout": [
                ["12", "  ", "11", "  ", "09", "  "],
                ["02", "  ", "01", "  ", "10", "  "],
                ["  ", "  ", "  ", "  ", "  ", "  "],
                ["  ", "  ", "  ", "  ", "  ", "  "],
                ["  ", "08", "  ", "07", "  ", "06"],
                ["  ", "03", "  ", "04", "  ", "05"],
            ],
            "walls": [
                {"cell1": [2, 1], "cell2": [3, 1]},
                {"cell1": [2, 2], "cell2": [3, 2]},
                {"cell1": [2, 3], "cell2": [3, 3]},
                {"cell1": [2, 4], "cell2": [3, 4]},
            ],
        },
    },
    {
        "image_path": "illustrations/puzzle_04.png",
        "output": {
            "layout": [
                ["  ", "  ", "  ", "  ", "  ", "  ", "  "],
                ["  ", "07", "  ", "06", "  ", "10", "  "],
                ["  ", "  ", "08", "  ", "09", "  ", "  "],
                ["  ", "12", "  ", "  ", "  ", "11", "  "],
                ["  ", "  ", "03", "  ", "02", "  ", "  "],
                ["  ", "04", "  ", "05", "  ", "01", "  "],
                ["  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ],
            "walls": [
                {"cell1": [0, 3], "cell2": [1, 3]},
                {"cell1": [1, 3], "cell2": [2, 3]},
                {"cell1": [2, 1], "cell2": [3, 1]},
                {"cell1": [2, 2], "cell2": [3, 2]},
                {"cell1": [2, 3], "cell2": [3, 3]},
                {"cell1": [2, 4], "cell2": [3, 4]},
                {"cell1": [2, 5], "cell2": [3, 5]},
                {"cell1": [3, 1], "cell2": [4, 1]},
                {"cell1": [3, 2], "cell2": [4, 2]},
                {"cell1": [3, 3], "cell2": [4, 3]},
                {"cell1": [3, 4], "cell2": [4, 4]},
                {"cell1": [3, 5], "cell2": [4, 5]},
                {"cell1": [4, 3], "cell2": [5, 3]},
                {"cell1": [5, 3], "cell2": [6, 3]},
            ],
        },
    },
    {
        "image_path": "illustrations/puzzle_05.png",
        "output": {
            "layout": [
                ["12", "11", "  ", "  ", "  ", "  "],
                ["  ", "10", "  ", "  ", "  ", "  "],
                ["09", "08", "07", "  ", "  ", "  "],
                ["  ", "  ", "  ", "05", "04", "  "],
                ["  ", "  ", "  ", "  ", "03", "  "],
                ["  ", "  ", "  ", "06", "01", "02"],
            ],
            "walls": [
                {"cell1": [1, 3], "cell2": [1, 4]},
                {"cell1": [1, 4], "cell2": [1, 5]},
                {"cell1": [4, 0], "cell2": [4, 1]},
                {"cell1": [4, 1], "cell2": [4, 2]},
            ],
        },
    },
    {
        "image_path": "illustrations/puzzle_06.png",
        "output": {
            "layout": [
                ["  ", "  ", "  ", "  ", "  ", "  ", "  "],
                ["04", "  ", "05", "17", "  ", "18", "06"],
                ["03", "  ", "  ", "16", "  ", "  ", "07"],
                ["13", "  ", "02", "01", "  ", "19", "08"],
                ["12", "  ", "15", "  ", "  ", "  ", "09"],
                ["11", "  ", "14", "21", "  ", "20", "10"],
                ["  ", "  ", "  ", "  ", "  ", "  ", "  "],
            ],
            "walls": [],
        },
    },
]


def get_extraction_prompt() -> str:
    """
    Constructs the prompt for the VL model, including few-shot examples.
    """
    base_instruction = """
    You are an expert at analyzing puzzle images. Your task is to first write down your step-by-step thinking process, and then provide a single markdown code block containing a JSON object.
    Do not use any tools. 

    The JSON object must have exactly two top-level keys: "layout" and "walls".

    1.  **"layout" key**: This must be a 2D array representing the grid.
        -   Empty cells are represented by "  " (two spaces).
        -   Numbered waypoints are represented by a two-digit string, e.g., "01", "07", "12".
        -   Obstacles (e.g., solid black squares) are represented by "xx".

    2.  **"walls" key**: This must be a list of wall objects.
        -   A wall is a barrier *between* two adjacent cells.
        -   Each wall object has two keys, "cell1" and "cell2", which hold the [row, col] coordinates of the cells it separates.

    Here are some examples of the required format for the JSON block:
    """

    examples_str = ""
    for example in few_shot_examples:
        # The image_path is a placeholder for the prompt, the actual image is sent separately.
        examples_str += f"""
    Image: {example['image_path']}
    Output:
    ```json
    {json.dumps(example['output'], indent=4)}
    ```
    """

    final_prompt = f"""{base_instruction.strip()}
{examples_str.strip()}

    Now, analyze the provided image and output the JSON in the exact same format.
    """
    return final_prompt.strip()
