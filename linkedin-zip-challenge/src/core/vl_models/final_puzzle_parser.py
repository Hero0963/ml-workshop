# src/core/vl_models/final_puzzle_parser.py

import sys
import json
from pathlib import Path
from typing import Any

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ---

try:
    from pydantic import BaseModel, Field, ValidationError
    from pydantic_ai import Agent
    from pydantic_ai.messages import BinaryContent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from src.core.utils import parse_puzzle_layout
except ImportError as e:
    print(f"Failed to import modules: {e}")
    exit(1)


# --- 1. Re-usable components from hf_parser.py and parser.py ---


class WallPair(BaseModel):
    """Represents a wall between two adjacent cells."""

    cell1: list[int] = Field(description="Coordinates [row, col] of the first cell.")
    cell2: list[int] = Field(
        description="Coordinates [row, col] of the second, adjacent cell."
    )


class SimplePuzzleOutput(BaseModel):
    """The JSON structure the model is instructed to generate."""

    layout: list[list[str]] = Field(
        description="2D array representing the grid. Use '  ' for empty cells, and two-digit strings like '01' for numbers."
    )
    walls: list[WallPair] = Field(description="A list of wall objects.")


def extract_json_block(text: str) -> str | None:
    """Extracts a JSON block from a markdown string."""
    match = __import__("re").search(
        r"```json\s*(\{.*\})\s*```", text, __import__("re").DOTALL
    )
    if match:
        return match.group(1)
    # Fallback for non-markdown responses
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index : end_index + 1]
    return None


def post_process_data(mllm_output: SimplePuzzleOutput) -> dict[str, Any]:
    """Converts the model's simple JSON output into the final solver-ready format."""
    final_puzzle_data = parse_puzzle_layout(mllm_output.layout)
    wall_set: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    if mllm_output.walls:
        for wall_pair in mllm_output.walls:
            cell1 = tuple(wall_pair.cell1)
            cell2 = tuple(wall_pair.cell2)
            standardized_wall = tuple(sorted((cell1, cell2)))
            wall_set.add(standardized_wall)
    final_puzzle_data["walls"] = wall_set
    return final_puzzle_data


# --- 2. Few-Shot Prompt Generation ---

# Extracted from conftest.py
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


# --- 3. Main Parsing Logic ---


def parse_puzzle_image(image_path: Path) -> dict[str, Any] | None:
    """Uses the Hybrid Strategy (pydantic-ai + prompt engineering) to parse a puzzle image."""
    print("\n" + "=" * 80)
    print(f"--- Starting Puzzle Parsing for: {image_path.name} ---")

    try:
        model_name = "openbmb/minicpm-o2.6"
        ollama_url = "http://localhost:11434/v1"

        agent = Agent(
            model=OpenAIChatModel(
                model_name=model_name, provider=OllamaProvider(base_url=ollama_url)
            ),
            output_type=str,
        )

        image_bytes = image_path.read_bytes()
        media_type = f"image/{image_path.suffix.lstrip('.')}"
        image_content = BinaryContent(data=image_bytes, media_type=media_type)

        prompt_text = build_puzzle_prompt()

        print(f"Sending request to {model_name}...")
        result = agent.run_sync([prompt_text, image_content])
        model_output_str = result.output
        print(f"\n--- Raw Model Output ---\n{model_output_str}")

        print("\n--- Parsing and Validation ---")
        json_str = extract_json_block(model_output_str)
        if not json_str:
            raise ValueError("Could not extract JSON block from model output.")

        parsed_data = json.loads(json_str)
        validated_output = SimplePuzzleOutput(**parsed_data)
        print("SUCCESS: Model output conforms to SimplePuzzleOutput schema.")

        print("\n--- Post-Processing ---")
        final_data = post_process_data(validated_output)
        print("SUCCESS: Post-processing complete.")
        return final_data

    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        print(
            f"FAILURE: Failed to parse, validate, or process model output. Error: {e}"
        )
        return None
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return None


# --- 4. Execution ---

if __name__ == "__main__":
    illustrations_dir = PROJECT_ROOT / "illustrations"

    # Test 1: Use puzzle_01.png
    # This tests if the model can reproduce one of the examples it was shown.
    puzzle_01_path = illustrations_dir / "puzzle_01.png"
    final_data_01 = parse_puzzle_image(puzzle_01_path)
    if final_data_01:
        print("\n--- Final Solver-Ready Data for Puzzle 01 ---")
        print(
            json.dumps(
                final_data_01,
                default=lambda o: sorted(list(o)) if isinstance(o, set) else str(o),
                indent=2,
            )
        )

    # Test 2: Use puzzle_04.png
    # This tests if the model can generalize to a new puzzle.
    puzzle_04_path = illustrations_dir / "puzzle_04.png"
    final_data_04 = parse_puzzle_image(puzzle_04_path)
    if final_data_04:
        print("\n--- Final Solver-Ready Data for Puzzle 04 ---")
        print(
            json.dumps(
                final_data_04,
                default=lambda o: sorted(list(o)) if isinstance(o, set) else str(o),
                indent=2,
            )
        )
