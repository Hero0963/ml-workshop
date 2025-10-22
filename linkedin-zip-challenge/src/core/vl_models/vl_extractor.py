# src/core/vl_models/vl_extractor.py

import json
import re
from pathlib import Path
from typing import List

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.messages import BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from src.core.vl_models.prompts import get_extraction_prompt


def extract_json_block(text: str) -> str | None:
    """Extracts a JSON block from a markdown string."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1)

    # Fallback: if no markdown block is found, find the first and last curly brace
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index : end_index + 1]

    return None


# --- Pydantic Models for Validation ---


class WallPair(BaseModel):
    """Represents a wall between two adjacent cells."""

    cell1: List[int] = Field(
        ..., description="The [row, col] coordinates of the first cell."
    )
    cell2: List[int] = Field(
        ..., description="The [row, col] coordinates of the second, adjacent cell."
    )


class PuzzleExtractionOutput(BaseModel):
    """The structured output for puzzle data, used for validation."""

    thinking_process: str = Field(
        ..., description="The model's step-by-step reasoning process."
    )
    layout: List[List[str]] = Field(
        ...,
        description=(
            "A 2D array representing the grid. Use '  ' for empty cells, 'xx' for obstacles, "
            "and two-digit strings like '01', '12' for numbered waypoints."
        ),
    )
    walls: List[WallPair] = Field(
        ...,
        description="A list of wall objects, where each object represents a wall between two cells.",
    )


class VLExtractor:
    """
    Extractor for identifying puzzle layout and walls from an image by parsing
    a JSON string from a VL model's text response.
    """

    def __init__(self, ollama_provider_url: str, ollama_model_name: str):
        self.ollama_provider_url = ollama_provider_url
        self.ollama_model_name = ollama_model_name

        self.ollama_model = OpenAIChatModel(
            model_name=self.ollama_model_name,
            provider=OllamaProvider(base_url=self.ollama_provider_url),
        )

        # The agent is now configured to expect a string output
        self.extractor_agent = Agent(
            self.ollama_model,
            system_prompt=get_extraction_prompt(),
            output_type=str,  # Expect a raw string
        )

    def extract_from_image(self, image_path: Path) -> PuzzleExtractionOutput:
        """
        Sends an image to the Ollama service, gets a string response, separates
        the thinking process from the JSON, parses and validates the JSON, and
        returns the structured Pydantic object.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        image_bytes = image_path.read_bytes()
        media_type = f"image/{image_path.suffix.lstrip('.')}"
        image_content = BinaryContent(data=image_bytes, media_type=media_type)

        messages = [
            {
                "role": "user",
                "content": [
                    "You MUST analyze the provided image. Do not invent a puzzle. Your response MUST be based on the content of the image. First, write your thinking process describing what you see in the image, then provide the JSON output as requested.",
                    image_content,
                ],
            }
        ]

        # 1. Print the raw content being sent to the model
        print("\n" + "-" * 30)
        print("  1. Content Sent to Model  ")
        print("-" * 30)
        # Note: We don't print the full image bytes, just the structure
        print(
            [
                {
                    "role": m["role"],
                    "content": [
                        c
                        if not isinstance(c, BinaryContent)
                        else "BinaryContent(image)"
                        for c in m["content"]
                    ],
                }
                for m in messages
            ]
        )

        try:
            # Get the raw string response from the agent
            agent_result = self.extractor_agent.run_sync(message_history=messages)
            raw_response_str = agent_result.output

            # 2. Print the raw content returned by the model
            print("\n" + "-" * 30)
            print("  2. Raw Response from Model  ")
            print("-" * 30)
            print(raw_response_str)

            if not raw_response_str:
                raise ValueError("Received an empty response from the model.")

            # Extract the JSON block from the string
            json_str = extract_json_block(raw_response_str)
            if not json_str:
                raise ValueError(
                    f"Could not find a JSON block in the model's response. Response: \n{raw_response_str}"
                )

            # The text before the JSON block is the thinking process
            thinking_process = raw_response_str.split("```json")[0].strip()

            # Parse the JSON string into a Python dictionary
            data = json.loads(json_str)

            # Validate the dictionary and create the final object
            data["thinking_process"] = thinking_process
            validated_output = PuzzleExtractionOutput(**data)

            # 3. Print the parsed and validated content
            print("\n" + "-" * 30)
            print("  3. Parsed and Validated Content  ")
            print("-" * 30)
            print(validated_output)

            return validated_output

        except (ValidationError, json.JSONDecodeError) as e:
            raise RuntimeError(
                f"Failed to parse or validate the model's JSON output: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Error during extraction with pydantic_ai agent: {e}")
