# src/core/vl_models/tool_calling_poc.py
# POC = Proof of Concept
import sys
from pathlib import Path

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ---

try:
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent
    from pydantic_ai.messages import BinaryContent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from src.settings import get_settings
except ImportError as e:
    print(f"Failed to import modules: {e}")
    exit(1)


class IdentifiedAnimal(BaseModel):
    """A simple model to store the output of an animal identification task."""

    animal_name: str = Field(
        ..., description="The name of the animal, e.g., 'cat', 'dog'."
    )
    color: str = Field(..., description="The primary color of the animal.")
    confidence: float = Field(
        ...,
        description="The confidence score of the identification, from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )


def run_tool_calling_test(image_path: Path):
    """Runs a tool-calling test for the given image."""
    if not image_path.exists():
        print(f"ERROR: Image file not found at '{image_path}'")
        return

    print("\n" + "=" * 80)
    print(f"--- Running Tool-Calling POC for: {image_path.name} ---")

    try:
        # 1. Load settings
        settings = get_settings()
        model_name = settings.OLLAMA_MODEL_NAME
        ollama_url = "http://localhost:11434/v1"

        print(f"  Ollama URL: {ollama_url}")
        print(f"  Model: {model_name}")

        # 2. Initialize the pydantic_ai Agent
        # This time, we set the output_type directly to our Pydantic model
        agent = Agent(
            model=OpenAIChatModel(
                model_name=model_name,
                provider=OllamaProvider(base_url=ollama_url),
            ),
            output_type=IdentifiedAnimal,  # The agent will force the model to use this as a tool
        )

        # 3. Define the prompt and image content
        image_bytes = image_path.read_bytes()
        media_type = f"image/{image_path.suffix.lstrip('.')}"
        image_content = BinaryContent(data=image_bytes, media_type=media_type)

        messages = [
            {
                "role": "user",
                "content": [
                    "Identify the animal in this image, its primary color, and your confidence level.",
                    image_content,
                ],
            }
        ]

        # 4. Run the agent
        result = agent.run_sync(message_history=messages)

        # 5. Print the structured result
        print("\n--- Tool-Calling Result (Structured Object) ---")
        if isinstance(result.output, IdentifiedAnimal):
            print(f"  - Identified Animal: {result.output.animal_name}")
            print(f"  - Confidence Score:  {result.output.confidence:.2f}")
            print("  - SUCCESS: Model returned a valid Pydantic object.")
        else:
            print("  - FAILED: Model did not return the expected Pydantic object.")
            print(f"  - Raw output: {result.output}")
        print("-" * 20)
        print(f"--- POC for {image_path.name} FINISHED ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    ILLUSTRATIONS_DIR = PROJECT_ROOT / "illustrations"
    # This test will use the cat image. You can change it to dog.jpg.
    test_image = ILLUSTRATIONS_DIR / "cat.jpg"

    print("Starting Tool-Calling Proof of Concept (POC)...")
    print("This test will check if the model can return a structured Pydantic object.")

    run_tool_calling_test(test_image)
