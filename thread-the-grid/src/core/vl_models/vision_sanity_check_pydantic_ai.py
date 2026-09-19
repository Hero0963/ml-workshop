# src/core/vl_models/vision_sanity_check_pydantic_ai.py

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


# 1. Define the desired structured output using Pydantic
class AnimalInfo(BaseModel):
    """A model to store the identified animal and its color."""

    animal_name: str = Field(
        ..., description="The name of the animal, e.g., 'cat', 'bird'."
    )
    primary_color: str = Field(..., description="The dominant color of the animal.")


def run_structured_vision_test(image_path: Path):
    """Runs a structured vision test using the pydantic_ai Agent."""
    if not image_path.exists():
        print(f"ERROR: Image file not found at '{image_path}'")
        return

    print("\n" + "=" * 80)
    print(f"--- Running Structured Vision Test for: {image_path.name} ---")

    try:
        # 2. Load settings
        settings = get_settings()
        model_name = settings.OLLAMA_MODEL_NAME
        ollama_url = "http://localhost:11434/v1"

        print(f"  Ollama URL: {ollama_url}")
        print(f"  Model: {model_name}")

        # 3. Instantiate the Agent, setting output_type to our Pydantic model
        agent = Agent(
            model=OpenAIChatModel(
                model_name=model_name,
                provider=OllamaProvider(base_url=ollama_url),
            ),
            output_type=AnimalInfo,
        )

        # 4. Prepare the image content and prompt
        image_bytes = image_path.read_bytes()
        media_type = f"image/{image_path.suffix.lstrip('.')}"
        image_content = BinaryContent(data=image_bytes, media_type=media_type)

        prompt_text = "Analyze the image and identify the animal and its primary color."

        # 5. Call agent.run_sync() with the mixed-content list
        print("Sending request via pydantic_ai.Agent.run_sync([prompt, content])...")
        result = agent.run_sync([prompt_text, image_content])

        # 6. Print the structured result
        print("\n--- Model Response (Structured) ---")
        if isinstance(result.output, AnimalInfo):
            print(f"  - Animal Name:   {result.output.animal_name}")
            print(f"  - Primary Color: {result.output.primary_color}")
            print("\nSUCCESS: Model returned a valid Pydantic object.")
        else:
            print("FAILURE: Model did not return the expected Pydantic object.")
            print(f"Raw output: {result.output}")

        print("-" * 20)
        print(f"--- Test for {image_path.name} FINISHED ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    ILLUSTRATIONS_DIR = PROJECT_ROOT / "illustrations"
    cat_image_path = ILLUSTRATIONS_DIR / "cat.png"
    bird_image_path = ILLUSTRATIONS_DIR / "bird.png"

    print("Starting structured vision test using pydantic_ai Agent...")

    run_structured_vision_test(cat_image_path)
    run_structured_vision_test(bird_image_path)
