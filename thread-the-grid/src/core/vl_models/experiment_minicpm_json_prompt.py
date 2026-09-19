# src/core/vl_models/experiment_minicpm_json_prompt.py

import sys
import json
from pathlib import Path

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ---

try:
    from pydantic_ai import Agent
    from pydantic_ai.messages import BinaryContent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider
except ImportError as e:
    print(f"Failed to import modules: {e}")
    exit(1)


def build_json_prompt() -> str:
    """Builds a prompt to instruct the model to return a JSON string."""
    return """
    Analyze the provided image and respond with ONLY a single JSON object.
    The JSON object must contain two keys: "animal_name" and "primary_color".

    EXAMPLE OF YOUR REQUIRED OUTPUT:
    ```json
    {
      "animal_name": "cat",
      "primary_color": "black"
    }
    ```

    Do not include any other text, explanations, or apologies in your response. Just the JSON object.
    Now, analyze the new image and generate the corresponding JSON object.
    """


def run_json_prompt_test(image_path: Path):
    """Tests if the model can return a JSON string based on a prompt instruction."""
    if not image_path.exists():
        print(f"ERROR: Image file not found at '{image_path}'")
        return

    print("\n" + "=" * 80)
    print(f"--- Running JSON Prompt Test for: {image_path.name} ---")

    try:
        # 1. Hardcode model name and prepare agent
        model_name = "openbmb/minicpm-o2.6"
        ollama_url = "http://localhost:11434/v1"

        print(f"  Ollama URL: {ollama_url}")
        print(f"  Model: {model_name}")

        # The agent expects a string output, which we hope is a JSON string
        agent = Agent(
            model=OpenAIChatModel(
                model_name=model_name,
                provider=OllamaProvider(base_url=ollama_url),
            ),
            output_type=str,
        )

        # 2. Prepare image content and the special JSON-requesting prompt
        image_bytes = image_path.read_bytes()
        media_type = f"image/{image_path.suffix.lstrip('.')}"
        image_content = BinaryContent(data=image_bytes, media_type=media_type)

        prompt_text = build_json_prompt()

        # 3. Call agent.run_sync() as discovered
        print("Sending request with JSON-instructing prompt...")
        result = agent.run_sync([prompt_text, image_content])
        model_output_str = result.output

        print(f"\n--- Raw Model Output ---\n{model_output_str}")

        # 4. Try to parse the string as JSON
        print("\n--- Parsing Result ---")
        try:
            # The model might return the JSON within a markdown block
            if "```json" in model_output_str:
                json_str = model_output_str.split("```json")[1].split("```")[0].strip()
            else:
                json_str = model_output_str

            parsed_json = json.loads(json_str)
            print("SUCCESS: Successfully parsed JSON from model output.")
            print(parsed_json)
        except (json.JSONDecodeError, IndexError) as e:
            print(f"FAILURE: Could not parse JSON from model output. Error: {e}")

        print("-" * 20)
        print(f"--- Test for {image_path.name} FINISHED ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    ILLUSTRATIONS_DIR = PROJECT_ROOT / "illustrations"
    cat_image_path = ILLUSTRATIONS_DIR / "cat.png"
    bird_image_path = ILLUSTRATIONS_DIR / "bird.png"

    print("Starting test to get JSON via prompt instruction...")

    run_json_prompt_test(cat_image_path)
    run_json_prompt_test(bird_image_path)
