import sys
import base64
from pathlib import Path

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ---

try:
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from src.settings import get_settings
except ImportError as e:
    print(f"Failed to import modules: {e}")
    exit(1)


def run_vision_test(image_path: Path):
    """Runs a simple vision test for the given image."""
    if not image_path.exists():
        print(f"ERROR: Image file not found at '{image_path}'")
        print("Please place your images in the 'illustrations' directory.")
        return

    print("\n" + "=" * 80)
    print(f"--- Running Sanity Check for: {image_path.name} ---")

    try:
        # 1. Load settings
        settings = get_settings()
        model_name = settings.OLLAMA_MODEL_NAME
        ollama_url = "http://localhost:11434/v1"

        print(f"  Ollama URL: {ollama_url}")
        print(f"  Model: {model_name}")

        # 2. Initialize a NEW, simple agent for this test only
        simple_agent = Agent(
            model=OpenAIChatModel(
                model_name=model_name,
                provider=OllamaProvider(base_url=ollama_url),
            ),
            system_prompt="You are a helpful assistant that identifies objects in images.",
            output_type=str,
        )

        # 3. Define the prompt and image content using the Base64 data URI scheme
        image_bytes = image_path.read_bytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        media_type = f"image/{image_path.suffix.lstrip('.')}"
        data_uri = f"data:{media_type};base64,{base64_image}"

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What animal is in this image, and what is its primary color? Respond in a short phrase.",
                    },
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

        # 4. Run the simple agent
        result = simple_agent.run_sync(message_history=messages)

        # 5. Print the result
        print("\n--- Model Response ---")
        print(result.output)
        print("-" * 20)
        print(f"--- Test for {image_path.name} FINISHED ---")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    # Please place 'cat.jpg' and 'dog.jpg' in the 'illustrations' directory
    ILLUSTRATIONS_DIR = PROJECT_ROOT / "illustrations"
    cat_image_path = ILLUSTRATIONS_DIR / "cat.png"
    dog_image_path = ILLUSTRATIONS_DIR / "bird.png"

    print("Starting vision sanity check...")
    print("This test will check the model's basic ability to identify a cat and a dog.")

    # --- Run Tests ---
    run_vision_test(cat_image_path)
    run_vision_test(dog_image_path)
