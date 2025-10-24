# src/core/vl_models/vision_sanity_check.py

import sys
import base64
from pathlib import Path

try:
    import requests
except ImportError:
    print("=" * 50)
    print("ERROR: 'requests' library not found.")
    print("Please ask the user to install it: uv add requests")
    print("=" * 50)
    exit(1)

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ---

try:
    from src.settings import get_settings
except ImportError as e:
    print(f"Failed to import modules: {e}")
    exit(1)


def run_vision_test(image_path: Path):
    """Runs a simple vision test for the given image using the native Ollama API."""
    if not image_path.exists():
        print(f"ERROR: Image file not found at '{image_path}'")
        return

    print("\n" + "=" * 80)
    print(f"--- Running Sanity Check for: {image_path.name} (Native Ollama API) ---")

    try:
        # 1. Load settings
        settings = get_settings()
        model_name = settings.OLLAMA_MODEL_NAME
        # Use the native Ollama API endpoint
        ollama_url = "http://localhost:11434/api/generate"

        print(f"  Ollama URL: {ollama_url}")
        print(f"  Model: {model_name}")

        # 2. Prepare the request payload in the native Ollama format
        image_bytes = image_path.read_bytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        prompt_text = "What animal is in this image, and what is its primary color? Respond in a short phrase."

        payload = {
            "model": model_name,
            "prompt": prompt_text,
            "images": [base64_image],
            "stream": False,
        }

        # 3. Send the request using the 'requests' library
        print("Sending request to native Ollama API...")
        response = requests.post(ollama_url, json=payload)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # 4. Parse and print the response
        response_data = response.json()

        print("\n--- Model Response ---")
        # The actual response content is in the 'response' key
        print(response_data.get("response", "No 'response' key found in JSON output."))
        print("-" * 20)
        print(f"--- Test for {image_path.name} FINISHED ---")

    except requests.exceptions.RequestException as e:
        print(f"\nAn error occurred while communicating with the Ollama server: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


if __name__ == "__main__":
    ILLUSTRATIONS_DIR = PROJECT_ROOT / "illustrations"
    cat_image_path = ILLUSTRATIONS_DIR / "cat.png"
    bird_image_path = ILLUSTRATIONS_DIR / "bird.png"

    print("Starting vision sanity check using native Ollama API...")

    run_vision_test(cat_image_path)
    run_vision_test(bird_image_path)
