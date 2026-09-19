# src/core/vl_models/experiment_pydantic_ai_extraction.py
import json
import sys
from pathlib import Path

# --- Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ---

try:
    from src.core.vl_models.vl_extractor import VLExtractor, PuzzleExtractionOutput
    from src.core.vl_models.hf_parser import (
        post_process_data,
    )  # Re-using the post-processor
    from src.settings import get_settings  # Import settings loader
except ImportError as e:
    print("=" * 50)
    print(f"Failed to import modules: {e}")
    print("Please ensure you are in the correct environment and all paths are correct.")
    print("=" * 50)
    exit(1)


def run_pydantic_ai_test(ollama_url: str, model_name: str, image_name: str):
    """
    Runs a single extraction test using the pydantic_ai and Ollama pipeline.
    """
    print("\n" + "=" * 80)
    print("--- Starting pydantic_ai Test ---")
    print(f"  Ollama URL: {ollama_url}")
    print(f"  Model: {model_name}")
    print(f"  Image: {image_name}")
    print("=" * 80)

    # 1. Define paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent
    illustrations_dir = project_root / "illustrations"
    target_image_path = illustrations_dir / image_name

    print(f"Target Image: {target_image_path}")

    if not target_image_path.exists():
        print(f"\nERROR: Target image not found at {target_image_path}")
        return

    try:
        # 2. Initialize the VLExtractor
        print("\nInitializing VLExtractor...")
        extractor = VLExtractor(
            ollama_provider_url=ollama_url, ollama_model_name=model_name
        )

        # 3. Run the extraction function
        print("\nCalling extractor.extract_from_image()... (This may take a moment)")
        # The type hint here is important for clarity
        mllm_output: PuzzleExtractionOutput = extractor.extract_from_image(
            target_image_path
        )

        # 4. Print the thinking process from the model
        print("\n" + "-" * 30)
        print("  MLLM Thinking Process  ")
        print("-" * 30)
        print(mllm_output.thinking_process)
        print("-" * 30)

        # 5. Post-process the data to get the final solver-ready dictionary
        # Note: post_process_data expects an object with .layout and .walls attributes,
        # which PuzzleExtractionOutput has.
        solver_ready_data = post_process_data(mllm_output)

        # 6. Print the final result
        if solver_ready_data:
            print("\n" + "-" * 30)
            print("  FINAL SOLVER-READY OUTPUT  ")
            print("-" * 30)

            def custom_serializer(obj):
                """Custom JSON serializer for printing sets and tuples."""
                if isinstance(obj, set):
                    return sorted(list(obj))
                if isinstance(obj, tuple):
                    return list(obj)
                return str(obj)

            print(json.dumps(solver_ready_data, indent=2, default=custom_serializer))
            print("-" * 80)
            print("--- Test PASSED ---")

        else:
            print("\n" + "!" * 80)
            print("--- Test FAILED: Post-processing returned None. ---")
            print("!" * 80)

    except Exception as e:
        print(f"\nAn unexpected error occurred during the test run: {e}")
        print("!" * 80)
        print("--- Test FAILED ---")
        print("!" * 80)


if __name__ == "__main__":
    # --- Configuration ---
    # For this test script run from the host, we explicitly use localhost for the URL.
    # The main application running inside Docker will use the `.env` setting (`ollama_server`).
    OLLAMA_HOST_URL = "http://localhost:11434/v1"

    # We still respect the model name from the .env file.
    try:
        settings = get_settings()
        OLLAMA_MODEL_NAME = settings.OLLAMA_MODEL_NAME
    except Exception as e:
        print(f"Could not load settings from .env file: {e}")
        print("Please ensure a .env file exists and is configured correctly.")
        exit(1)

    TEST_IMAGE_NAME = "puzzle_04.png"

    print(
        "IMPORTANT: This test requires an Ollama server to be running and accessible from the host."
    )
    print(
        f"Attempting to connect to: '{OLLAMA_HOST_URL}' with model '{OLLAMA_MODEL_NAME}'"
    )
    print("You can run the server via: `docker compose up ollama`")

    # --- Run Test ---
    run_pydantic_ai_test(
        ollama_url=OLLAMA_HOST_URL,
        model_name=OLLAMA_MODEL_NAME,
        image_name=TEST_IMAGE_NAME,
    )
