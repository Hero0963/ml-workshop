# src/core/vl_models/experiment_hf_extraction.py
import json
from pathlib import Path
import sys

# --- Add project root to sys.path ---
# This is necessary for the script to find the 'src' module when run directly
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
# ---

try:
    from src.core.vl_models.hf_parser import (
        load_model_and_processor,
        parse_puzzle_image_hf,
    )
except ImportError as e:
    print("=" * 50)
    print(f"Failed to import from hf_parser: {e}")
    print("Please ensure you are in the correct environment and all paths are correct.")
    print("=" * 50)
    exit(1)


def run_test(model_name: str, image_name: str):
    """
    Runs a single extraction test for a given model and image.

    Args:
        model_name: The name of the Hugging Face model to use.
        image_name: The filename of the image to test (must be in the 'illustrations' dir).
    """
    print("\n" + "=" * 80)
    print("--- Starting Test ---")
    print(f"  Model: {model_name}")
    print(f"  Image: {image_name}")
    print("=" * 80)

    # 1. Define paths
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.parent
    illustrations_dir = project_root / "illustrations"
    target_image_path = illustrations_dir / image_name

    print(f"Project Root: {project_root}")
    print(f"Illustrations Dir: {illustrations_dir}")
    print(f"Target Image: {target_image_path}")

    if not target_image_path.exists():
        print(f"\nERROR: Target image not found at {target_image_path}")
        return

    try:
        # 2. Load model and processor
        model, processor = load_model_and_processor(model_name)

        # 3. Run the parsing function
        solver_ready_data = parse_puzzle_image_hf(
            model=model,
            processor=processor,
            target_image_path=target_image_path,
        )

        # 4. Print the results
        if solver_ready_data:
            print("\n" + "-" * 30)
            print("  FINAL SOLVER-READY OUTPUT  ")
            print("-" * 30)

            def custom_serializer(obj):
                """Custom JSON serializer for printing sets and tuples."""
                if isinstance(obj, set):
                    # Sort the list for consistent output
                    return sorted(list(obj))
                if isinstance(obj, tuple):
                    return list(obj)
                return str(obj)

            print(json.dumps(solver_ready_data, indent=2, default=custom_serializer))
            print("-" * 80)
            print("--- Test PASSED ---")

        else:
            print("\n" + "!" * 80)
            print("--- Test FAILED: The parser returned None. ---")
            print("!" * 80)

    except Exception as e:
        print(f"\nAn unexpected error occurred during the test run: {e}")
        if "CUDA out of memory" in str(e):
            print(
                "ERROR: CUDA out of memory. Consider using a smaller model or a GPU with more VRAM."
            )
        print("!" * 80)
        print("--- Test FAILED ---")
        print("!" * 80)


if __name__ == "__main__":
    # --- Configuration ---
    # You can change these values to test different models or images
    HF_MODEL_NAME = "Qwen/Qwen3-VL-4B-Thinking"

    # Let's test a different image than the one in hf_parser.py to ensure robustness
    TEST_IMAGE_NAME = "puzzle_04.png"

    # --- Run Test ---
    run_test(model_name=HF_MODEL_NAME, image_name=TEST_IMAGE_NAME)
