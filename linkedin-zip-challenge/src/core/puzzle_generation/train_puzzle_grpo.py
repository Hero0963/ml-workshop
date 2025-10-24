# src/core/puzzle_generation/train_puzzle_grpo.py

import json
from pathlib import Path

from datasets import Dataset
from PIL import Image
from pydantic import BaseModel, ValidationError

# --- Unsloth & Transformers Imports ---
try:
    from unsloth import FastLanguageModel
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    print("Unsloth or trl not installed. Please ensure requirements are met.")
    exit(1)

# --- Constants ---
DATASET_PATH = (
    "finetune_dataset_20251024_090929"  # The name of the generated dataset folder
)
MODEL_NAME = "unsloth/Qwen3-VL-4B-Thinking-bnb-4bit"


# --- 1. Pydantic Models for Reward Function ---
# This helps validate the structure of the generated JSON
class WallPair(BaseModel):
    cell1: list[int]
    cell2: list[int]


class SimplePuzzleOutput(BaseModel):
    layout: list[list[str]]
    walls: list[WallPair]


# --- 2. Dataset Loading and Preparation ---
def load_puzzle_dataset_for_rl(dataset_dir: Path) -> Dataset:
    """Loads images and creates prompts for RL training."""
    print(f"Loading dataset from {dataset_dir} for RL...")
    images_dir = dataset_dir / "images"
    image_files = sorted(list(images_dir.glob("*.png")))

    items = [{"image": str(img_file)} for img_file in image_files]
    print(f"Found {len(items)} images.")
    return Dataset.from_list(items)


# --- 3. Model and Tokenizer Setup ---
def setup_model_and_tokenizer():
    """Loads the Qwen3-VL model and tokenizer using Unsloth."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=2048,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    return model, tokenizer


# --- 4. Reward Function ---
def extract_json_block(text: str) -> str | None:
    """Extracts a JSON block from a markdown string."""
    match = __import__("re").search(
        r"```json\s*(\{.*?\})\s*```", text, __import__("re").DOTALL
    )
    if match:
        return match.group(1)
    start_index = text.find("{")
    end_index = text.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index : end_index + 1]
    return None


def reward_function(completions: list[str], **kwargs) -> list[float]:
    """Scores the model's generated text based on JSON validity and structure."""
    scores = []
    for text in completions:
        score = 0.0
        json_str = extract_json_block(text)
        if not json_str:
            score = -5.0  # Heavy penalty for not producing any JSON-like block
        else:
            try:
                parsed_json = json.loads(json_str)
                score = 2.0  # Reward for being valid JSON
                try:
                    SimplePuzzleOutput(**parsed_json)
                    score = 5.0  # Higher reward for matching the Pydantic schema
                except ValidationError:
                    score = 1.0  # Lower reward if it's JSON but not the right structure
            except json.JSONDecodeError:
                score = (
                    -2.0
                )  # Penalty for producing something that looks like JSON but isn't
        scores.append(score)
    return scores


# --- 5. Main Training Function ---
def main():
    """Main function to orchestrate the GRPO fine-tuning process."""
    project_root = Path(__file__).parent.parent.parent
    dataset_dir = project_root / "linkedin-zip-challenge" / DATASET_PATH
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found at {dataset_dir}")
        return

    dataset = load_puzzle_dataset_for_rl(dataset_dir)
    model, tokenizer = setup_model_and_tokenizer()

    def format_prompt(example):
        image = Image.open(example["image"])
        prompt_text = "Analyze the puzzle in this image and provide the JSON data, including 'layout' and 'walls'."
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        return {"image": image, "prompt": prompt}

    dataset = dataset.map(format_prompt, batched=False)

    training_args = GRPOConfig(
        output_dir="./outputs/grpo_qwen3_vl_puzzle",
        num_train_epochs=1,  # RL training can be shorter
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        learning_rate=5e-6,
        logging_steps=1,
        log_completions=True,  # Log generated completions to see what model is doing
        num_generations=2,  # How many responses to generate for each prompt
        max_prompt_length=1024,
        max_completion_length=1024,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        optim="adamw_8bit",
        seed=42,
        # GSPO settings from notebook
        importance_sampling_level="sequence",
        loss_type="dr_grpo",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_function=reward_function,
        data_collator=None,  # Not needed for GRPOTrainer with this data format
    )

    print("--- Starting GRPO Training ---")
    trainer.train()
    print("--- Training finished. Saving final model... ---")
    trainer.save_model("./outputs/grpo_qwen3_vl_puzzle/final_model")
    print("Final model saved successfully.")


if __name__ == "__main__":
    main()
