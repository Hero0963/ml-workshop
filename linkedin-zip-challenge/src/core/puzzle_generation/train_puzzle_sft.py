# src/core/puzzle_generation/train_puzzle_sft.py

import json
from pathlib import Path
import random

import torch
from datasets import Dataset, DatasetDict
from PIL import Image

# --- Unsloth & Transformers Imports ---
try:
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
except ImportError:
    print(
        "Unsloth or transformers/trl not installed. Please ensure requirements are met."
    )
    exit(1)

# --- Constants ---
# You can change these parameters
DATASET_PATH = (
    "finetune_dataset_20251024_090929"  # The name of the generated dataset folder
)
MODEL_NAME = "unsloth/Qwen3-VL-4B-Thinking-bnb-4bit"  # Using Unsloth's 4bit version for efficiency

# --- 1. Dataset Loading and Preparation ---


def load_puzzle_dataset(
    dataset_dir: Path, test_split_ratio: float = 0.1
) -> DatasetDict:
    """Loads images and JSON labels, formats them, and splits into train/test."""
    print(f"Loading dataset from {dataset_dir}...")
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"

    items = []
    label_files = sorted(list(labels_dir.glob("*.json")))

    for label_file in label_files:
        image_file = images_dir / f"{label_file.stem}.png"
        if image_file.exists():
            with open(label_file, "r", encoding="utf-8") as f:
                label_json = json.load(f)
            items.append(
                {
                    "image": str(image_file),
                    "label": json.dumps(
                        label_json, indent=2
                    ),  # Store the ground truth as a JSON string
                }
            )

    print(f"Found {len(items)} matching image-label pairs.")
    random.shuffle(items)  # Shuffle before splitting

    # Create Hugging Face Dataset
    full_dataset = Dataset.from_list(items)

    # Split into training and testing sets
    split_dict = full_dataset.train_test_split(test_size=test_split_ratio, seed=42)

    return split_dict  # Returns a DatasetDict with 'train' and 'test' keys


# --- 2. Model and Tokenizer Setup ---


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
        r=16,  # LoRA rank
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


# --- 3. Main Training Function ---


def main():
    """Main function to orchestrate the fine-tuning process."""
    # Find the dataset directory in the parent project
    project_root = Path(__file__).parent.parent.parent
    dataset_dir = project_root / "linkedin-zip-challenge" / DATASET_PATH
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found at {dataset_dir}")
        return

    # Load and prepare dataset
    dataset_dict = load_puzzle_dataset(dataset_dir)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]  # Use the test split for evaluation

    # Setup model
    model, tokenizer = setup_model_and_tokenizer()

    # Define a formatting function to create the prompt string
    def format_prompt(example):
        image = Image.open(example["image"])
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": "Analyze the puzzle in this image and provide the JSON data.",
                        },
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = tokenizer.apply_chat_template(
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": example["label"]}],
                }
            ],
            tokenize=False,
        )
        # SFTTrainer expects a single `text` field
        return {"image": image, "text": prompt + response}

    train_dataset = train_dataset.map(format_prompt, batched=False)
    eval_dataset = eval_dataset.map(format_prompt, batched=False)

    # Configure Training
    training_args = TrainingArguments(
        output_dir="./outputs/sft_qwen3_vl_puzzle",
        num_train_epochs=3,  # A few epochs are usually enough for fine-tuning
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        # --- Evaluation and Saving Strategy ---
        evaluation_strategy="steps",
        eval_steps=20,  # Evaluate every 20 steps
        save_strategy="steps",
        save_steps=20,  # Save checkpoint every 20 steps
        load_best_model_at_end=True,  # Load the best model at the end of training
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,  # Only keep the best and the latest checkpoint
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",  # We created this field in format_prompt
        max_seq_length=2048,
        dataset_num_proc=2,
        packing=False,  # Can be True for text-only but must be False for vision
        args=training_args,
    )

    # Train the model
    print("--- Starting SFT Training ---")
    trainer.train()

    # Save the final best model
    print("--- Training finished. Saving best model... ---")
    trainer.save_model("./outputs/sft_qwen3_vl_puzzle/best_model")
    print("Best model saved successfully.")


if __name__ == "__main__":
    main()
