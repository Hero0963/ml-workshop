# scripts/sample_text.py
"""
Generate text from a trained nanoGPT model.

Usage:
    uv run python scripts/sample_text.py
    uv run python scripts/sample_text.py --prompt "ROMEO:" --temperature 0.8
    uv run python scripts/sample_text.py --num_tokens 1000 --top_k 50
"""

import os
import sys
import argparse

import torch

# Reuse model definition from train_nanogpt — must happen before local import
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from train_nanogpt import GPT  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained nanoGPT")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint"
    )
    parser.add_argument("--prompt", type=str, default="\n", help="Starting prompt")
    parser.add_argument(
        "--num_tokens", type=int, default=500, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples")
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args()

    # Device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Find checkpoint
    project_dir = os.path.dirname(script_dir)
    ckpt_path = args.checkpoint or os.path.join(
        project_dir, "out", "shakespeare_char.pt"
    )

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Please train the model first:")
        print("  uv run python scripts/train_nanogpt.py")
        sys.exit(1)

    # Load checkpoint
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    chars = checkpoint["chars"]

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s):
        return [stoi[c] for c in s]

    def decode(ids):
        return "".join([itos[i] for i in ids])

    # Load model
    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Model loaded: {model.get_num_params() / 1e6:.2f}M parameters")

    # Generate
    print(
        f"\nSettings: temperature={args.temperature}, top_k={args.top_k}, tokens={args.num_tokens}"
    )
    print("=" * 60)

    for i in range(args.num_samples):
        prompt_ids = encode(args.prompt)
        context = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        with torch.no_grad():
            generated = model.generate(
                context,
                max_new_tokens=args.num_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )

        text = decode(generated[0].tolist())

        print(f"\n--- Sample {i + 1} ---")
        print(text)
        print()

    print("=" * 60)
    print("Tips:")
    print("  --temperature 0.5  → more conservative, repetitive")
    print("  --temperature 1.0  → balanced creativity")
    print("  --temperature 1.5  → wild and creative")
    print("  --prompt 'ROMEO:'  → start with specific character")


if __name__ == "__main__":
    main()
