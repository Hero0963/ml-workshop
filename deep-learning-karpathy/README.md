# Deep Learning Karpathy

Deep learning tutorials based on Andrej Karpathy's educational video series.

## Topics

| # | Topic | Source | Description |
|---|-------|--------|-------------|
| 1 | GPT Tokenizer | [minBPE](https://github.com/karpathy/minbpe) | Byte Pair Encoding (BPE) tokenization |
| 2 | nanoGPT | [nanoGPT](https://github.com/karpathy/nanoGPT) | Training GPT from scratch |

## Quick Start

```bash
# Install dependencies
cd deep-learning-karpathy/
uv sync

# Run tokenizer training
uv run python scripts/train_tokenizer.py

# Compare tokenizers (Character vs BPE vs GPT-4)
uv run python scripts/compare_tokenizers.py

# Train nanoGPT on Shakespeare
uv run python scripts/train_nanogpt.py

# Generate text from trained model
uv run python scripts/sample_text.py
```

## Project Structure

```
deep-learning-karpathy/
├── tutorials/              # Tutorial documents (Traditional Chinese)
│   ├── 01_tokenizer/       # Part 1: GPT Tokenizer (minBPE)
│   └── 02_nanogpt/         # Part 2: nanoGPT
├── scripts/                # Runnable Python scripts
├── notebooks/              # Jupyter notebooks (TBD)
├── references/             # Reference repos (git cloned)
│   ├── minbpe/
│   └── nanoGPT/
├── ai-collab/              # AI collaboration docs
└── pyproject.toml          # Python dependencies (managed by uv)
```

## Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- GPU recommended for nanoGPT training (CPU works but slower)
