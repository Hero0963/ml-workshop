# src/lm_course/artifacts.py
"""The shared pieces several labs build on: the course tokenizer, the tokenized TinyStories
split, and the base model pretrained in lab 06 (reused by labs 09, 11, 12 and 13).

Everything is created on first use and cached under ``checkpoints/`` and ``data/``.
"""

from pathlib import Path

import numpy as np
from loguru import logger

from lm_course.chat import SPECIAL_TOKENS
from lm_course.data import load_stories, split_stories
from lm_course.model import GPT, GPTConfig, load_model, nanochat_style_config
from lm_course.tokenizer import BPETokenizer
from lm_course.training import tokenize_stories
from lm_course.utils import CHECKPOINT_DIR, DATA_DIR

VOCAB_SIZE = 4096
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
BASE_MODEL_PATH = CHECKPOINT_DIR / "base_model.pt"


def course_stories() -> tuple[list[str], list[str]]:
    """TinyStories (GPT-4 validation file) split 90 / 10 by story, always the same way."""
    return split_stories(load_stories(), val_fraction=0.1, seed=0)


def course_tokenizer() -> BPETokenizer:
    """Byte-level BPE with 4096 tokens trained on the training stories (lab 03), including
    the chat special tokens of lesson 11 so every lab shares one vocabulary."""
    if TOKENIZER_PATH.exists():
        return BPETokenizer.load(TOKENIZER_PATH)
    train, _ = course_stories()
    logger.info("training the course tokenizer (about 10 s)")
    tokenizer = BPETokenizer.train(train, VOCAB_SIZE, special_tokens=SPECIAL_TOKENS)
    tokenizer.save(TOKENIZER_PATH)
    return tokenizer


def course_tokens(tokenizer: BPETokenizer) -> tuple[np.ndarray, np.ndarray]:
    """Token streams of the training and validation stories, cached as .npy files."""
    train, val = course_stories()
    cache = DATA_DIR / "tokens"
    return (
        tokenize_stories(tokenizer, train, cache / "train.npy"),
        tokenize_stories(tokenizer, val, cache / "val.npy"),
    )


def base_model_config(**overrides) -> GPTConfig:
    """The lab 06 model: nanochat-style, 4 layers, width 256, context 256 (about 5M params)."""
    base = dict(
        vocab_size=VOCAB_SIZE, context_length=256, n_layer=4, n_head=4, d_model=256
    )
    return nanochat_style_config(**(base | overrides))


def load_base_model(path: Path = BASE_MODEL_PATH) -> GPT:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found: run notebooks/06_pretraining.ipynb (or "
            "`uv run python scripts/pretrain.py`) first"
        )
    model, _ = load_model(path)
    return model.eval()
