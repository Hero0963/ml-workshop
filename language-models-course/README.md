# Language Models Course

A hands-on course on language models, from word2vec to a small chat model that uses a tool and
is improved with reinforcement learning. It integrates the core of five sources — **word2vec**,
**GPT-2**, **Stanford CS336** (Language Modeling from Scratch), **nanochat** and **text
embedding models** — into one sequence ordered the way a model is built, with the overlaps
taught once. Every idea is implemented from scratch in PyTorch, tested, and run in a lab
notebook on a laptop CPU.

The lessons are written in Traditional Chinese (each one explains the intuition in plain
language first, then gives the formal derivation); the code and its comments are in English.
Start with [`lessons/00_course_map.md`](lessons/00_course_map.md).

## Syllabus

| # | Lesson | Lab notebook |
|---|--------|--------------|
| 00 | [Course map and setup](lessons/00_course_map.md) | — |
| 01 | [Language models: probability, cross-entropy, bits per byte, n-grams](lessons/01_language_models_and_ngrams.md) | [`01_ngram.ipynb`](notebooks/01_ngram.ipynb) |
| 02 | [Word vectors: word2vec and the distributional hypothesis](lessons/02_word2vec.md) | [`02_word2vec.ipynb`](notebooks/02_word2vec.ipynb) |
| 03 | [Tokenization: byte-level BPE](lessons/03_tokenization_bpe.md) | [`03_bpe.ipynb`](notebooks/03_bpe.ipynb) |
| 04 | [Attention and the Transformer block](lessons/04_attention_and_transformer.md) | [`04_attention.ipynb`](notebooks/04_attention.ipynb) |
| 05 | [GPT-2: architecture, initialization, decoding, loading the real weights](lessons/05_gpt2.md) | [`05_gpt2.ipynb`](notebooks/05_gpt2.ipynb) |
| 06 | [Pretraining a small GPT: the training loop, AdamW, Muon, modern architectures](lessons/06_pretraining.md) | [`06_pretraining.ipynb`](notebooks/06_pretraining.ipynb) |
| 07 | [Compute, memory and scaling laws](lessons/07_scaling_laws.md) | [`07_scaling.ipynb`](notebooks/07_scaling.ipynb) |
| 08 | [Systems: arithmetic intensity, FlashAttention, mixed precision, parallelism](lessons/08_systems.md) | [`08_systems.ipynb`](notebooks/08_systems.ipynb) |
| 09 | [Inference: KV cache, batching, speculative decoding, quantization](lessons/09_inference.md) | [`09_inference.ipynb`](notebooks/09_inference.ipynb) |
| 10 | [Data: filtering, deduplication, mixing, contamination](lessons/10_data.md) | [`10_data.ipynb`](notebooks/10_data.ipynb) |
| 11 | [From base model to chat: SFT, step-by-step answers, tool use](lessons/11_sft_and_chat.md) | [`11_sft.ipynb`](notebooks/11_sft.ipynb) |
| 12 | [Preferences and RL: RLHF, DPO, GRPO](lessons/12_rl_and_preferences.md) | [`12_rl.ipynb`](notebooks/12_rl.ipynb) |
| 13 | [Embedding models: from word vectors to sentence vectors](lessons/13_embedding_models.md) | [`13_embeddings.ipynb`](notebooks/13_embeddings.ipynb) |
| 14 | [Evaluation, and a walk through nanochat](lessons/14_evaluation_and_nanochat.md) | — |
| 15 | [The 2026 frontier: a map for further reading](lessons/15_frontier_2026.md) | — |
| A | [Notation and formula cheat sheet](lessons/A_notation.md) | — |

Curated reading list with verification dates: [`references.md`](references.md).

## Setup

The project is self-contained, like every project in this repository. Dependencies are added
by hand (repository rule), and `pyproject.toml` already routes `torch` to the PyTorch CUDA 12.6
wheel index, so the same commands give a GPU build on Windows and Linux:

```bash
cd language-models-course
uv add torch numpy matplotlib loguru regex ipykernel
uv add --dev pytest
uv run pytest
```

Then open any notebook in `notebooks/` with the `.venv` of this project as the kernel.
Downloads go to `data/` on first use (ignored by git): TinyStories (22.5 MB) for every lab, and
GPT-2 124M (vocabulary 1.5 MB, weights 548 MB) for labs 03 and 05. Tests that need the network
are marked `network` and skip themselves when offline.

**Order of the labs.** Lab 06 pretrains the base model (`checkpoints/base_model.pt`) that labs
09, 11 and 13 load, and lab 11 saves the chat model that lab 12 loads. Lab 03 saves the course
tokenizer; later labs train an identical one if it is missing.

**Hardware.** Every lab runs on a CPU; the executed outputs come from a 4-core cloud CPU. Most
labs take a few minutes; the longest are lab 07 (the IsoFLOP sweep, about 80 minutes; its first
two budgets alone take about 25), lab 06 (about 40 minutes, 27 of them pretraining), lab 13
(about 23 minutes) and lab 11 (about 18 minutes). Do not run two training notebooks at the same
time: they compete for the same cores. With a GPU, raise the step counts at the top of each
notebook, or pretrain for longer from the command line:

```bash
uv run python scripts/pretrain.py --steps 3000
```

## Project structure

```
language-models-course/
├── lessons/                 # the course text (Traditional Chinese)
├── notebooks/               # executed lab notebooks, one per lesson with a lab
├── src/lm_course/           # reference implementation used by every lab
│   ├── ngram.py             #   count-based byte n-gram models, bits per byte
│   ├── word2vec.py          #   skip-gram with negative sampling, PPMI + SVD, analogies
│   ├── tokenizer.py         #   byte-level BPE (training, encoding), GPT-2's vocabulary
│   ├── model.py             #   one configurable GPT: GPT-2 / Llama-style / nanochat-style, KV cache,
│   │                        #   loading OpenAI's GPT-2 weights (safetensors reader included)
│   ├── sampling.py          #   temperature, top-k, top-p, cached generation, speculative decoding
│   ├── optim.py             #   AdamW groups, Muon, cosine and WSD schedules
│   ├── training.py          #   pretraining loop, validation loss and bits per byte
│   ├── scaling.py           #   FLOP and memory accounting, scaling-law fits
│   ├── kernels.py           #   online softmax, tiled (FlashAttention-style) attention, roofline
│   ├── quantization.py      #   absmax int8 / int4 weight quantization
│   ├── data_pipeline.py     #   Gopher rules, MinHash + LSH dedup, quality classifier, contamination
│   ├── chat.py              #   chat format, SFT masks, addition task, topic reward, calculator tool, chat engine
│   ├── rl.py                #   policy-gradient / GRPO loss, KL penalty, DPO, pass@k
│   ├── embeddings.py        #   pooling, InfoNCE, Matryoshka, retrieval metrics, BM25
│   ├── artifacts.py         #   shared tokenizer, token streams and base model
│   └── data.py              #   downloads and generated corpora
├── scripts/pretrain.py      # the lab 06 pretraining run from the command line
├── tests/                   # checks against closed-form answers and reference outputs (pytest)
├── references.md            # reading list with verification dates
├── NOTICE.md                # third-party licences and attribution
└── ai-collab/               # roadmap, dev log, project guide
```

## Licensing and attribution

All code and text in this project were written for it; no code was copied from nanochat,
nanoGPT, CS336 or any paper's reference implementation. CS336's lecture materials are linked,
not reproduced; students enrolled in CS336 should follow its honor code. Model weights (GPT-2)
and data (TinyStories, CDLA-Sharing-1.0) are downloaded at run time and not redistributed; the
executed notebooks show some TinyStories excerpts and GPT-2-generated text, attributed in
[`NOTICE.md`](NOTICE.md).
