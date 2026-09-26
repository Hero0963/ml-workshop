# ml-workshop

Hands-on machine learning projects and implementations.

Each project is self-contained with its own `pyproject.toml`, `uv.lock` and virtual environment.
Run commands from inside a project directory (`cd <project> && uv sync && uv run pytest`) — the repo-root
environment only holds shared dev tooling. See [AGENTS.md](./AGENTS.md) for the full working guide and
[rules.md](./rules.md) for code conventions.

## Projects

### [Thread the Grid](./thread-the-grid/README.md)

Grid path puzzles — one line through every cell, visiting the numbered waypoints in order — generated, read from screenshots by a fine-tuned vision model, and solved by nine solvers behind a Gradio UI and a Svelte editor. Inspired by the Zip puzzle games found online. Start with its [roadmap](./thread-the-grid/ai-collab/roadmap.md).

### [Board Game RL](./board-game-rl/README.md)

Reinforcement learning on board games (Tic-Tac-Toe): Q-Learning, Alpha-Beta and DQN agents behind a FastAPI + Gradio play interface. See its [project guide](./board-game-rl/ai-collab/project_guide.md).

### [Diffusion Models Course](./diffusion-models-course/README.md)

A self-study course on diffusion models, from DDPM derivations through score-based SDEs, DDIM, classifier-free guidance and flow matching to diffusion transformers and a 2026 research map. Twelve lessons (Traditional Chinese), executed lab notebooks that run on a CPU, and a tested reference implementation. Start with the [course map](./diffusion-models-course/lessons/00_course_map.md).

### [Language Models Course](./language-models-course/README.md)

A self-study course on language models that integrates word2vec, GPT-2, Stanford CS336, nanochat and text embedding models: n-gram baselines, word2vec, byte-level BPE, a configurable GPT that loads OpenAI's GPT-2 weights, pretraining with Muon, scaling laws, systems, inference, data pipelines, SFT with tool use, GRPO and DPO, and contrastive embedding models, plus a 2026 frontier map. Sixteen lessons (Traditional Chinese), executed lab notebooks that run on a CPU, and a tested reference implementation. Start with the [course map](./language-models-course/lessons/00_course_map.md).

### [Deep Learning Karpathy](./deep-learning-karpathy/README.md)

Tutorials reproducing Andrej Karpathy's material: GPT tokenizers (minBPE) and nanoGPT. See its [README](./deep-learning-karpathy/README.md) for details.

### [Lingua Tutor](./lingua-tutor/README.md)

An AI-powered language learning assistant for speech-to-text transcription and evaluation. See its [README](./lingua-tutor/README.md) for details.

### [More Simple Reinforcement Learning](./more_simple_reinforcement_learning/readme.md)

A collection of Jupyter notebooks implementing various reinforcement learning algorithms (e.g., Q-Learning, DQN, PPO, SAC). See its [README](./more_simple_reinforcement_learning/readme.md) for details.

### [Notes](./notes/README.md)

A collection of personal notes, references, and code snippets on machine learning topics. See its [README](./notes/README.md) for details.