# ml-workshop

Hands-on machine learning projects and implementations.

Each project is self-contained with its own `pyproject.toml`, `uv.lock` and virtual environment.
Run commands from inside a project directory (`cd <project> && uv sync && uv run pytest`) — the repo-root
environment only holds shared dev tooling. See [AGENTS.md](./AGENTS.md) for the full working guide and
[rules.md](./rules.md) for code conventions.

## Projects

### [LinkedIn Zip Puzzle Solver Challenge](./linkedin-zip-challenge/README.md)

A project exploring algorithms to solve the "LinkedIn Zip" puzzle game, featuring multiple solvers, procedural generation, and web UIs. Start with its [roadmap](./linkedin-zip-challenge/ai-collab/roadmap.md).

### [Board Game RL](./board-game-rl/README.md)

Reinforcement learning on board games (Tic-Tac-Toe): Q-Learning, Alpha-Beta and DQN agents behind a FastAPI + Gradio play interface. See its [project guide](./board-game-rl/ai-collab/project_guide.md).

### [Deep Learning Karpathy](./deep-learning-karpathy/README.md)

Tutorials reproducing Andrej Karpathy's material: GPT tokenizers (minBPE) and nanoGPT. See its [README](./deep-learning-karpathy/README.md) for details.

### [Lingua Tutor](./lingua-tutor/README.md)

An AI-powered language learning assistant for speech-to-text transcription and evaluation. See its [README](./lingua-tutor/README.md) for details.

### [More Simple Reinforcement Learning](./more_simple_reinforcement_learning/readme.md)

A collection of Jupyter notebooks implementing various reinforcement learning algorithms (e.g., Q-Learning, DQN, PPO, SAC). See its [README](./more_simple_reinforcement_learning/readme.md) for details.

### [Notes](./notes/README.md)

A collection of personal notes, references, and code snippets on machine learning topics. See its [README](./notes/README.md) for details.