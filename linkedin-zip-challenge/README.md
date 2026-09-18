# LinkedIn Zip Puzzle Solver Challenge

> **Status: wrapped up on 2026-09-19.** Development has stopped; what works, what was
> verified, and what is left undone are in the
> [final report](./ai-collab/reports/2026-09-19_project-wrap-up.md) (Traditional Chinese).

The LinkedIn "Zip" puzzle, taken end to end: **make** a board, **read** one out of a
screenshot, and **solve** it. One FastAPI application serves all three, with a Gradio console
and a Svelte canvas editor on top, and the whole stack starts with a single command.

Two of those three pieces are learned. A **fine-tuned vision-language model** turns a
screenshot into a board — that is the reading step, not a solver. Solving itself is done two
ways, and comparing them is the point of the project: **classical search** (an exact
constraint solver among others) against a **neural policy trained by imitation** that plays
the board move by move.

The exact solver already wins on speed and correctness, so what the learned solver is really
for is to find out *where learning helps, where it does not, and how you would know*. The
measurements that answer that — including the ones that refuted our own hypotheses — are in
[`ai-collab/`](./ai-collab/).

---

## About the "Zip" puzzle

Draw a single continuous path that visits every open cell exactly once.

*   The path covers every visitable cell, and no cell is visited twice.
*   It must pass through the numbered waypoints in ascending order (1 → 2 → 3 …) and end on the
    highest one.
*   It cannot cross a wall (drawn as `|` or `—`).

Formally this is a **Hamiltonian path with ordering constraints** — NP-hard in general, which
is exactly why comparing an exact solver against learned ones is interesting on a board this
small.

![Solution animation](./solution.gif)

---

## Quick start

You need **Docker** (Docker Desktop, or Docker Engine with Compose V2) and any
**Python 3.9 or newer** on the host — the launcher uses only the standard library. The app image
is about 6 GB.

```bash
git clone https://github.com/Hero0963/ml-workshop.git
cd ml-workshop/linkedin-zip-challenge
python start.py
```

`start.py` creates `.env` from the template, starts the machine's one Ollama container, builds
and starts this checkout's app, **waits until the API actually answers**, and then says what
works and what does not. Open:

| | |
|---|---|
| Gradio console | <http://127.0.0.1:7440/ui> |
| Svelte editor | <http://127.0.0.1:7440/svelte-ui/> |
| API docs | <http://127.0.0.1:7440/docs> |

| Command | What it does |
|---|---|
| `python start.py` | Production stack: code baked into the image |
| `python start.py --dev` | Development stack: hot reload plus the Svelte dev server on `:5173` |
| `python start.py --status` | What is running, and which optional pieces are missing |
| `python start.py --down` | Stop and remove the containers |

**The first build takes a couple of minutes** — 88 seconds from a clean cache on the author's
machine (2026-09-19), most of it downloading Python wheels. After that, start-up to a healthy
API is about ten seconds.

**What works on a fresh clone.** Generating puzzles, editing them, and eight of the nine
solvers work straight away. Two things need model weights that are **not in this repository**
(see [Model weights](#model-weights)), and both degrade honestly without them: the learned
solver answers `503` with the path of the missing checkpoint, and screenshot reading answers
`503` naming the missing model. `start.py` tells you about both before you find out the hard
way. Screenshot reading additionally needs an **NVIDIA GPU**; without one, `start.py` says so
and starts everything else.

An operator's guide — what each container is for, acceptance checks, how to diagnose a
failure — is in [`ai-collab/deployment-guide.md`](./ai-collab/deployment-guide.md)
(Traditional Chinese).

---

## Model weights

Neither model is in version control: the vision model is 9.1 GB, and the RL policy, although
only 14 MB, cannot be regenerated bit for bit (its training data is procedurally generated with
a wall-clock timeout, so a rerun produces a different set of puzzles).

| Model | Size | Status (2026-09-19) | Without it |
|---|---|---|---|
| RL policy `bc_multi_456_e6` | 14 MB | **Not yet published.** Planned as a GitHub release asset of this repository. | `RL (behaviour cloning)` answers 503; every other solver works |
| Vision model `zip-qwen35-4b-p4c:f16` | 9.1 GB (two GGUF files) | **Not yet published.** Planned as a Hugging Face model repository. | `/api/vision/solve` answers 503 |

Once published, installing them is a download each — the RL file goes to
`models/rl_a2/bc_multi_456_e6/checkpoints/model_final.zip` (no restart needed), the two GGUF
files are imported into the Ollama container with a two-line Modelfile. The exact commands, the
checksums, and why the vision model must **not** be pulled through `ollama run hf.co/...` are in
[`ai-collab/model-weights.md`](./ai-collab/model-weights.md).

Fallbacks until then: train the RL policy yourself (generate a dataset, then about five minutes
of GPU training — see `ai-collab/model-weights.md` §7; it will be a different model with close but
not identical numbers), or point `.env` at the un-finetuned `qwen3.5:4b-q8_0` with
`VISION_PROMPT_VARIANT=sized` — it reads numbers and layout well but misses walls (wall F1 about
0.44).

---

## What's in the box

### 1. Make a puzzle

Procedural generation that draws a Hamiltonian path first and then carves the puzzle out of
it, so every generated board is solvable by construction (though not necessarily uniquely:
87.5% of the generated 6×6 training boards have more than one solution). Available from the
Gradio console, the Svelte editor, and as a deterministic dataset generator
(`src/core/rl/generate_dataset_v2.py`) that keeps the solution path with each puzzle — which
is what made imitation learning possible.

### 2. Read a puzzle from a screenshot

`POST /api/vision/solve` takes an image, a fine-tuned vision-language model turns it into a
board, and a solver solves it. The response carries **warnings** and a **`solvable`** flag
rather than only an answer, because a misreading is the failure mode that matters and those
two fields are how you notice it. See [The two models](#the-two-models).

### 3. Solve it

`POST /api/solver/solve` takes a board and a solver name; `GET /api/solver/list` says which
names are accepted. Nine of the ten solvers below are served; Particle Swarm Optimization is
kept in the code but not offered.

---

## The solvers

Every solver lives in `src/core/solvers/` and shares one puzzle representation
(`src/core/utils.py`). The list served by the API, the screenshot endpoint, the Gradio
dropdown and the Svelte editor comes from a single registry (`src/core/solvers/registry.py`);
the editor reads it from `GET /api/solver/list`.

| Solver | Kind | Served today | Notes |
|---|---|---|---|
| **CP-SAT** | exact | yes | Constraint solver. Fastest and always right — the default. |
| **DFS** | exact | yes | Depth-first with pruning. |
| **A\*** | exact | yes | Best-first over the same space. |
| **RL (behaviour cloning)** | learned | yes, **with weights** | A neural policy for 4×4 to 6×6. Not exact, not guaranteed — see below. |
| Ant Colony Optimization | metaheuristic | yes | Not exact, not guaranteed — see below. |
| Genetic Algorithm | metaheuristic | yes | " |
| Simulated Annealing | metaheuristic | yes | " |
| Tabu Search | metaheuristic | yes | " |
| Monte Carlo | metaheuristic | yes | " The baseline: independent random walks. |
| Particle Swarm Optimization | metaheuristic | **no** | Implemented and measured, not served: its swap moves break walks apart, so it almost never solves a 6×6 board (`ai-collab/reports/2026-09-19_pso-not-served.md`). |

The five served metaheuristics are there for comparison, not for speed — on a board this small CP-SAT
beats them on every axis. Each one returns the best path it saw whether or not that path solves
anything, so the registry reruns it until an answer passes `src/core/solvers/verify.py`, for a
fixed 5 seconds per request, and otherwise answers "could not find a solution". For a
heuristic that means it gave up, not that the board has none. The budget is deliberately not an
API parameter: they count their effort in different units, and a fixed budget is what makes
them comparable (`ai-collab/reports/2026-09-12_heuristic-solvers-on-the-api.md`).

---

## The two models

### Vision: reading a board out of a screenshot

| | |
|---|---|
| **Model chosen** | **Qwen3.5-4B** (Apache-2.0), fine-tuned with LoRA and served by Ollama as `zip-qwen35-4b-p4c:f16` |
| **Why this one** | It had an official fine-tuning notebook and a realistic chance of fitting; 4B loads fully onto a 16 GB card with no CPU offload. Q4 of the same model **could not emit valid JSON at all** — quantisation mattered more than size. |
| **How it was trained** | Synthetic screenshots rendered from generated puzzles, LoRA on a Colab L4: **975 steps, 1.56 h**, peak VRAM 20.9 of 22.0 GiB. Merged into the base model, converted to GGUF, imported into Ollama. |
| **What it fixed** | The un-finetuned bottleneck was **walls, and only walls**: layout 0.947 and numbers 0.917 on real screenshots, but **wall F1 0.438** and 2/6 end to end. After fine-tuning, the synthetic held-out set is **200/200** end to end. |
| **Checked again at wrap-up** | Six boards rendered fresh on 2026-09-19 (2 to 12 walls, light and dark themes) plus four held-out ones: **10/10** — layout, walls, and a path that solves the *labelled* board. |
| **Export cost** | Zero: the same 200 held-out images produce **byte-identical** output locally and on Colab, and **6.5× faster** (34.5 s → 5.3 s per image). |

⚠ The goal was to read **boards this project renders**, and that is what the numbers show. They
do not show that it reads any LinkedIn screenshot (six real ones: 5/6 end to end — too few to
claim). The synthetic held-out set is saturated at 1.000, so it can no longer tell two
approaches apart; making the evaluation harder would be the next step, not a bigger model.

⚠ The model tag and the prompt variant **must match** (`VISION_PROMPT_VARIANT=finetune` with
the fine-tuned tag). Pairing the fine-tune prompt with an un-finetuned model asks it a question
it never saw, and the failure is quiet: a `200` response containing an empty board.

### Reinforcement learning: playing the puzzle move by move

The environment is a one-stroke walk: the observation is eight 8×8 feature planes plus a short
scalar vector, the action is one of four directions, and illegal moves are removed by an action
mask before the policy ever sees them. Reward is frozen-lake style — `+1` for solving, nothing
otherwise.

| | |
|---|---|
| **Model** | Three padded 3×3 convolutions (64 channels, no pooling) → 256-d features → policy and value heads. **1.17 M parameters**, of which 89.7% are the flattening layer. |
| **How it is trained today** | **Behaviour cloning.** Every generated puzzle ships with its solution, so ~1.17 M `(board, next move)` pairs are a supervised dataset. Masked cross-entropy, **6 epochs** (training longer makes the policy sharper and loses diversity), batch 512 — **about 5.5 minutes on one GPU**. |
| **What it replaced** | MaskablePPO with a reverse curriculum: 8 M steps and ~2,000 s, and it **lost** to minutes of supervised training on every board and every inference setting. Fine-tuning the cloned policy with PPO made single attempts better and best-of-32 worse. |
| **One model, three boards** | **A single policy serves 4×4, 5×5 and 6×6.** Against size-specific controls trained on the same data it is better on a single deterministic run (+0.032 to +0.037 on every board), level under a best-of-32 budget, and cheaper at inference. |

**Results** of the served policy (held-out test, 1,931 / 2,001 / 2,000 puzzles). *Best-of-32*
means: sample up to 32 full attempts and keep the first that verifies — legitimate here because
a Zip solution checks itself. Attempts stop at the first success, so the average cost is far
below 32.

| Board | Greedy baseline | Policy, single deterministic run | Policy, best-of-32 | Attempts used |
|---|---|---|---|---|
| 4×4 | 0.1156 | 0.9410 | **0.9953** | 1.41 |
| 5×5 | 0.0346 | 0.7701 | not measured | — |
| 6×6 | 0.0046 | 0.5430 | **0.9465** | 5.24 |

The goals (best-of-32 ≥ 0.90 on 4×4 and ≥ 0.85 on 6×6) are met. Since 2026-09-19 every judge
also requires the path to end on the highest number; that stricter rule moves the 6×6
best-of-32 from 0.948 to 0.9435 (the table uses the earlier rule).

**What this actually says.** The policy is roughly **8× the greedy baseline on 4×4 and 118× on
6×6**, and with a small inference budget it solves most boards — but it is neither exact nor
guaranteed, and CP-SAT beats it on both counts. The honest summary of the track is that
**this problem barely needs reinforcement learning**: rewards are extremely sparse, perfect
demonstrations are free, solutions verify themselves, and after masking the mean branching
factor is 1.5 — so the three things RL is good at are all things this puzzle does not need.
Knowing when *not* to reach for RL is the most solid result the track produced.

**Why a single attempt stops short of 100%.** A 6×6 board takes about **14 real choices**, so a
deterministic solve rate of 0.543 means the served policy gets each one right about **95.8%** of
the time (`0.958 ^ 14.22 ≈ 0.543`); reaching 0.85 would take 98.9%, a 3.7× cut in the error rate. The fatal choice is usually made several moves before the walk gets
stuck: even a perfect one-step lookahead adds only +0.03. What is missing is a value that says
"this position can no longer be solved" — which, unlike in Go, an exact solver can label for
free — and search inside the training loop. The analysis, an answer to whether the one-stroke
rule was the right one, and a ranked list of what to try next are in
[`ai-collab/reports/2026-09-19_rl-where-next.md`](./ai-collab/reports/2026-09-19_rl-where-next.md).

Concepts explained from scratch — behaviour cloning, DAgger, PPO fine-tuning, AlphaZero-style
self-improvement — in [`ai-collab/notes/`](./ai-collab/notes/).

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | No NVIDIA GPU or no NVIDIA container toolkit. Expected: `start.py` carries on without the vision model. |
| The build fails in `uv sync` on an Apple Silicon Mac or ARM Linux | torch 2.4.1+cu121 has x86_64 Linux wheels only. The compose files pin `platform: linux/amd64`, so Docker should build under emulation — slow, and **not tested** on ARM hardware. |
| `port is already allocated` | Another checkout's stack (or a `uv run` server) holds the port. `python start.py --status` shows which; set another `APP_PORT` in `.env`. |
| Windows: containers are healthy but `http://127.0.0.1:7440` does not answer | WSL's `networkingMode=mirrored` in `%USERPROFILE%\.wslconfig` breaks Docker Desktop's published ports ([microsoft/WSL#10494](https://github.com/microsoft/WSL/issues/10494)). Comment it out and run `wsl --shutdown`. |
| Screenshot reading answers 503 | The vision model is not in Ollama (see [Model weights](#model-weights)), or Ollama is still loading it — the first call takes about a minute. |
| The RL solver answers 503 | No checkpoint under `models/` (see [Model weights](#model-weights)). |

---

## Project structure

```
linkedin-zip-challenge/
├── start.py                  # One command from a fresh clone to a running stack
├── docker-compose.yml        # Production app; one stack per checkout
├── docker-compose.dev.yml    # Development app (+ hot reload, Svelte dev server)
├── docker-compose.ollama.yml # The one Ollama on the machine, shared by every checkout
├── .devcontainer/
│   ├── Dockerfile            # Multi-stage: Svelte build, then the Python app
│   └── Dockerfile.dev        # Development image (uvicorn --reload)
├── .env.example              # Configuration template; `.env` is created from it
│
├── src/
│   ├── app/                  # FastAPI application
│   │   ├── main.py           # App, CORS, static mounts, Gradio mount
│   │   ├── routers/          # echo / solver / vision endpoints
│   │   └── schemas/          # Request and response models
│   ├── core/
│   │   ├── solvers/          # Classical solvers, the shared registry, the independent judge
│   │   ├── puzzle_generation/# Procedural generator (path first, then carve)
│   │   ├── vl_models/        # Screenshot → board: prompts, backends, scoring
│   │   ├── rl/               # Environment, training, and the serving adapter
│   │   │   ├── rl_env_v2.py            # The environment: masking, rewards, curriculum
│   │   │   ├── train_behaviour_cloning.py
│   │   │   ├── train_maskable_ppo.py
│   │   │   ├── train_config.py         # Every training knob lives here
│   │   │   └── solver_service.py       # The only serving-side file
│   │   ├── tests/            # Core, solver, RL and generation tests
│   │   └── utils.py          # Puzzle type, parser, fitness, visualisation
│   ├── ui/gradio_app.py      # Gradio console (an adapter over the API)
│   ├── custom_components/    # Svelte canvas editor (source + build)
│   └── settings.py           # The single source of configuration
│
├── ai-collab/                # Development documentation (see below)
├── illustrations/            # Screenshots and sample puzzles
├── models/  datasets/  logs/ # Large local artifacts — not in version control
└── pyproject.toml            # Dependencies, pinned via uv.lock
```

---

## Running it without Docker

```bash
cd linkedin-zip-challenge
uv sync                                     # Python 3.11, pinned by .python-version
cp .env.example .env
uv run python -m src.app.main               # http://127.0.0.1:7440/ui
```

The Svelte editor needs a build before it appears at `/svelte-ui`:

```bash
cd src/custom_components/puzzle_editor/frontend && npm install && npm run build
```

Screenshot reading needs an Ollama serving the vision model; `.env`'s `OLLAMA_PROVIDER_URL`
points at the port the compose stack publishes, so running just that container is enough.

---

## Using the console

The Gradio console at `/ui` has one tab per capability:

*   **Generate Puzzle** — create a random board, optionally with blocked cells.
*   **Puzzle Solver (Naive)** — paste a layout and walls as text.
*   **Puzzle Solver (Interactive)** — a canvas editor: click cells to add waypoints, obstacles
    and walls, then solve.
*   **Solve from Screenshot** — upload an image; the answer carries warnings and a `solvable`
    flag alongside the solution.
*   **Echo Test** — confirm the backend is responsive.

Screenshots are in [`illustrations/`](./illustrations/).

---

## Development

```bash
cd linkedin-zip-challenge
uv sync
uv run pytest            # 330 passed, 1 skipped, 8 xfailed as of 2026-09-19
uv run ruff check .
```

The eight `xfail`s are deliberate: they pin defects in the retired v1 environment, so an
unexpected pass means someone changed it. The skip needs the RL checkpoint under `models/`
(inside the app container, with the checkpoint mounted, it runs: 331 passed).

To check a running stack end to end — every served solver over HTTP, with each returned path
judged independently rather than trusting a `200` — use the acceptance scripts in
[`ai-collab/reports/artifacts/wrap-up-acceptance/`](./ai-collab/reports/artifacts/wrap-up-acceptance/).

### Documentation map

| Document | What it holds |
|---|---|
| [`ai-collab/reports/2026-09-19_project-wrap-up.md`](./ai-collab/reports/2026-09-19_project-wrap-up.md) | **The final report.** Status, verification, what is left. **Start here.** |
| [`ai-collab/roadmap.md`](./ai-collab/roadmap.md) | Status, settled decisions, and the history of each track |
| [`ai-collab/model-weights.md`](./ai-collab/model-weights.md) | Where the two models' weights should live and how to install them |
| [`ai-collab/project_guide.md`](./ai-collab/project_guide.md) | Architecture, module responsibilities, how to run each environment |
| [`ai-collab/deployment-guide.md`](./ai-collab/deployment-guide.md) | Docker: what each container does, acceptance checks, troubleshooting |
| [`ai-collab/notes/`](./ai-collab/notes/) | Concepts explained: what the methods are, how to read the numbers, how inference works |
| [`ai-collab/reports/`](./ai-collab/reports/) | One report per experiment — method, numbers, and what was refuted |
| [`ai-collab/handover-rl-solver.md`](./ai-collab/handover-rl-solver.md) | Picking up the RL track: dead ends with evidence, what to do next |
| [`ai-collab/handover-vlm-parser.md`](./ai-collab/handover-vlm-parser.md) | Picking up the vision track |
| [`ai-collab/handover-solvers.md`](./ai-collab/handover-solvers.md) | Picking up the solver registry and API |
| [`ai-collab/dev_log.md`](./ai-collab/dev_log.md) | Full chronological history (long — search it, don't read it) |
| [`AGENTS.md`](./AGENTS.md) | Working agreement for this sub-project (human or AI) |

A Traditional Chinese overview is in [`README_zh-TW.md`](./README_zh-TW.md).

---

## Future work

Development stopped on 2026-09-19. What would come next, in order, is listed with reasons in
the [final report](./ai-collab/reports/2026-09-19_project-wrap-up.md) §10; the short version:

*   **Publish the two models' weights** — the one step that needs the author's accounts.
*   **Learned solver:** finish the half-measured expert-iteration comparison, adopt a single
    move-budget metric (solve within 4n² moves, restarts and backtracking allowed), then train a
    solvability value from exact-solver labels and use it to guide search.
*   **Vision:** make the synthetic evaluation harder; it is saturated at 1.000.
*   **Use the generator as a benchmark** for general computer-use agents: unlimited fresh
    boards, an exact judge, and a canvas an agent can drag on
    ([survey](./ai-collab/reports/2026-09-19_computer-use-agents-and-zip.md)).
