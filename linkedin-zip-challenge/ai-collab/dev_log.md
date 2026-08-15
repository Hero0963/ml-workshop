# Development Log

> Chronological development history of `linkedin-zip-challenge`, **newest first**.
> For the current status and next steps, read [roadmap.md](roadmap.md) instead — this file is the full archive.
> Add one entry per development session, dated `## YYYY-MM-DD`.

## 2026-08-15

### VLM Track P0 + P1: Deployment Smoke Test and Untuned Baseline (branch `feat/vlm-parser`, worktree `zip-vlm`)

Executed stages P0 and P1 of [`plans/2026-08-15_track-vlm-parser.md`](plans/2026-08-15_track-vlm-parser.md).
Full numbers, method and caveats: **[`reports/2026-08-15_vl-p0-p1-baseline.html`](reports/2026-08-15_vl-p0-p1-baseline.html)**.

-   **New: `src/core/vl_models/benchmark.py`** — the measurement harness. Imports the *existing*
    `final_puzzle_parser.build_puzzle_prompt()` rather than copying it, so the baseline cannot drift
    from the prompt it claims to measure. Scores four layers (JSON parse rate; per-cell accuracy,
    waypoint recall and wall P/R/F1 against `src/core/tests/conftest.py`; end-to-end via CP-SAT;
    latency plus `nvidia-smi` peak). Two transports behind `--client`: Ollama's native `/api/chat`
    (full timing counters) and `pydantic-ai` (the path the shipped parser will use). Every call is
    persisted to `ai-collab/reports/artifacts/` with its seed and raw output.
-   **Environment**: `docker compose pull ollama` took the container from **0.16.1 → 0.32.13**; the
    pinned volume kept the 15GB of 2025-10 models. Pulled `gemma4:e4b`, `gemma4:e4b-it-q8_0`,
    `qwen3.5:4b`, `qwen3.5:4b-q8_0` from the official library. All four load **100% on GPU** — a 16GB
    card is not the bottleneck for 4B-class Q8 (worst case 9582 MiB peak).
-   **Quantisation matters far more than expected, and in opposite directions per family.**
    `qwen3.5:4b` at Q4 never emits JSON at all — it thinks for 16,505 characters and burns 6,215
    output tokens hitting the ceiling. The same model at Q8 reads the grid perfectly in 6.2s.
    `gemma4:e4b` is the reverse: Q8 misreads a 6×6 grid as 7×6 while Q4 gets it right.
-   **Disabling thinking is a free, large win — for one family only.** With `think: false`,
    `qwen3.5:4b-q8_0` goes from 3/6 to **6/6** parseable, gets **6/6 grid sizes right** (including the
    two 7×7 puzzles it previously could not answer at all), and runs **5.8× faster** (44.5s → 7.7s).
    The same switch makes `gemma4:e4b` *worse* on every structural metric. Measure it per model;
    never carry the setting over.
-   **The remaining problem is almost purely walls.** Best untuned configuration
    (`qwen3.5:4b-q8_0` + no thinking) scores cell accuracy 0.924 and waypoint recall 0.910 across the
    six screenshots, but wall F1 only 0.410 — and end-to-end is **1/6**. `gemma4:e4b` is 0/6.
    Wall *false positives* are as fatal as misses: on `puzzle_03` gemma4 found all 4 real walls yet
    the puzzle was unsolvable because it hallucinated 2 more.
-   **Two metric traps found and fixed.** Wall-free puzzles score a free F1 of 1.0, which inflated
    gemma4's wall mean from 0.268 to 0.512 — added `mean_wall_f1_walled_only`. And `seed` plus
    `temperature=0` does **not** guarantee determinism: `gemma4:e4b` Q4 produced different answers on
    cold vs warm runs, while the Q8 models were stable. Comparisons need repeats.
-   **Two zero-training interventions, both measured.** Beyond disabling thinking, a `sized` prompt
    variant adds an explicit grid-counting step and a synthetic 7×7 example (generator seed 20260815,
    CP-SAT verified; deliberately *not* puzzle_04/06, which are evaluation data). For
    `qwen3.5:4b-q8_0` this lifts cell accuracy 0.924 → **0.961**, wall F1 0.410 → 0.438, end-to-end
    matches 1/6 → **2/6**, and latency 7.7s → **5.4s**. Read that 2/6 carefully: puzzle_01–03 have
    their answers inside the few-shot prompt, so the previously-correct puzzle_03 was leaked — the
    newly correct one is **puzzle_05, which is not in the prompt**, so the genuinely generalising
    count went 0 → 1.
-   **`gemma4:e4b` got worse under both interventions.** The sized prompt drops it 4/6 → 3/6 on grid
    size, 0.444 → 0.315 on cells and 0.268 → 0.023 on wall F1; told that grids are often not 6×6, it
    over-corrects and reads the genuinely-6×6 puzzle_01 as 7×7. Two independent interventions now
    point the same way: **Qwen absorbs instructions, Gemma is unstable under prompt perturbation.**
    Best untuned configuration is `qwen3.5:4b-q8_0` + no thinking + `--prompt sized`, and that is the
    bar fine-tuning has to clear.
-   **Fine-tuning order revised.** Both families ship an official Unsloth vision notebook at the size
    we need, so that criterion ties. Recommendation is now **Qwen3.5-4B first if paid Colab is
    acceptable** — it only has to learn walls, whereas Gemma must learn size, digits and walls — and
    **Gemma 4 E4B if the free tier is a hard constraint**, since Unsloth explicitly advises against
    QLoRA for Qwen3.5 ("no matter MoE or dense, due to higher than normal quantization differences")
    and a free T4 has no bf16. Also verified first-hand: Qwen3.6 is 27B minimum, Qwen3.7 has no open
    weights, Qwen3.8 is 27B/2.4T — **Qwen3.5 is the only generation with sizes that fit a 16GB card**,
    and Gemma 4 is symmetric (official vision fine-tuning covers E2B/E4B only). Full generation table,
    release dates and a re-verification recipe are in §9 of the report.
-   **Dependencies took three rounds to settle**; see the `build(zip)` commit for the full reasoning.
    `pydantic-ai` was pinned to `==1.107.5`, but the meta-package forces `huggingface-hub>=1.3.4`,
    which is incompatible with `transformers<5`, and `transformers` 5.x silently disables its PyTorch
    backend against the pinned `torch 2.4.1` — that would have left the planned transformers VL
    backend unable to load a model. Settled on **`pydantic-ai-slim[openai]==1.107.5`** (what the
    official Ollama docs recommend, and all this project uses), which dropped **104 packages** —
    anthropic, boto3, cohere, groq, mistralai, google-genai, xai-sdk, temporalio, logfire, mcp and
    the whole opentelemetry stack — and freed `transformers` to stay at 4.57.6.
    ⚠ Separately, **`uv add` could not resolve at all** in this project: the cu121 index is declared
    before PyPI and `index-strategy` lives under `[tool.uv.pip]`, which does not apply to
    `uv add`/`uv lock`/`uv sync`. Fixed structurally by marking that index `explicit` and routing the
    torch trio through `[tool.uv.sources]`; as a bonus the lock now pins them solely to the cu121
    build. **The RL track would have hit the same wall when changing torch.** Two casualties of the
    re-resolution were repaired: `ruff` (only present transitively, so it was pruned and broke the
    documented `uv run ruff check .` — now an explicit dev dependency) and `griffe` (pydantic-ai now
    depends on the renamed `griffelib`, and uninstalling old `griffe` took the shared module files
    with it). Verified after all of it: `pytest` 46 passed, `ruff check` clean,
    `transformers.utils.is_torch_available()` True, torch still `2.4.1+cu121`, and the Gradio UI
    rendered in Chrome — also compared against the main worktree's 5.49.1 to confirm the
    `gradio` 5 → 6 jump caused no regression.
### RL Track A1 — one-stroke env v2, dataset, and the baselines A2 must beat

Curriculum decision changed by the developer before A1 started: **one-stroke all the way,
with reverse curriculum instead of the "allow backtracking, tighten later" phases**
(rationale recorded in the track plan §4). Consequences: revisits are masked from step one,
so the v1 2-cycle is impossible by construction and the `visit_count` / `visit_recency`
channels were dropped; every training success is now a legal Zip solution.

-   **`src/core/rl/rl_env_v2.py`** — `PuzzleEnvV2`: Dict observation (8 channels padded to
    8×8 + 8 scalars), `action_masks()` covering bounds / blocked / walls / visited /
    out-of-order numbers, dead-end termination before an all-False mask can reach the
    sampler, sparse reward (+1 success, 0 otherwise) with optional potential-based coverage
    shaping. Legality mirrors `dfs.py:96-105`, and reset collects number 1 exactly like
    `dfs.py:72-77` — the detail v1 got wrong. 21 unit tests, all passing, including the
    ground-truth replay v1 failed 0/7.
-   **`src/core/rl/generate_dataset_v2.py`** — deterministic dataset builder that *keeps the
    solution path* (the old `generate_rl_dataset.py:59` discards it, which reverse curriculum
    cannot afford) and splits train/val/test per size. The old script is untouched.
-   **Generation cost fixed, 18x**: the generator's default `timeout_per_attempt=20s` is spent
    proving that wrong-parity start cells are impossible. Measured on 7×7: successful searches
    finish in ≤0.415s at a 0.5s cutoff and ≤1.606s at 2s. Dropping the cutoff to 0.5s took the
    5,100-puzzle build from a projected ~23 hours to **45 seconds**, and 100 7×7 puzzles from
    ~14 minutes to **35 seconds**. This is a call-site parameter; the shared generator was not
    modified.
-   **Baselines on 510 held-out puzzles × 20 episodes** (`logs/rl_baselines/`):

    | policy | 4×4 | 5×5 | 6×6 |
    |--------|-----|-----|-----|
    | masked random | 8.8% | 0.9% | 0.0% |
    | greedy (distance to next number) | 10.2% | 3.7% | 0.8% |

    Dead ends account for 90–100% of failures, confirming that under one-stroke rules the
    dominant failure mode is getting trapped, not running out of budget. Greedy is the ceiling
    of what distance-based shaping can teach, and it collapses by 6×6 despite the highest
    coverage (0.595) — **an experimental confirmation of restart-plan §2.2**, which until now
    was a static argument.
-   **Dependency settled**: `uv add sb3-contrib==2.7.1 --index-strategy unsafe-best-match`
    (the project's `index-strategy` lives under `[tool.uv.pip]`, which `uv add` ignores).
    Verified afterwards: `torch 2.4.1+cu121` and `stable-baselines3 2.7.0` unchanged,
    `MaskablePPO` imports, suite still green. sb3-contrib is the official SB3 contrib package
    (Antonin Raffin / DLR, MIT); 2.7.1 was released 2025-12-05.

Suite after A1: **76 passed, 8 xfailed**, `ruff check` clean. Next is A2 (Phase 1 training on
4×4 with reverse curriculum), which is the first stage that actually trains anything.

### RL Track A0 — env v1 is not merely hard to learn, it is unsolvable (branch `feat/rl-masked-ppo`)

The A0 sanity stage of [plans/2026-08-15_track-rl-solver.md](plans/2026-08-15_track-rl-solver.md) ran in the
`zip-rl` worktree. Baseline first: `uv sync`, `uv run pytest` → **46 passed**, `ruff` clean, matching the
2026-08-08 record. Then six probes were run against **unmodified** `src/core/rl/rl_env.py`. Full write-up:
[reports/2026-08-15_a0-env-v1-findings.md](reports/2026-08-15_a0-env-v1-findings.md).

-   **Replaying a ground-truth solution never terminates — 0/7 puzzles.** `reset()` puts the agent on
    waypoint 1 but leaves `_next_waypoint_idx` at 0, and the collection check only runs *after* a move
    (`rl_env.py:143-146`, `:199-208`). A legal one-stroke path never re-enters the start cell, so the
    waypoint index is pinned at 0 and `terminated` is unreachable. Every fixture path covered all cells
    (36/36, 49/49) and still scored about −35 to −48.
-   **The success bonus is reserved for illegal paths.** Prefixing the same solution with a single
    step off and back onto the start cell terminates **6/6** fixtures with **+999.01** (episode totals
    +2359 to +4946). `all_cells_visited` uses `len(set(path_taken))`, so revisits are not penalised at
    the terminal check. env v1's reward is therefore anti-correlated with the rules of Zip: the best
    scoring strategy it can teach is a cheat. This amends §2.4 of the restart plan — the probability of
    a *legal* positive sample was not "close to zero", it was exactly zero.
-   **The 2-cycle hypothesis is confirmed, so the v2 design stands.** Oscillating between two visited
    cells yields exactly 2 distinct observation hashes over 8 steps, and a deterministic 2-state policy
    built from them ran 69 steps to truncation without ever escaping, touching only those 2 cells.
-   **Illegal moves do not consume the step budget**: 82 boundary bumps against a budget of 72 produced
    1 distinct observation and never reported truncation (`:176-180` hard-codes `truncated=False`).
-   **Side finding for A1**: `generate_puzzle` fails on odd open grids by parity — a 5×5 start-cell sweep
    gave 13/13 success on `(r+c)` even and 0/12 on odd, so ~2.5% of seeds exhaust all retries and return
    `None`. Dataset generation must retry with a new seed. The generator itself was left untouched
    (shared module, read-only per the track plan).

Added: `src/core/rl/diagnose_env_v1.py` (six probes, JSON evidence to the git-ignored
`logs/rl_diagnostics/`), `src/core/tests/rl/test_rl_env_v1_diagnosis.py`, and `src/core/rl/action_space.py`
(shared path→action encoding). The unsolvable replays are pinned with `xfail(strict=True)` so the suite
stays green while failing loudly if v1 is ever changed. After the additions:
**55 passed, 8 xfailed**, `ruff check` clean. `rl_env.py` and the old checkpoints were not touched.

### Planning Reports for the VLM and RL Tracks (research only, no code changed)

Two design reports were written to unblock roadmap items #2 (VL integration) and #3 (RL restart).
No source code was modified in this session; the work was code reading plus external verification.

-   **`ai-collab/reports/2026-08-15_vlm-model-survey.html`** — model selection and fine-tuning plan
    for `image -> puzzle JSON`:
    -   Recommends **Qwen3.5-4B** (natively multimodal, Apache-2.0) with Unsloth **bf16 LoRA**
        (Unsloth explicitly advises against QLoRA for Qwen3.5); `Qwen3-VL-4B/8B` as the fallback.
    -   Training data comes from the existing puzzle generator: labels are free and exact, but a new
        LinkedIn-style renderer is required — `save_solution_as_image()` draws a *solution* in a
        different visual style than the real screenshots (black circles, thick wall bars, UI chrome).
    -   Two deployment landmines were found and documented: unsloth#3899 (garbled GGUF after vision
        fine-tuning) and ollama#14730 (imported GGUF + mmproj fails on some architectures). Hence the
        plan starts with a **deployment smoke test before any training**.
    -   Colab: the official Colab CLI (2026-06-05) is **Linux/macOS only**, so on Windows either use
        WSL2 or the VS Code Colab kernel extension.
    -   Also evaluates the "one-shot" variant the developer asked about, splitting it into an
        *agent-orchestrated* one-shot (parse → existing solver, cheap and reliable) and a
        *model end-to-end* one-shot (research-grade, hands off to the RL report).
-   **`ai-collab/reports/2026-08-15_rl-restart-plan.html`** — the RL restart plan required by
    roadmap item #3, covering two routes:
    -   **Route A (recommended first)**: a dedicated agent with **action masking**. Reading
        `rl_env.py` produced a sharper root cause for the 2025-10 failure than the original
        diagnosis: because `ch_path` is binary and the step counter is absent from the observation,
        an agent oscillating between two already-visited cells produces an observation sequence
        `o_A, o_B, o_A, ...` — a genuine 2-cycle, so a deterministic policy is *provably* trapped.
        Illegal moves are an even more degenerate single-state loop. Masking removes both by
        construction. Other findings: the potential used for shaping (distance to the next waypoint)
        is not isomorphic to the real objective (full coverage), and the observation exposes only the
        *next* waypoint, making long-horizon planning impossible in principle.
    -   **Route B**: GRPO/GSPO post-training of a language model, using the existing solver and
        `calculate_fitness_score()` as a verifier (RLVR). Recommends text-only input and 4x4 grids
        first, so vision and reasoning are not debugged simultaneously.

### Both reports revised to v2 after review

The developer reviewed both reports and pushed back on five points; both were rewritten the same day.

-   **VLM report v2**:
    -   The v1 survey was **out of date** and is now corrected: **Gemma 4** shipped 2026-03/04 with
        five image-capable sizes (E2B/E4B/12B/26B-A4B/31B) under a plain **Apache-2.0** license, while
        **Qwen 3.7/3.8 went closed (API-only)** — the newest *small open-weight* Qwen VL is still the
        Qwen3.5 series, and Qwen3.6 has nothing under 10B.
    -   New section on **what this machine can actually run**: 7–9B inference is *not* a problem
        (~6GB at Q4), the ceiling is training — Qwen3.5-9B bf16 LoRA needs 22GB and does not fit 16GB.
    -   Recommendation changed from a single model to a **split by Colab tier**: the free T4 is Turing
        and has no bf16, and Unsloth advises against QLoRA for Qwen3.5, so the free path is
        **Gemma 4 E4B QLoRA (10GB)** while the paid L4/A100 path is **Qwen3.5-4B bf16 LoRA**. Plan now
        trains both families on the same data and compares.
    -   New section on **OCR-specialist models** (PaddleOCR-VL 0.9B, DeepSeek-OCR 2, dots.ocr): the
        task looks like OCR but the bottleneck — "this bar separates cell (2,3) from (3,3)" — is a
        relation-extraction problem outside their pretraining. Verdict: cheap enough to run as a
        parallel B-arm, not the main line.
    -   New section on the **Unsloth notebook catalogue** (250+ notebooks; 30+ vision, 40+ GRPO/RL,
        OCR incl. DeepSeek-OCR and Paddle OCR). Notably a **Gemma 4 E2B Sudoku GRPO notebook** exists,
        which is the closest available template for the RL route B reward design.
-   **RL report v2** — route A's experiment was **redesigned from scratch** at the developer's request:
    -   Adopts the developer's two proposals: **allow backtracking** (easier to train than forcing a
        one-stroke path from the start) and **put visit counts in the observation**. The second one
        directly dissolves the 2-cycle diagnosed in v1; the report adds that a **strictly monotonic
        `steps_used / budget` scalar** is also needed, because a clipped visit counter can saturate.
    -   Observation is now 9 channels (valid mask, two wall planes, **visit count**, visit recency,
        agent, next/future/done waypoints) plus 6 global scalars; the `wp_future` plane fixes the v1
        finding that the agent could not see waypoints beyond the next one.
    -   Reward is now **FrozenLake-style**: +1 on success, 0 otherwise, with "finish faster" expressed
        by the discount factor rather than a per-step penalty (the old -1/step accumulated to -72 and
        drowned the +1000 terminal signal). Shaping potential switched from *distance to next waypoint*
        to **coverage ratio**, which is isomorphic to the real objective.
    -   Three-phase curriculum on constraint strictness: free backtracking → priced backtracking →
        hard-masked one-stroke. Explicitly notes that only the last phase produces a *legal* Zip
        solution, so "soft success rate" and "legal one-stroke rate" must be reported separately.
    -   **Dependency conflict found**: `MaskablePPO` lives in `sb3-contrib`, whose latest (2.9.0)
        requires `stable-baselines3>=2.9.0`, which in turn requires `torch>=2.8` — but this project
        pins `torch==2.4.1+cu121` (a deliberate decision recorded in `../AGENTS.md §5`). Three options
        documented; the choice needs the developer's call since packages are installed manually.
    -   Also adds a build-your-own assessment: ~620–860 lines using sb3-contrib, ~970–1310 lines fully
        hand-rolled (masking itself is ~40 lines in the CleanRL style).

### Ollama brought back as a Docker service (dev stack)

An earlier claim in this session — "Ollama is not installed" — was **wrong**: it had only been checked
as a native install. It runs in Docker here, and the 2025-10 assets were all still intact.

-   **Found**: image `ollama/ollama` present; volume `linkedin-zip-challenge_ollama_data` still holds
    ~15GB of blobs with three models (`openbmb/minicpm-o2.6`, `qwen2.5vl:7b`,
    `bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`). `docker-compose.yml.vl_version` already contained
    a working `ollama` service definition, but it was never merged into the active dev stack.
-   **Changed**:
    -   `docker-compose.dev.yml` — added the `ollama` service with the NVIDIA device reservation, a
        healthcheck, and a `volumes:` block pinning `ollama_data` to the pre-existing
        `linkedin-zip-challenge_ollama_data` so the 15GB is reused rather than re-downloaded.
        Container is named `zip_ollama_server` and the host port is `${OLLAMA_HOST_PORT:-11435}`:
        the name `ollama_server` and port 11434 are already taken on this machine by an unrelated
        project (verified via the container's compose labels).
    -   `.env` (not versioned) — re-enabled `OLLAMA_MODEL_NAME` / `OLLAMA_PROVIDER_URL` (both had been
        commented out). In-network URL is `http://ollama:11434/v1`; the host-side alternative is noted
        in a comment.
    -   `run_docker_dev.py` — added a non-fatal `check_ollama_ready()` that polls `/api/tags` and
        prints which models are available, plus the Ollama endpoint in the final summary.
-   **Verified** (2026-08-15): `docker compose -f docker-compose.dev.yml config` OK;
    `docker compose up -d ollama` starts; inside the container `nvidia-smi` reports
    `NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB` (GPU passthrough works) and `ollama --version` is
    0.16.1; `ollama list` shows all three old models; host `GET :11435/api/tags` returns 200;
    `uv run ruff check run_docker_dev.py` passes.
-   **Not verified**: the app container reaching `http://ollama:11434/v1` (would require building the
    app image); and whether ollama 0.16.1 — a ~6-month-old cached image — can serve Qwen3.5 / Gemma 4.
    `docker compose pull ollama` is required before P0.

### Task plans written for two parallel worktree tracks

The two reports explain *why*; these new plans say *what to do*, and are written for a fresh agent
landing in an empty worktree.

-   **New directory `ai-collab/plans/`** (documented in both `AGENTS.md` files):
    -   `2026-08-15_track-vlm-parser.md` — P0 deployment smoke test → P1 baseline → P2 data pipeline
        → P3 real eval set → P4 SFT → P5 integration → P6 one-shot endpoint.
    -   `2026-08-15_track-rl-solver.md` — A0 env sanity → A1 env v2 (9 channels + 6 scalars, masking,
        FrozenLake-style reward) → A2/A3/A4 three-phase curriculum → A5 ship as the 10th solver.
-   **Worktree gotchas documented up front**, because a fresh worktree only gets version-controlled
    files: `.env` must be copied by hand (its absence broke startup back in 2025-10), each worktree
    needs its own `uv sync`, `datasets/rl_datasets/` is empty even in the main tree, and `models/`
    holds only the failed 2025-10 DQN checkpoints (do not resume from them). `illustrations/
    puzzle_01..06.png` *are* tracked, so the VLM track has its eval material from the start.
-   **Cross-track coordination rules**: code barely overlaps (`vl_models/` vs `rl/`), but
    `src/core/utils.py` and `src/core/puzzle_generation/` are read-only for both;
    `pyproject.toml`/`uv.lock` changes are serialised through the developer; `roadmap.md` and
    `dev_log.md` edits stay in each track's own section; and the Docker stack must not be started
    from two worktrees at once (container name and host port are machine-unique).

## 2026-08-08

### AI Collaboration Scaffold and Environment Recovery

The project had been dormant since 2025-10-30 (~9 months). This session rebuilt the collaboration
documentation so that any agent (or future self) can pick the project up without re-deriving context,
and verified that the toolchain still works.

-   **New documentation structure** (mirroring the conventions already used by `board-game-rl` and
    `deep-learning-karpathy` in this monorepo):
    -   `AGENTS.md` — the single source of truth for how to work on this project: startup routine,
        document ownership, five-step task workflow, environment/verification requirements, a
        task→file map, code conventions, and hard limits.
    -   `CLAUDE.md` — a one-line `@AGENTS.md` import, so opening Claude Code directly in this
        directory loads the same rules.
    -   `ai-collab/roadmap.md` — **the new first stop**: current status, prioritised next steps with
        explicit done-criteria, and a "settled decisions, do not reopen" table distilled from the
        660 lines of history in this file.
    -   `ai-collab/project_guide.md` — architecture (the three-layer Core/App/UI split), module
        responsibilities, the nine solvers, the API contract, and all three ways to run the app.
    -   `ai-collab/commands.txt` — reusable prompts and a command cheatsheet.
    -   `ai-collab/reports/` — directory for future task reports.

-   **File moves and link updates**:
    -   `dev_log.md` moved from the project root into `ai-collab/` via `git mv`, matching the other
        two sub-projects. Links in `README.md`, `README_zh-TW.md` and `gemini_readme_raw.md` updated.
    -   The duplicated `# Development Log` heading at the top of this file was removed.
    -   `gemini_readme_raw.md` marked as a superseded historical artefact, pointing to `AGENTS.md`.

-   **Environment determinism**:
    -   Added `.python-version` pinning **3.11**, so `uv sync` no longer has to guess a version that
        satisfies `requires-python = ">=3.11,<3.12"`.
    -   Documented explicitly (here and in the repo-level `AGENTS.md`) that this project is
        **deliberately kept out of the root `uv` workspace** — it pins `torch==2.4.1` with a cu121
        index against Python 3.11, which conflicts with the repo-root 3.9 devtools environment.

-   **Recovery verification** (the point of the exercise — the baseline is good):
    -   `uv sync` → resolved 190 packages, audited 172, no changes required.
    -   `uv run python -c "import sys; print(sys.version)"` → `3.11.13`.
    -   `uv run pytest` → **46 passed in 8.10s**.
    -   `uv run pre-commit run --all-files` → `ruff-format` and `ruff` both passed.

-   **End-to-end verification against a live server** (started with the documented
    `uv run python -m src.app.main`, not `TestClient`):
    -   `GET /` → 200; `GET /api/echo/health` → `{"status": "ok"}`; `POST /api/echo/` → `Echo: zip`; `GET /docs` → 200.
    -   `POST /api/solver/solve` for **DFS**, **A\* (heapq)** and **CP-SAT** against `puzzle_01`: all three
        returned a 36-step path **identical cell-by-cell to `solution_01` in `conftest.py`**, plus a
        ~74 KB animated GIF and a ~7.6 KB final PNG each (byte-identical across solvers, i.e. they
        converge on the same solution). The rendered PNG was inspected: waypoints, walls and the green
        step-order overlay all match the returned path (start `(1,1)` = step 1, `(0,0)` = step 5).
    -   Error paths: malformed layout → 400; unknown solver → 404.
    -   Chrome headless (`--dump-dom` + `--screenshot`) on `/ui`, `/svelte-ui` and `/docs`: the Gradio
        console renders all four tabs, the Svelte editor hydrates and draws its 320×320 canvas grid,
        and Swagger lists every endpoint.

### Small Defects Found During Verification (recorded, not fixed)

-   **Swagger shows the Echo endpoints twice.** `src/app/main.py` includes the router with
    `tags=["Echo"]` while `src/app/routers/echo.py` already declares `tags=["echo"]`; FastAPI merges
    both, producing two identical groups in `/docs`.
-   **The Svelte "Instructions" panel prints raw Markdown** — `**middle**` and `**border**` render as
    literal asterisks because that text is not passed through a Markdown renderer.
-   **The 3-of-9 solver gap is confirmed on both front-ends**, not just the API: the compiled Svelte
    bundle only contains the strings `DFS`, `A* (heapq)` and `CP-SAT`.

-   **Documentation fix**: the "Running Tests" snippet in `README.md` rendered as a broken two-line
    command (`.` followed by `un_tests.bat`) because the backslash-r was consumed; corrected to
    `.\run_tests.bat` and preceded by the `uv sync` / `uv run pytest` workflow.

### Known Gaps Recorded (not fixed in this session)

-   `src/app/routers/solver.py` only exposes **3 of the 9 implemented solvers** (`DFS`, `A* (heapq)`,
    `CP-SAT`). The six metaheuristic solvers are implemented and tested but unreachable from the API,
    the Gradio dropdown, or the Svelte dropdown. This is now the top item in `roadmap.md`.
-   `src/custom_components/puzzle_editor/frontend/Dockerfile` still exists, although the 2025-10-28
    entry below records it as removed during the Docker overhaul.

## 2025-10-30

### Documentation Refinement and Environment Verification

Conducted a comprehensive review and update of project documentation (`README.md`, `README_zh-TW.md`), alongside a thorough verification of local and Dockerized development environments. This phase focused on improving clarity, consistency, and ensuring the project's operational readiness.

-   **Documentation Enhancement**:
    -   Updated Svelte UI descriptions to accurately reflect its Canvas-based WYSIWYG editing capabilities.
    -   Clarified service access instructions, emphasizing unified access via `APP_PORT` and segregating developer-specific hot-reloading details.
    -   Added "Highlights" and "Technologies Used" sections to `README.md` for a comprehensive project overview.
    -   Integrated a note directing users to the `illustrations/` directory for visual aids and UI screenshots.

-   **Environment Operationalization & Debugging**:
    -   **Unified Settings Management**: Migrated `SVELTE_PORT` to `src/settings.py` for centralized configuration. Its reliance on the `.env` file for `docker-compose.dev.yml` was removed by hardcoding the value in the compose file.
    -   **`run_docker_dev.py` Debugging**:
        -   Resolved initial `SVELTE_PORT` not set errors (addressed by ensuring `.env` was correctly configured).
        -   Diagnosed and fixed FastAPI application startup failures within Docker containers.
        -   Identified that `Dockerfile.dev` initially lacked a `CMD`, causing containers to exit prematurely (addressed by adding `CMD ["tail", "-f", "/dev/null"]`).
        -   Discovered `docker compose exec -d` suppressed FastAPI startup logs (addressed by removing the `-d` flag).
        -   Pinpointed `pydantic.ValidationError` for `ollama_model_name` and `ollama_provider_url` (due to `.env` not being copied into the container).
        -   Corrected `Dockerfile.dev` to copy the `.env` file into the container, ensuring environment variables are properly loaded by `pydantic-settings`.
        -   (Note: Temporarily set default empty strings for `ollama_model_name` and `ollama_provider_url` in `src/settings.py` as a workaround for startup.)

-   **Environment Verification**:
    -   Confirmed the local environment setup instructions are accurate.
    -   Confirmed the Docker development environment (`run_docker_dev.py`) is operational after resolving startup issues.
    -   Confirmed the Docker production environment (`docker-compose.yml`) instructions are accurate.

## 2025-10-28

### Project Production-Ready Refactoring

Conducted a major refactoring initiative to improve code quality, streamline the user interface, and professionalize the deployment workflow. This effort touched upon configuration management, code duplication, error handling, and the entire Docker setup.

#### Phase 1: Code Quality and Consistency

-   **Settings Centralization**:
    -   Standardized all core application settings in `src/settings.py`.
    -   Centralized `app_port` and `app_host` to remove hardcoded values in the UI and utility scripts.
    -   Formalized `ollama_model_name` and `ollama_provider_url` to use Python's `snake_case` convention for internal consistency.
    -   Identified and removed the obsolete `svelte_port` setting after the frontend integration.

-   **DRY Principle Refactoring**:
    -   Identified significant code duplication in the setup phase of various solvers.
    -   Created a new `prepare_solver_input` utility function in `src/core/utils.py` to consolidate common logic for puzzle parameter extraction and validation.
    -   Refactored the `dfs.py` and `a_star.py` solvers to use the new utility function, significantly reducing their boilerplate code.
    -   Extended the refactoring to `generate_random_path` in `utils.py`, benefiting all metaheuristic solvers that depend on it.

-   **Error Handling and Logging**:
    -   Reviewed the API endpoint (`src/app/routers/solver.py`) and the Gradio UI (`src/ui/gradio_app.py`).
    -   Enhanced exception logging by replacing `logger.error(f"...")` with `logger.exception("...")` in the main solver API, ensuring full stack traces are captured for unexpected errors.
    -   Added error logging to the Gradio UI's API calling functions, which previously failed silently in the server logs.

#### Phase 2: UI Enhancements and Frontend Integration

-   **Svelte UI Integration**:
    -   Successfully integrated the standalone Svelte frontend into the main FastAPI application.
    -   Modified `vite.config.ts` to set the `base` path to `/svelte-ui/`, fixing asset loading issues.
    -   The FastAPI application in `src/app/main.py` now serves the built static files (`dist` directory) from the `/svelte-ui` path.

-   **New "Generate Puzzle" Feature**:
    -   Added a new "Generate Puzzle" tab to the Gradio UI.
    -   Implemented the UI with a dropdown to select the number of blocked cells (0, 1, or 2) and a button to trigger generation.
    -   The UI displays a preview image of the generated puzzle and provides the layout/walls in a copy-paste friendly format.
    -   Added a new unit test (`test_generate_puzzle_ui_success`) for this feature, using mocking to ensure its reliability.

#### Phase 3: Docker Workflow Overhaul

-   **Dual-Environment Strategy**:
    -   To balance development convenience with production-readiness, a dual-environment Docker setup was implemented.
    -   **Development (`docker-compose.dev.yml`)**: A new configuration was created to restore the two-container (backend + Svelte dev server) setup, enabling full hot-reloading for both frontend and backend development.
    -   **Production (`docker-compose.yml`)**: The main compose file was streamlined to define a single, self-contained service for production.

-   **Multi-Stage Production Dockerfile**:
    -   The main `.devcontainer/Dockerfile` was converted into a multi-stage build file.
    -   A `node:lts-alpine` stage is now used to build the production-optimized Svelte frontend (`npm run build`).
    -   The final Python stage copies the application code and the compiled frontend `dist` directory, creating a single, efficient, and immutable production image.

-   **Workflow Automation**:
    -   The `run_docker_dev.py` script was updated to default to using the new `docker-compose.dev.yml`, ensuring the best out-of-the-box experience for developers.
    -   Obsolete files (`frontend/Dockerfile`) and settings (`svelte_port`) were identified and removed to maintain project cleanliness.


## 2025-10-24 (Second Entry)

### Environment Deep Dive: Resolving Fine-Tuning Dependencies

With the decision made to proceed with fine-tuning, the next phase involved setting up the development environment to handle the complex dependencies required by Unsloth. This process revealed several layers of platform and package incompatibilities.

-   **Initial `xformers` Failure:** An attempt to install `unsloth` on the host Windows machine failed due to the `xformers` package lacking compatible wheels for Windows. This validated the necessity of using the project's Dockerized Linux environment for all fine-tuning tasks.

-   **Dependency Resolution in Docker:** Moving into the Docker container revealed a series of deeper dependency conflicts when trying to install `unsloth` into the project's existing environment:
    1.  A `numpy` version conflict arose due to `uv`'s default index strategy, which was resolved by using the `--index-strategy unsafe-best-match` flag.
    2.  A subsequent, more complex conflict was discovered between the project's pinned versions of `torch` and `transformers`, and the different versions required by `unsloth`.

-   **Root Cause Analysis: Build-time vs. Runtime Environment:** The final installation attempt failed while trying to build the `flash-attn` package. The error `OSError: CUDA_HOME environment variable is not set` and the warning `nvcc was not found` led to the root cause: the service's base Docker image (`python:3.11-slim`) was a **runtime** image, lacking the NVIDIA CUDA development toolkit required to **compile** custom CUDA extensions.

-   **Solution: Environment Isolation and `devel` Image:**
    1.  The `.devcontainer/Dockerfile` was modified to use `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` as its base image, which includes the full CUDA toolkit.
    2.  A new workflow was established: create a separate, isolated virtual environment (`unsloth_env`) inside the rebuilt Docker container to prevent any conflicts with the main project's dependencies.
    3.  A robust, multi-step `pip install` process was defined to first install `torch` from its specific index, followed by installing `unsloth` and its dependencies using the `--no-build-isolation` flag to ensure the build process could find the pre-installed `torch`.

-   **Next Step:** With a correctly configured and isolated environment, the next step is to execute the SFT training script (`train_puzzle_sft.py`) inside the new container setup.


## 2025-10-24

### Final VL Model Validation & Success of the Hybrid Strategy

Following the previous entry, the initial plan to pivot to Strategy A was revised to conduct a final, conclusive test of Strategy B (`pydantic-ai`).

-   **Final Capability Test of Model 1 (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`)**
    -   The test script was modified to request a structured Pydantic object (`AnimalInfo`) as the `output_type`.
    -   **Finding:** The model successfully returned a **structurally correct but empty** Pydantic object.
    -   **Conclusion:** This definitively proved that the `bsahane` model **supports tool-calling**, but its core **vision module is defective**, preventing it from providing any content.

-   **Capability Test of Model 2 (`openbmb/minicpm-o2.6`)**
    -   After replacing the model with `openbmb/minicpm-o2.6`, the same structured output test was performed.
    -   **Finding:** Received a definitive `400 Bad Request` error from the Ollama server with the message: `...does not support tools`.
    -   **Conclusion:** This proved that the `minicpm` model **does not support** the tool-calling API required by `pydantic-ai`.

-   **The Hybrid Strategy: Proposal and Success**
    -   Faced with a dilemma where one model had tool support but broken vision, and the other had working vision but no tool support, a new "hybrid strategy" was adopted. This approach continues to use `pydantic-ai` for its convenient API, but sets the `output_type` to `str` and leverages **Prompt Engineering** to instruct the model to generate a JSON-formatted string in its raw text response.
    -   The `experiment_minicpm_json_prompt.py` script was created to validate this strategy.
    -   **Result:** **Complete success.** The `minicpm` model correctly identified the image content (cat, bird) and returned a perfectly formatted JSON string, which was then successfully parsed in Python.

-   **Final Conclusion**
    -   A complete and viable technical pipeline has been established. The combination of a **vision-capable model (`minicpm`)** with the **`pydantic-ai` + Prompt Engineering** software pattern will serve as the foundation for the actual puzzle parser development.


## 2025-10-22 (Fourth Entry)

### VL Model Strategy Refinement & Tool-Calling Explained

Building on the experimental plan from the "Third Entry," this entry refines the VL model validation strategy and provides a deep dive into "Tool-Calling" to clarify why it is core to `pydantic_ai`'s structured output.

-   **Important Clarification: Experimental Phase**
    -   All current Vision-Language (VL) model integration work is in an **experimental phase**.
    -   All code within the `src/core/vl_models/` directory (including `vl_extractor.py`, `hf_parser.py`, and the new PoC scripts) should be considered a **"Scratchpad"**.
    -   The purpose of these scripts is to rapidly validate model capabilities and integration feasibility. They should not be considered final production code until the features are proven and standardized.

-   **Phase 1: Vision Sanity Check (Refined)**
    -   **Objective:** To validate the basic visual understanding of the new model (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`).
    -   **Test Assets:** The test images will be updated to `cat.jpg` and `bird.jpg`.
    -   **Methodology:** Continue using the `vision_sanity_check.py` script, asking the model a question (e.g., "Please describe the animal in the image and its primary color"), and expecting a reasonable natural language string response.

-   **Phase 2: Tool-Calling Proof of Concept (PoC)**
    -   **Objective:** To strictly verify if the model supports the "Tool-Calling" feature required by `pydantic_ai` for structured data output.
    -   **Methodology:** Use the `tool_calling_poc.py` script, which defines an `IdentifiedAnimal` Pydantic model and configures the `pydantic_ai` Agent with `output_type=IdentifiedAnimal`. This will directly test if the model can return a JSON object compliant with the Pydantic model, rather than just a `str`.

### Technical Deep Dive: "Tool-Calling" & `pydantic_ai`

-   **What is "Tool Support" (Tool-Calling)?**
    -   This is a key capability of an LLM (or VLM). **It does not mean the model "executes" code itself**.
    -   Instead, it means the model is trained to understand the "tool" definitions (i.e., a function's schema, including its name, parameters, and parameter types).
    -   When the model believes it needs to use a tool to answer a query (e.g., user asks "What's the weather in Miami?"), it **outputs a structured JSON request**, such as: `{"name": "get_weather", "arguments": {"city": "Miami"}}`.
    -   Our application (e.g., the Python script) receives this JSON and *then* the application *itself* executes the corresponding `get_weather("Miami")` function.
    -   This capability allows the model to interact with external APIs, databases, or local functions to retrieve real-time information or perform actions.

-   **How does `pydantic_ai` use Tool-Calling for `output_type`?**
    -   `pydantic_ai` cleverly abstracts this "Tool-Calling" mechanism.
    -   When we set `output_type=IdentifiedAnimal` in a `pydantic_ai` Agent:
        1.  `pydantic_ai` automatically reads the structure of the `IdentifiedAnimal` Pydantic model.
        2.  It converts this Pydantic structure into a "Tool" schema that the LLM can understand (something like: `{"name": "IdentifiedAnimal", "parameters": {"animal_name": "string", "color": "string", ...}}`).
        3.  `pydantic_ai` sends this schema, along with our prompt, to the VL model.
        4.  **If** the model supports tool-calling (like the `...-Instruct` version), it will recognize that we want it to "call" the `IdentifiedAnimal` tool and will generate a JSON string matching that schema.
        5.  **If** the model does not support it (like our previous `qwen2.5vl:7b`), it will ignore the schema and just return whatever natural language `str` it wants, causing `pydantic_ai` to fail parsing.
    -   This is precisely why the `bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh` model is critical; it claims to support this feature, which is the core hypothesis the Phase 2 PoC is designed to test.

### References

-   [1] IBM (2025). *What Is Tool Calling?*. Retrieved 2025-10-22, from: `https://www.ibm.com/think/topics/tool-calling`
-   [2] Analytics Vidhya (2025). *Guide to Tool Calling in LLMs*. Retrieved 2025-10-22, from: `https://www.analyticsvidhya.com/blog/2024/08/tool-calling-in-llms/`
-   [3] Medium (2025). *Understanding LLM Tool Calling*. Retrieved 2025-10-22, from: `https://medium.com/garantibbva-teknoloji/understanding-llm-tool-calling-traditional-vs-embedded-approaches-fc7e576d05de`
-   [4] Medium (2024). *Tool Calling for LLMs: A Detailed Tutorial*. Retrieved 2025-10-22, from: `https://medium.com/@yasir_siddique/tool-calling-for-llms-a-detailed-tutorial-a2b4d78633e2`
-   [5] PromptLayer Blog (2024). *Tool Calling with LLMs: How and when to use it?*. Retrieved 2025-10-22, from: `https://blog.promptlayer.com/tool-calling-with-llms-how-and-when-to-use-it/`
-   [6] LangChain Docs (2025). *Tool calling*. Retrieved 2025-10-22, from: `https://python.langchain.com/docs/concepts/tool_calling/`

### To-Do / Next Steps

1.  **[User]** Prepare `cat.jpg` and `bird.jpg` image files and place them in the `illustrations/` directory.
2.  **[Dev]** Ensure the `src/core/vl_models/vision_sanity_check.py` script is updated to use `cat.jpg` and `bird.jpg` for testing.
3.  **[User]** Execute the Phase 1 test: `python src/core/vl_models/vision_sanity_check.py` and report the results.
4.  **[User]** If Phase 1 is successful, execute the Phase 2 test: `python src/core/vl_models/tool_calling_poc.py` and report the results.
5.  **[Dev]** Based on the results of Phase 1 and Phase 2, jointly decide on the next implementation strategy for Puzzle extraction.

## 2025-10-22 (Third Entry)

### VL Model Experimental Plan

Finalized a two-phase experimental plan to validate the capabilities of the newly selected VL model (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`) before integrating it into the main puzzle-solving workflow. This approach defers the decision on the final implementation (manual JSON parsing vs. direct tool-calling) until the model's capabilities are confirmed.

-   **User-Provided Research:** The new model was selected based on user research indicating that it is an instruction-tuned vision model that explicitly supports the "tool-calling" feature, which was the blocker for the previous model.

-   **Phase 1: Vision Sanity Check**
    -   **Objective:** To perform a basic test of the model's core visual understanding.
    -   **Implementation:** A new script, `src/core/vl_models/vision_sanity_check.py`, was created.
    -   **Methodology:** This script uses the existing `VLExtractor` (which expects a `str` output) to ask the model to identify the animal and its primary color from `cat.jpg` and `dog.jpg`. This tests the model's ability to follow simple instructions and describe an image without complex formatting requirements.

-   **Phase 2: Tool-Calling Proof of Concept (PoC)**
    -   **Objective:** To verify if the new model truly supports the `pydantic_ai` tool-calling feature for structured data output.
    -   **Implementation:** A second new script, `src/core/vl_models/tool_calling_poc.py`, was created.
    -   **Methodology:** This script defines a simple `IdentifiedAnimal` Pydantic model with `animal_name`, `color`, and `confidence` fields. It then configures a `pydantic_ai` Agent with `output_type=IdentifiedAnimal`, directly testing if the model can return a structured Pydantic object instead of a raw string.

-   **To-Do / Next Steps:**
    1.  The user will prepare the `cat.jpg` and `dog.jpg` image files in the `illustrations` directory.
    2.  The user will execute the Phase 1 test: `python src/core/vl_models/vision_sanity_check.py`.
    3.  If Phase 1 is successful, the user will execute the Phase 2 test: `python src.core/vl_models/tool_calling_poc.py`.
    4.  The results of these experiments will determine the final implementation strategy for the puzzle extraction feature.



## 2025-10-22 (Second Entry)

### VL Model Debugging and Strategy Pivot

Conducted a deep debugging session on the Ollama-based Vision-Language model integration (Strategy B) and established a new, phased experimental plan.

-   **Initial State:** The test script (`run_pydantic_ai_test.py`) was failing with various errors, preventing successful communication with the VL model.

-   **Debugging Journey & Discoveries:**
    1.  **`ImportError` Resolution:** A series of `ImportError` and `NameError` issues were traced back to version differences in the `pydantic_ai` library. By inspecting the locally installed package files, the correct import paths and class names (`OpenAIChatModel`, `OllamaProvider`) were identified and fixed.
    2.  **Networking `404` Error:** A `404 Not Found` error was diagnosed as a mismatch between the Docker-internal hostname (`ollama_server`) defined in the `.env` file and the required `localhost` for scripts run from the host machine. The test script was updated to explicitly use `http://localhost:11434/v1`.
    3.  **Pydantic Validation Error:** A `ValidationError` for `extra_forbidden` was resolved by configuring the `Settings` class in `src/settings.py` to ignore extra fields from the `.env` file (e.g., `svelte_port`).
    4.  **`does not support tools` Error:** The final and most critical error was a `400 Bad Request` from the Ollama server, explicitly stating that the model (`qwen2.5vl:7b`) does not support the "tool-calling" feature. This is the core mechanism `pydantic_ai` uses for structured JSON output.

-   **Analysis of External Resources:** Based on user-provided research, it was confirmed that:
    *   Instruction-tuned model variants (e.g., `...-Instruct`) are critical for complex tasks.
    *   A community-provided model on Ollama Hub (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`) explicitly claims to support tool-calling.

-   **Revised Strategy & Next Steps:**
    1.  **Pause on "Tool-Calling":** Per user instruction, the current "manual JSON parsing" implementation in `vl_extractor.py` will be kept as a baseline. The more advanced tool-calling implementation is deferred.
    2.  **New Model Preparation:** The immediate next step is for the user to prepare the new, more capable model (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`) in their Ollama instance.
    3.  **Sanity Check:** A new test script (`vision_sanity_check.py`) will be created to perform a basic vision test (e.g., identifying a cat/dog) using the new model. This validates the model's core visual processing before attempting complex extraction.
    4.  **Proof of Concept:** A separate script (`tool_calling_poc.py`) will be created to demonstrate and validate the "tool-calling" capability of the new model in isolation.


## 2025-10-22

### Vision-Language Model Integration Strategy

Analyzed the new requirement to parse puzzles from uploaded images using a Vision-Language (VL) model. Two parallel implementation strategies were identified in the existing codebase (`src/core/vl_models/`).

-   **Strategy A: Integrated Hugging Face Transformers (`hf_parser.py`)**
    -   **Architecture:** Loads and runs a VL model (e.g., `Qwen/Qwen3-VL-4B-Thinking`) directly within the main application process using the `transformers` library.
    -   **Pros:** Self-contained, simplifies the end-to-end testing of the core extraction logic. The existing script appears more mature and includes a runnable test block.
    -   **Cons:** Tightly couples the main application with the resource-intensive VL model, potentially leading to high memory (VRAM) consumption.

-   **Strategy B: Microservice with Ollama (`vl_extractor.py`, `docker-compose.yml`)**
    -   **Architecture:** Defines a separate `ollama` service in Docker Compose to host the VL model. The main application communicates with it via an API, using `pydantic_ai` as a client.
    -   **Pros:** Superior service-oriented design. Decouples the VL model from the main application, improving scalability and reducing the main application's resource footprint. This is the preferred final architecture.
    -   **Cons:** The current implementation is more preliminary and introduces the complexity of inter-service communication and dependency on an external service.

-   **Identified Issues & Decisions:**
    -   A key inconsistency was found: the term for walls is `walls` in `hf_parser.py` but `blocked_cells` in other files. This must be standardized to `walls` to match the existing solver framework.
    -   **Decision:** The development will proceed in a phased approach. First, **Strategy A** will be completed to quickly deliver a functional end-to-end feature. Subsequently, this implementation can be refactored to follow the more robust **Strategy B** microservice architecture.

### Next Steps

-   Proceed with completing Strategy A (`hf_parser.py`).
-   Standardize all data structures and prompts in the `vl_models` directory to use the `walls` keyword and the `WallPair` Pydantic model for consistency.
-   Develop a standalone test script to validate the image-to-dictionary conversion before API and UI integration.



## 2025-10-21

### Dockerized Development Workflow Automation

To streamline the development process and simplify the startup of the containerized environment, this commit introduces a new automation script and enhances the project's containerization strategy.

-   **Docker Compose Enhancement**:
    -   The `docker-compose.yml` file was updated to define a complete, multi-service development environment, including the FastAPI backend (`zip-challenge-app`) and the Svelte frontend (`svelte-frontend`).
    -   Configuration was refined to ensure proper volume mounting for live code reloading and inter-container communication.

-   **Automated Startup Script**:
    -   Created a new Python script, `run_docker_dev.py`, to provide a one-command solution for launching the entire development stack.
    -   The script automates the following sequence:
        1.  Stops and removes any existing containers (`docker compose down`).
        2.  Builds fresh images and starts all services in the background (`docker compose up --build -d`).
        3.  Waits briefly for the main application container to initialize.
        4.  Executes the command to start the FastAPI server inside the running container, ensuring the virtual environment is activated.
    -   This script eliminates the need for manual `docker exec` commands and simplifies the developer onboarding experience.

-   **Documentation Update**:
    -   Updated the `README.md` and `README_zh-TW.md` files with a new "Running with Docker" section, explaining how to use the `run_docker_dev.py` script.
    -   This ensures that the documentation is synchronized with the latest, most efficient development workflow.

## 2025-10-20 (another commit)

### Svelte Frontend UX and Test Suite Refinements

This commit enhances the Svelte frontend's user experience and ensures the stability of the existing test suite.

-   **Svelte UI Enhancements**:
    -   **In-place Cell Editing**: The cell editing UX was significantly improved by replacing the browser's default `prompt()` dialog. A new, dynamic in-place editing mechanism was implemented. Now, clicking a cell overlays an `<input>` element directly onto the canvas grid, allowing for a more seamless and intuitive editing workflow.
-   **Test Suite Maintenance**:
    -   **Gradio Test Fix**: Corrected a failing test case in `test_gradio_app.py`. The assertion was updated to correctly handle the HTML-formatted error messages now returned by the Gradio UI, bringing the test suite back to a passing state.

## 2025-10-20

### Gradio UI Overhaul and Interactive Solver Implementation

This phase focused on building a highly interactive and user-friendly puzzle editor within the Gradio web UI, moving from a text-based input to a full "What You See Is What You Get" (WYSIWYG) experience.

-   **Interactive Puzzle Editor ("V2")**:
    -   Replaced the initial text-based "naive" solver tab with a new "Interactive" tab.
    -   Implemented a dynamic grid creation system where users can specify puzzle dimensions (`m x n`).
    -   **Refactored Wall Editor**: Based on user feedback regarding the initial confusing checkbox-based UI, the wall editor was completely redesigned.
        -   Users now input wall coordinates using four simple number boxes (`r1, c1, r2, c2`).
        -   A list view displays all current walls, with a proper "select-then-click" button to delete walls.
    -   **Live Image Preview**: Added a new preview panel that generates and displays an image of the puzzle in real-time. The preview automatically updates whenever the user edits the puzzle grid (adding numbers/obstacles) or modifies the wall list.
    -   **New UI Controls**: Implemented a "Reset" button to clear all interactive components to their default state.

-   **Debugging and Stability**:
    -   **Extensive Bug Fixing**: Resolved a long series of bugs discovered during iterative development, including `IndentationError`, `NameError`, `AttributeError`, `UnboundLocalError`, and several data format mismatches between the frontend and backend (e.g., `'x'` vs `'xx'`, `dict` vs `set`).
    -   **Enhanced Logging**: Added detailed `loguru` logging to both the frontend (`gradio_app.py`) and backend (`solver.py`). These logs capture the raw UI payload and the parsed puzzle data, which was critical in diagnosing the data flow issues. Also added logging for temporary file deletion in the backend.
    -   **Code Maintenance**: Fixed a `FutureWarning` from the `pandas` library by migrating from the deprecated `Styler.applymap` to `Styler.map`.

-   **Architectural Refinements**:
    -   The frontend `gradio_app.py` was refactored multiple times to serve as a robust "Adapter", translating intuitive user actions into the precise data formats expected by the backend API.
    -   The core backend logic in `utils.py` and `solver.py` was validated and corrected to ensure it properly handles obstacles and other puzzle constraints.

## 2025-10-16

### Implementation of Service-Oriented Architecture (Phase 1)

Following the pivot from pure algorithmic development, the first phase of the user-facing web service has been implemented. This phase establishes the core architecture and a functional user interface.

-   **Web Service Backend (FastAPI):**
    -   Initialized a FastAPI application (`src/app/main.py`) to serve as the backend.
    -   Implemented a robust, layered configuration system using `pydantic-settings` (`src/settings.py`) that reads from a `.env` file, making settings like port numbers easily configurable.
    -   Refactored the API structure into a scalable `routers` and `schemas` pattern. All API endpoints are now modularly organized (e.g., `src/app/routers/echo.py`, `src/app/routers/solver.py`).
    -   Created a `/api/solver/solve` endpoint that receives puzzle data, calls the appropriate core solver, and returns a JSON response containing the solution path and Base64-encoded images.
    -   Improved code quality by replacing magic numbers for HTTP status codes with `fastapi.status` constants.

-   **Web User Interface (Gradio):**
    -   Developed a multi-tab Gradio interface (`src/ui/gradio_app.py`) for user interaction.
    -   The UI is mounted directly within the FastAPI application, creating a single, unified service.
    -   Implemented a "Puzzle Solver naive version" tab that allows users to paste puzzle layouts and walls, select a solver, and receive a visual solution.
    -   The UI now displays both an animated GIF of the solution process and a static image of the final result.

-   **Visualization Enhancements:**
    -   Created a new `save_detailed_animation_as_gif` function in `utils.py` to generate GIFs with enhanced visuals, including a highlighted path head (blue) and sequential step numbers (green).
    -   Added a `save_solution_as_image` function to generate a static PNG of the final solved puzzle.
    -   The backend now uses these new functions to provide richer visual feedback to the user.

-   **Bug Fixes & Refinements:**
    -   Standardized file path comments in `.py` files to use forward slashes (`/`) for cross-platform consistency.
    -   Resolved a `PermissionError` on Windows related to `tempfile` by implementing a more robust file handling pattern in the solver API.
    -   Corrected multiple `IndentationError` syntax issues that arose during refactoring.
    -   Standardized type hint styles in Pydantic schemas to the modern `|` union operator as per project conventions.

### Quality Assurance and Refactoring

-   **Unit Test Implementation**: Added a comprehensive suite of unit tests for the new service-oriented architecture. This includes API endpoint tests using `TestClient` (`src/app/tests/`), UI logic tests using `unittest.mock` (`src/ui/tests/`), and smoke tests for new visualization utilities in `src/core/tests/`.
-   **Project Structure Refactoring**: To improve modularity, moved `puzzle_generator.py` and `generate_dataset.py` into a new dedicated `src/core/puzzle_generation/` directory and updated all corresponding import paths across the project.
-   **Performance Tuning**: Modified the `generate_dataset.py` script to limit the multiprocessing pool to 75% of available CPU cores, ensuring system responsiveness during heavy computation.

### Next Steps

-   **Interactive UI**: Implement the "Puzzle Solver interact version" tab in the Gradio UI.
-   **Containerization**: Introduce a `Dockerfile` to allow the entire web service to be built and run as a container.

## 2025-10-15

### Reinforcement Learning Development Paused

Due to the inherent challenges in reward function design and overall training complexity, the Reinforcement Learning (RL) development effort is being temporarily paused.

Future work in this area will be resumed after a period of deeper research into advanced RL concepts and architectures. The planned areas of study include:
-   Architectures of seminal models like **AlphaGo** and **AlphaZero**.
-   Reviewing the hands-on examples in the local `more_simple_reinforcement_learning` directory.
-   Studying the "Hands-on Reinforcement Learning" course materials (from `hrl.boyuai.com`).

When RL development resumes, a revised approach will be considered to simplify the problem, such as:
-   Reducing the `map_size` to a smaller dimension.
-   Relaxing the environment's constraints (e.g., allowing the agent to revisit paths, transforming the problem from finding a single Hamiltonian path to a more flexible pathfinding task).

### Project Pivot to Service-Oriented Architecture

The project's immediate focus will shift from algorithmic development to building a user-facing service. The goal is to create an application with a UI that allows users to upload their own puzzles and receive a computed solution.

### New To-Do List

-   **Service Backend:** Implement a web backend using **FastAPI**.
-   **User Interface:** Create an interactive web UI with **Gradio**.
-   **Future Exploration:** Investigate the integration of **MCP (Model-View-Controller Pattern)** and **multi-modal** capabilities.

### Archived Progress 

*This section documents the last active development goal before the pivot.*

The previous focus was on attempting to solve a 6x6 map using an RL approach. The strategy was to first test and solve the problem on a **single map** (i.e., achieve overfitting) as a proof of concept. The successful completion of this step would then serve as a foundation for the ultimate goal of **generalizing** the solution to arbitrary 6x6 maps. The starting point for this development was the implementation of the `src/core/rl/train_single_sb.py` script.

## 2025-10-13

### Deep Dive into Deterministic Loop & Reward Shaping

Following the successful overfitting of the MLP-based model during training and its subsequent failure in deterministic evaluation, a series of experiments were conducted to resolve the underlying "deterministic policy loop" issue with a new CNN-based model.

-   **Problem Persistence & State Representation Fix**: Despite refactoring the environment to use a 6-channel image-like state representation (including separate layers for walls and obstacles) and switching to a `CnnPolicy`, the agent continued to fail during deterministic evaluation. It achieved high rewards during training (with exploration) but fell into inescapable loops when `deterministic=True`.

-   **Hypothesis 1: Insufficient Penalty for Inefficiency.** The first hypothesis was that the `-1.0` time penalty was not enough to discourage looping.
    -   **Experiment:** A "soft constraint" was added to the reward function in `rl_env.py`, applying a `-2.0` penalty for revisiting any cell already in the `path_taken`.
    -   **Result:** **Failure.** The evaluation log (`evaluation_path_2025-10-13_13-37-36.log`) showed that while the agent explored more territory, it ultimately still fell into a tight loop (`(4, 0) <-> (5, 0)`), indicating the revisit penalty was not sufficient to overcome the root cause.

-   **Hypothesis 2: Dense Reward Traps.** The primary suspect shifted to the distance-based reward shaping (`(dist_before - dist_after) * weight`), which could be creating local optima ("reward traps") that are more attractive than exploring a path to the true goal.
    -   **Experiment:** The reward shaping weight was reduced by an order of magnitude, from `0.1` to `0.01`. The parameter was also refactored into the `PuzzleEnv` constructor and the training script's `CONFIG` for easier tuning.
    -   **Result:** **Failure.** The evaluation log (`evaluation_path_2025-10-13_14-05-36.log`) again showed the agent getting stuck in a terminal loop, proving that even a very small positive incentive towards the goal can create a powerful enough trap to derail the deterministic policy.

-   **Final Diagnosis:** The distance-based reward shaping, even with a minimal weight, is fundamentally at odds with the sparse penalty system. It encourages a "greedy" local-optimization behavior that results in policy loops. The agent is unwilling to incur a small penalty (by moving away from the target) to find a path around an obstacle, as the dense reward signal is too dominant.

### To-Do List

-   **[Next Step]** Completely eliminate the dense reward signal by setting `DISTANCE_REWARD_WEIGHT` to `0` in `train_single_cnn_sb.py`.
-   Re-train the model from scratch using the purely sparse reward function (only step/revisit/invalid penalties and waypoint/goal rewards).
-   Perform a deterministic evaluation on the new model to verify if the looping issue is finally resolved.
-   If the issue persists, the final recourse is to escalate the "soft constraint" on revisits to a "hard constraint" by making it an invalid move.

## 2025-10-12

### RL Agent Deep Debugging and Analysis

A deep-dive debugging session was conducted to diagnose why the DQN agent, despite successful training metrics, failed during deterministic evaluation.

-   **Initial State & Problem:** The agent, whether custom-built or using `stable-baselines3`, showed high average rewards during training but consistently failed to complete a puzzle during deterministic evaluation (`epsilon=0`), always timing out at the maximum step limit.

-   **Hypothesis 1: Insufficient Evaluation Steps.** The initial hypothesis was that the evaluation loop's step limit was too low. This was proven false, as increasing the limit in the evaluation script had no effect. The root cause was identified as a hardcoded `_max_steps` limit within the `PuzzleEnv` itself.

-   **Hypothesis 2: Flawed Reward Shaping.** The second hypothesis was that the distance-based reward shaping (`(dist_before - dist_after) * 1.0`) was creating a "reward trap" or local optimum, causing the agent to loop near the goal. An experiment was conducted by reducing the shaping weight to `0.1`. While this produced even better training metrics, the deterministic evaluation still failed in the exact same manner.

-   **Final Diagnosis: Deterministic Policy Loop.** The conclusive diagnosis is that the agent's learned deterministic policy contains an inescapable loop. The successful, shorter-episode training runs were an illusion created by random exploration (`epsilon > 0`) accidentally "bumping" the agent out of its learned loop, allowing it to reach the goal. When this randomness is removed, the policy's fatal flaw is revealed.

-   **Framework Enhancement:** To facilitate debugging, the `PuzzleEnv` was refactored to allow its `max_steps` limit to be configured externally during instantiation. The evaluation scripts (`evaluate_sb.py`) were updated to use this new parameter, providing a more flexible testing environment.

### Reinforcement Learning Framework Q&A

A summary of the RL agent's core mechanics was documented to clarify understanding.

-   **Q1: What are the agent's movement rules?**
    -   The agent has a discrete action space (Up, Down, Left, Right). It is permitted to reverse its direction and revisit cells it has previously occupied. There are no rules preventing revisits.

-   **Q2: What is the agent's goal and behavior?**
    -   **Goal:** To navigate from a starting position, visiting a sequence of numbered waypoints in the correct order, and finally arriving at the last waypoint.
    -   **Behavior:** The agent's behavior is governed by a policy network (an MLP). This network takes the current state (`agent_location`, `next_waypoint_location`) and outputs Q-values for each of the four actions. The agent selects the action with the highest Q-value, which it predicts will lead to the maximum cumulative future reward.

-   **Q3: How does the agent interact with the environment?**
    -   The interaction follows the standard RL loop. The agent submits an `action` to the environment via `env.step(action)`. The environment transitions to a `next_state` and returns a `reward`, a `terminated` flag (for goal completion), a `truncated` flag (for timeouts), and an `info` dictionary. The agent uses this feedback to update its policy.

-   **Q4: What is the reward function?**
    -   The reward function is composed of several components:
        -   `+1000.0` for reaching the final waypoint.
        -   `+200.0` for reaching an intermediate waypoint.
        -   `-10.0` for an invalid move (hitting a wall, obstacle, or boundary).
        -   `-1.0` as a time penalty for every step taken.
        -   `(dist_before - dist_after) * 0.1` as a small, dense reward for reducing the Manhattan distance to the next target.

-   **Q5: What logging is available besides the GIF animation?**
    -   **Console Logs:** Real-time statistical tables from `stable-baselines3` during training.
    -   **File Logs:** Detailed, timestamped logs saved by `loguru` to the `logs/` directory.
    -   **TensorBoard Logs:** The most powerful tool. Detailed, interactive graphs of all training metrics (reward, loss, etc.) are saved to `logs/sb_tensorboard/`. This can be launched via the command `tensorboard --logdir ./logs/sb_tensorboard/`.

### To-Do List

-   Review the visual `evaluation_sb.gif` and TensorBoard logs to pinpoint the exact location and pattern of the agent's deterministic loop.
-   Based on the loop's characteristics, redesign the reward function to specifically penalize or disincentivize the observed looping behavior.
-   If reward redesign is insufficient, consider redesigning the environment's rules of interaction (e.g., adding a penalty for immediately revisiting the previous state).

## 2025-10-12

### Reinforcement Learning (RL) Solver Framework

-   **Architectural Design**: Designed a complete framework to solve puzzles using Deep Reinforcement Learning. The approach is based on a DQN (Deep Q-Network) agent interacting with a custom environment, with a focus on making the training pipeline robust and reproducible.
-   **Custom RL Environment (`rl_env.py`)**: Implemented a `gymnasium.Env`-compatible environment, `PuzzleEnv`, to wrap the puzzle logic.
    -   Features a sophisticated **reward shaping** mechanism to provide dense rewards, guiding the agent by calculating the change in Manhattan distance to the next waypoint.
    -   The state space is defined by the agent's location and the next target waypoint, making the problem tractable for a neural network.
-   **DQN Agent (`dqn_agent.py`)**: Implemented a complete DQN agent, including:
    -   A `DQNModel` (MLP) to approximate the Q-function.
    -   A `ReplayBuffer` for experience storage and sampling.
    -   The core `DQNAgent` class encapsulating the learning logic, epsilon-greedy action selection, and target network updates.
-   **Two-Stage Training Pipeline**: Decoupled data generation from training for better workflow and reproducibility.
    -   **Dataset Generation (`generate_rl_dataset.py`)**: Created a multiprocessing-enabled script to generate and save large puzzle datasets (`6x6` and `7x7`). It outputs both a human-readable log for verification and a `pickle` file for the trainer to consume.
    -   **Training Script (`train.py`)**: Developed the main training script that loads the pre-generated dataset, manages the training loop, logs progress with `tqdm` and `loguru`, and saves the final trained model.

### Code Quality and Bug Fixes

-   **Pathing Logic**: Corrected a path calculation error in `generate_rl_dataset.py` and `train.py` that resulted in an incorrect, duplicated output directory path. The logic for determining the project root was made more robust.
-   **Linter Compliance**: Resolved a `SyntaxError` reported by `ruff` in `dqn_agent.py` by refactoring a multi-line expression to be more robust, ensuring the codebase passes all `pre-commit` checks.
-   **Dependency Management**: Identified and added necessary dependencies (`gymnasium`, `torch`, `tqdm`) for the new RL framework, using the project's `uv add` workflow.

## 2025-10-08

### Puzzle Generation Framework

-   **Procedural Puzzle Generator:** Created a new, sophisticated puzzle generation module (`src/core/puzzle_generator.py`).
    -   The core logic is built upon a **randomized backtracking algorithm** (`_generate_hamiltonian_path`) that generates a guaranteed valid solution path covering all visitable cells.
    -   Introduced a robust generation process with a **retry and decrement** mechanism: if generating a puzzle with `N` obstacles fails, it automatically retries, and if still unsuccessful, it gracefully degrades to attempt generation with `N-1` obstacles.
    -   Implemented a true **internal timeout** within the pathfinding algorithm to terminate and abandon attempts that take too long, preventing the process from hanging and saving CPU resources.
-   **Automated Dataset Creation Script:** Developed a powerful script (`src/core/generate_dataset.py`) to automate the creation of large puzzle datasets.
    -   Leverages the `multiprocessing` module to generate multiple puzzles in **parallel**, significantly speeding up the process.
    -   The script is highly configurable and creates a clean, **timestamped directory structure** for each run, organizing the generated puzzle data (`puzzles.py`) and GIF animations (`gifs/`) separately.
    -   Waypoint count is now **dynamically calculated** based on puzzle size (1/4 to 1/3 of path length) to create more balanced puzzles.

### Code Quality and Refactoring

-   **Improved Type Safety:** Introduced a `Puzzle` `TypedDict` in `utils.py` to provide a strict data contract for puzzle objects, replacing generic dictionaries and improving type safety across the codebase. All relevant functions (`puzzle_generator`, `utils`, etc.) were updated to use this precise type.
-   **DRY Principle Refactoring:** Refactored `puzzle_generator.py` to call the canonical `parse_puzzle_layout` function instead of manually re-implementing the puzzle object construction logic.
-   **Code Style and Conventions:**
    -   Standardized all new modules to use the `pathlib` library for path manipulations, adhering to project conventions.
    -   Updated all new modules to use absolute imports (e.g., `from src.core...`) as per user preference.
    -   Eliminated all "magic numbers" by defining them as named constants at the top of modules (e.g., `MAX_RETRIES_PER_COUNT`).
    -   Updated `gemini_readme_raw.md` to formally document the `pathlib` and absolute import style rules.
-   **Bug Fixes and Linting:**
    -   Fixed a critical `NameError` bug in `puzzle_generator.py` where `logger` was used but not imported.
    -   Fixed a `NameError` in the `generate_dataset.py` multiprocessing worker where `logger` was not available in the child process scope.
    *   Fixed a visual bug in `save_animation_as_gif` where `blocked_cells` were not being rendered; they are now correctly drawn as black squares.
    -   Resolved multiple `ruff` linter errors (`F841`: unused variable) in `utils.py`.

### Testing

-   **Generator Test Suite:** Created a new test file `src/core/tests/test_puzzle_generator.py`.
    -   Added a comprehensive **smoke test** (`test_generate_puzzle_smoke`) that validates the integrity of a complex generated puzzle (with walls and obstacles) and its solution.
    -   Added a dedicated test (`test_generate_puzzle_default_waypoints`) to verify the new **dynamic default waypoint calculation** logic.
-   **Standardized Test Output:** Replaced all `print()` statements in the new test file with `logger` calls to maintain consistency with project standards.

## 2025-10-04

### Expansion of Metaheuristic Solver Suite

-   **Simulated Annealing (SA) Solver:** Implemented `solve_puzzle_simulated_annealing` in a new `simulated_annealing.py` module. The development process uncovered a critical bug in the initial neighbor generation logic:
    -   An initial `2-opt` swap strategy, common in TSP-like problems, was found to produce non-contiguous paths (i.e., "jumps") on a grid. This bug was identified thanks to user feedback.
    -   The logic was corrected by replacing `2-opt` with a robust "truncate and regrow" strategy in the `_generate_neighbor_path` helper function, which guarantees path contiguity.
-   **Genetic Algorithm (GA) Solver:** Implemented `solve_puzzle_genetic_algorithm` in `genetic_algorithm.py`.
    -   To avoid the path contiguity issues inherent in traditional crossover operations, a pragmatic "no-crossover" variant was designed. 
    -   The implemented GA relies on elitism (carrying over the best solutions) and mutation (using the new `generate_neighbor_path` function) for reproduction and population evolution.
-   **Tabu Search (TS) Solver:** Implemented `solve_puzzle_tabu_search` in `tabu_search.py`.
    -   The solver uses a `collections.deque` with a fixed `maxlen` as an efficient short-term memory (the "tabu list").
    -   To save memory, hashes of path tuples (`hash(tuple(path))`) are stored in the tabu list instead of the paths themselves.
    -   An aspiration criterion is included. The logic for this criterion was significantly refined based on user feedback:
        -   A critical logical flaw in the initial implementation (`score > best_score`), where the condition would never be met for a tabu item, was identified by the user.
        -   The final, more flexible and effective implementation (`score >= aspiration_threshold * best_score`) was also proposed by the user, and the `aspiration_threshold` parameter was added accordingly.
-   **Particle Swarm Optimization (PSO) Solver:** Implemented a discrete adaptation of PSO in `particle_swarm_optimization.py`.
    -   A particle's "position" is defined as a path, and its "velocity" is defined as a list of swap operations.
    -   Discrete analogues for velocity and position updates were implemented. This approach relies on the fitness function's heavy penalty for non-contiguous "jumps" to guide the swarm toward valid paths.
    -   During a detailed review, the user correctly pointed out that the sequential application of swap operations (the "velocity") causes "distortion," as the effect of a later swap is dependent on the state change from an earlier swap. It was clarified that this is an accepted and inherent characteristic of this discrete PSO adaptation, providing a form of stochastic perturbation that aids in exploration, with the fitness function acting as the ultimate arbiter of path quality.

### Major Refactoring and Code Quality Enhancements

-   **Centralized Path Utilities:** To eliminate code duplication across solvers, the common helper functions `generate_random_path` and `generate_neighbor_path` were moved from individual solver files into the shared `src/core/utils.py` module. `monte_carlo.py` and `simulated_annealing.py` were refactored to use these new shared utilities.
-   **Fitness Function Hardening:** The `calculate_fitness_score` function in `utils.py` was made more robust. A Manhattan distance check was added to penalize non-contiguous path "jumps", which was a weakness identified during the SA implementation.
-   **Increased Test Coverage:** 
    -   Added smoke tests for all new metaheuristic solvers (SA, GA, TS, PSO) to ensure they run and produce correctly formatted output.
    -   Added new, dedicated unit tests to `test_utils.py` for the shared `generate_random_path` and `generate_neighbor_path` functions to validate their core logic (e.g., path contiguity, no duplicates, correct start point).
-   **Code Style and Linting:** Fixed several `pre-commit` errors reported by `ruff`, including an `F821 Undefined name` error from a missing `import` and an `E402 Module level import not at top of file` style violation.

## 2025-10-01

### Codebase Modernization and Toolchain Overhaul

-   **Path Handling Refactoring:** Replaced all instances of `os.path` with the modern `pathlib` library across the test suite (`conftest.py`, `test_dfs.py`). This improves path manipulation logic, making it more readable, consistent, and object-oriented.
-   **Alternative A* Solver Implementation:** Implemented a new A* solver variant, `solve_puzzle_a_star_sortedlist`, which leverages `sortedcontainers.SortedList` as its priority queue instead of the standard `heapq`. A corresponding parametrized unit test was added to `test_a_star.py` to ensure its correctness against the full puzzle suite.
-   **Pre-Commit and CI/CD Pipeline Refinement:**
    -   **Test Pathing Resolution:** Resolved a critical `ModuleNotFoundError` during test collection by migrating the Python path configuration from a `sys.path` manipulation in `conftest.py` to a centralized `pythonpath` setting in `pytest.ini`. This aligns with `pytest` best practices.
    -   **Toolchain Consolidation:** Diagnosed and fixed a persistent formatting conflict loop between `black`, `isort`, and `ruff`. The pre-commit configuration was completely refactored to use `ruff` exclusively for all linting, import sorting, and code formatting, removing `isort` and `black` for a faster and simpler CI pipeline.

### Metaheuristic Search Framework and Baseline Implementation

-   **Fitness Function Design & Implementation:**
    -   Designed and implemented a comprehensive `calculate_fitness_score` function in `utils.py`. This function establishes the core evaluation metric for all metaheuristic solvers, incorporating a system of penalties and rewards (for path length, waypoint sequencing, etc.).
    -   The function was enhanced to return both the path's current score and the puzzle's theoretical perfect score, providing a clear benchmark for solution quality.
    -   Added a full suite of unit tests in `test_utils.py` to validate the fitness function's behavior.

-   **Monte Carlo Solver:**
    -   Implemented the first metaheuristic solver, `solve_puzzle_monte_carlo`, as a baseline for performance comparison. The solver generates a specified number of random paths and returns the one with the highest fitness score.
    -   The solver's logging was integrated with the new fitness function output to display comparative scores (e.g., `Best score: 420200/1720360`).
    -   A unit test was created to verify the integrity of the Monte Carlo solver, ensuring it produces valid paths.

### Code Quality and Refactoring

-   **DRY Principle Refactoring:** Refactored all existing exact solvers (`dfs.py`, `a_star.py`, `cp.py`) to consume the `num_map` from the puzzle dictionary, eliminating redundant code.
-   **Bug Fixes:** Diagnosed and resolved multiple `NameError` exceptions in `a_star.py` and `test_utils.py` that were introduced during refactoring, ensuring the entire test suite passes.

## 2025-09-25

### Advanced Solver Implementation and Analysis

-   **A* Solver:** Implemented a complete A* solver (`a_star.py`) using a priority queue (`heapq`) and a Manhattan distance heuristic. Iteratively debugged the implementation, correcting a critical flaw in the `closed_set` logic to ensure proper state tracking, which resulted in all test cases passing.
-   **CP-SAT Solver:** Developed a solver using Google's OR-Tools (`cp.py`). Modeled the puzzle as a Constraint Satisfaction Problem, and after multiple iterations, resolved an `INFEASIBLE` status by re-modeling the problem. The final, successful implementation uses the "dummy node" technique to correctly represent a Hamiltonian path with an `AddCircuit` constraint.
-   **Algorithm Analysis:** Performed a detailed theoretical analysis of the Time and Space Complexity (TC/SC) for the DFS, A*, and CP-SAT solvers. Compared their trade-offs in terms of memory usage, practical speed, and implementation paradigm.

### Major Project Structure Refactoring

-   Relocated all solver implementations (`dfs.py`, `a_star.py`, `cp.py`) into a new, dedicated `src/core/solvers/` directory to improve modularity and separation of concerns.
-   Mirrored the source code structure within the test directory by creating `src/core/tests/solvers/` and moving the corresponding test files. This refactoring enhances test organization and future scalability.
-   Updated all relevant `import` statements across the test suite to reflect the new file locations, ensuring all 19 tests pass after the refactoring.

### To-Do List

-   **Metaheuristic Solvers:** Begin implementation of non-deterministic, metaheuristic algorithms.
    -   Define a robust **fitness/cost function** to score partial or imperfect solutions.
    -   Implement a baseline **Monte Carlo (Random Sampling) Search**.
    -   Implement other metaheuristics such as **Simulated Annealing**, **Genetic Algorithm**, or **Ant Colony Optimization**.
    -   All metaheuristic solvers should accept an `attempts` parameter to control the number of iterations.

## 2025-09-23

### Solver Verification and Visualization Overhaul

-   **DFS Solver Logic Verified:** Through a process of debugging and adding detailed logging, it was determined that the core DFS solver algorithm was logically correct. The previously observed test failures were traced back to incorrect reference solutions in the test data.
-   **Test Data Corrected:** Fixed typos in the ground-truth data within `conftest.py`, leading to all 9 unit tests passing and validating the solver's correctness.
-   **Advanced Visualization Implemented:** Iteratively redesigned and implemented multiple solution-visualization features in `utils.py` based on interactive feedback:
    -   Implemented two distinct console-based styles: a simple `[bracket]` highlighter and a more advanced ANSI background-color highlighter.
    -   Added console-based animation functions (`animate_solution_*`) to display the step-by-step pathfinding process, addressing the need to show path order.
    -   To handle layout "wobbling" during animation, the printing logic was refactored to pre-calculate and enforce a fixed grid size across all animation frames.
-   **GIF Animation Generation:** Implemented a new feature, `save_animation_as_gif`, using the Pillow library to generate and save high-quality, shareable GIF animations of puzzle solutions, complete with wall rendering.

## 2025-09-22

### Input System Refactoring and Test Data Integration

-   **Input Refactoring:** Overhauled the puzzle input system. Puzzles are now defined with a readable, text-based `puzzle_layout`, which is then processed by a dedicated `parser` in `utils.py`.
-   **Utility Functions:** Created `src/core/utils.py` to house shared functions, including the new `parse_puzzle_layout` parser and a `visualize_solution` function for displaying results.
-   **Test Data Enhancement:** Integrated the user-provided, ground-truth solutions for all six puzzles (`puzzle_01` to `puzzle_06`) into the `conftest.py` test suite, enabling strict path verification.

### To-Do List

-   **Unit Testing:** Write and pass unit tests for the new utility functions in `src/core/utils.py`.
-   **Algorithm Validation:** Run the full test suite to verify the DFS solver's correctness against all 6 ground-truth solutions.
-   **Debugging:** Based on test results, debug any discrepancies between the solver's output and the expected solutions.
-   **Visualization Polish:** Re-evaluate and possibly redesign the presentation of the visualized solution for better clarity during debugging.

## 2025-09-21

### Test Suite and Architecture Overhaul

-   **Test Case Expansion:** Transcribed and added puzzles 01 through 06 from image files into the test suite.
-   **Test Architecture Refactoring:** Refactored the entire test workflow to be scalable and reusable. Test data is now centralized in `conftest.py` and dynamically loaded into a single test function in `test_dfs.py` using `pytest.parametrize`.
-   **Input Refactoring:** Enhanced the core solver and input data structure to support "blocked cells" in addition to "walls", making the algorithm more versatile.
-   **Workflow Update:** Updated the internal Gemini README to define collaboration rules regarding package management and test execution.

## 2025-09-20

### Core Solver Implementation

-   Initialized the project structure.
-   Implemented the core puzzle-solving logic in `src/core/dfs.py` using a backtracking Depth-First Search (DFS) algorithm.
-   The solver handles grids with numbered waypoints and walls that blocking paths.

### Testing and Reporting Setup

-   Introduced `pytest` as the testing framework.
-   Created a test suite in `src/core/tests/test_dfs.py` with multiple test cases, including simple solvable puzzles, puzzles with walls, and puzzles designed to be unsolvable.
-   Iteratively refined the "unsolvable" test cases after discovering the solver was more robust than initially anticipated.

### Automation and Workflow Refinement

-   Set up `loguru` to provide detailed, professional-grade logging for test execution.
-   Configured the logger to output to timestamped files (`log_[timestamp].log`) with UTC timestamps in the filename and local timezone information in the log messages.
-   Engineered a system to automatically generate test reports that mirror the console output.
-   After exploring `pytest.ini` and `conftest.py` hooks, finalized the reporting mechanism using a `run_tests.bat` script for maximum reliability and platform consistency. This script redirects all console output to a timestamped `test_report_[timestamp].txt` file.
-   The final workflow is simplified to a single command: `.\run_tests.bat`
