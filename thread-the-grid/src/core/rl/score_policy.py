# src/core/rl/score_policy.py
"""The ruler the RL reports are measured with: deterministic solve rate plus best-of-N.

Run (from thread-the-grid/):
    # the served model on the frozen held-out set
    uv run python -m src.core.rl.score_policy score --run-id bc_multi_456_e6 --size 6 \
        --eval-set rl_eval_sets/seed20300000_n20000_456_test
    # an intermediate checkpoint needs a label, so no published artifact is overwritten
    uv run python -m src.core.rl.score_policy score --run-id bc_multi_456_sweep \
        --checkpoint model_epoch_4 --label strict_bc_multi_456_sweep_e4 --size 6 \
        --eval-set rl_eval_sets/seed20300000_n20000_456_test
    # deterministic only: --max-attempts 0
    # markdown rows for a report, read from the written artifacts
    uv run python -m src.core.rl.score_policy table bc_multi_456_e6 --size 6

Moved into version control on 2026-09-24 from the private probes that produced every
best-of-N number in `ai-collab/reports/` (`probe_best_of_n.py`, `probe_cross_size.py`, with
`eval_checkpoint_det.py` and `summarise_probes.py` folded in). The accounting is unchanged,
so what this writes is comparable with the published artifacts:

*   Each puzzle gets one deterministic episode (argmax), then up to `max_attempts` *sampled*
    episodes that stop at the first solve. The deterministic episode is not attempt 1 --
    that is where this differs from `solver_service`, which spends attempt 1 on the argmax.
*   Best-of-N for every N <= max comes off the same pass: solved at N iff the first success
    came at attempt <= N; the cost at N is min(first success, N), or N if never solved.
*   One numpy rng seeds every env reset, puzzle after puzzle, and torch's generator draws
    the sampled actions. Same checkpoint, puzzles, order, seed and device -> same numbers.

⚠ The judge is whatever `rl_env_v2._is_solved` says today, which is strict since 2026-09-19
(the walk has to end on the highest number). Artifacts written before then were scored
loose; this module cannot reproduce them, and records its judge so the two are not mixed.
"""

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from loguru import logger

from src.core.rl.baselines import (
    DATASET_ROOT,
    POLICIES,
    PolicyFn,
    load_split,
    make_eval_env,
    run_episode,
)
from src.core.rl.eval_set import MANIFEST_FILENAME, file_sha256, load_eval_set
from src.core.rl.rl_env_v2 import PuzzleEnvV2, PuzzleSample
from src.core.rl.train_config import ResourceSettings
from src.core.rl.train_maskable_ppo import (
    CHECKPOINT_DIRNAME,
    FINAL_CHECKPOINT_NAME,
    MODEL_ROOT,
    apply_resource_limits,
    model_policy,
)

ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "logs" / "rl_probes"
# Historical: the probe was first written to score a checkpoint on a size it never trained
# on. Every published best-of-N artifact carries this prefix, so new ones keep it.
ARTIFACT_PREFIX = "cross_size"
DEFAULT_SEED = 20260815
DEFAULT_MAX_ATTEMPTS = 32  # the judged standard is best-of-32 (2026-09-11)
REPORTED_N = (1, 2, 4, 8, 16, 32, 64)
TABLE_N = (1, 4, 16, 32)
JUDGE = "strict"


def attempts_until_solved(play: Callable[[], bool], max_attempts: int) -> int | None:
    """The 1-based attempt that first solved, or None. Nothing is played after a solve."""
    for attempt in range(1, max_attempts + 1):
        if play():
            return attempt
    return None


def best_of_n(
    first_success: Sequence[int | None],
    max_attempts: int,
    reported_n: Sequence[int] = REPORTED_N,
) -> dict[str, dict[str, float]]:
    """Solve rate and cost at every reported N the run's budget covers."""
    total = len(first_success)
    table: dict[str, dict[str, float]] = {}
    for n in reported_n:
        if n > max_attempts:
            continue
        hits = sum(1 for a in first_success if a is not None and a <= n)
        # Early stopping: a puzzle first solved at attempt a costs min(a, n) episodes at
        # this N, an unsolved one the whole budget -- what deploying at N would spend.
        cost = sum(min(a, n) if a is not None else n for a in first_success)
        table[str(n)] = {
            "solve_rate": hits / total,
            "episodes_per_puzzle": cost / total,
        }
    return table


def _solved(env: PuzzleEnvV2, policy: PolicyFn, rng: np.random.Generator) -> bool:
    return bool(run_episode(env, policy, rng)["solved"])


def score_samples(
    samples: Sequence[PuzzleSample],
    greedy: PolicyFn,
    sampled: PolicyFn,
    seed: int,
    max_attempts: int,
) -> tuple[int, list[int | None]]:
    """Deterministic solves, and per puzzle the attempt that first solved it (or None)."""
    rng = np.random.default_rng(seed)
    deterministic_solved = 0
    first_success: list[int | None] = []
    for sample in samples:
        env = make_eval_env(sample)
        deterministic_solved += _solved(env, greedy, rng)
        first_success.append(
            attempts_until_solved(partial(_solved, env, sampled, rng), max_attempts)
        )
    return deterministic_solved, first_success


def summarise_attempts(first_success: Sequence[int | None]) -> dict[str, Any]:
    solved = [a for a in first_success if a is not None]
    return {
        "median": statistics.median(solved) if solved else None,
        "mean": statistics.fmean(solved) if solved else None,
        # A hazard rate that decays to nothing says the unsolved puzzles are not being
        # missed by luck, so a larger N would not buy them.
        "histogram": {str(a): solved.count(a) for a in sorted(set(solved))},
    }


def load_samples(args: argparse.Namespace) -> tuple[list[PuzzleSample], dict[str, Any]]:
    """The puzzles to score, filtered to one size, plus where they came from."""
    if args.eval_set is not None:
        manifest, samples = load_eval_set(args.eval_set)
        source = {
            "dataset": manifest["dataset"],
            "split": manifest["split"],
            "eval_set": args.eval_set.name,
            "content_sha256": manifest["content_sha256"],
        }
    else:
        dataset_dir = args.dataset_root / args.dataset
        samples = load_split(dataset_dir, args.split)
        manifest = json.loads(
            (dataset_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        source = {
            "dataset": args.dataset,
            "split": args.split,
            "content_sha256": manifest.get("content_sha256", {}).get(args.split),
        }

    samples = [s for s in samples if s.puzzle["grid_size"][0] == args.size]
    if args.limit is not None:
        samples = samples[: args.limit]
    return samples, source


def score(args: argparse.Namespace) -> Path:
    samples, source = load_samples(args)
    label = args.label or args.run_id or args.baseline
    artifact = (
        args.output_dir
        / f"{ARTIFACT_PREFIX}_{label}_{args.size}x{args.size}_{source['split']}.json"
    )
    if artifact.exists() and not args.overwrite:
        raise SystemExit(f"{artifact} exists; pass --label or --overwrite.")

    apply_resource_limits(ResourceSettings())
    # Seeded before the model is built: constructing the network draws from torch's
    # generator too, and the published runs seeded in this order.
    th.manual_seed(args.seed)

    checkpoint_sha256 = None
    if args.run_id is not None:
        from sb3_contrib import MaskablePPO  # noqa: PLC0415

        path = (
            args.model_root
            / args.run_id
            / CHECKPOINT_DIRNAME
            / f"{args.checkpoint}.zip"
        )
        checkpoint_sha256 = file_sha256(path)
        model = MaskablePPO.load(path, device=args.device)
        greedy = model_policy(model, deterministic=True)
        sampled = model_policy(model, deterministic=False)
    else:
        # A baseline consults the rng at every step, so its "deterministic" column is one
        # sampled rollout and best-of-N is the same policy given more of them.
        greedy = sampled = POLICIES[args.baseline]

    started = datetime.now(timezone.utc)
    deterministic_solved, first_success = score_samples(
        samples, greedy, sampled, args.seed, args.max_attempts
    )
    seconds = (datetime.now(timezone.utc) - started).total_seconds()
    total = len(samples)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": label,
        "checkpoint": args.checkpoint if args.run_id is not None else None,
        "checkpoint_sha256": checkpoint_sha256,
        **source,
        "size": args.size,
        "seed": args.seed,
        "device": args.device,
        "judge": JUDGE,
        "puzzles": total,
        "max_attempts": args.max_attempts,
        "seconds": seconds,
        "deterministic_solve_rate": deterministic_solved / total,
        "best_of_n": best_of_n(first_success, args.max_attempts),
        "attempts_when_solved": summarise_attempts(first_success),
    }

    logger.info(
        f"{label} on {args.size}x{args.size} [{source['split']}], {total} puzzles"
    )
    logger.info(f"deterministic  solve={summary['deterministic_solve_rate']:.6f}")
    for n, stats in summary["best_of_n"].items():
        logger.info(
            f"best-of-{n:<3s}    solve={stats['solve_rate']:.6f} "
            f"episodes/puzzle={stats['episodes_per_puzzle']:.3f}"
        )
    logger.info(f"elapsed {seconds:.1f}s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.success(f"written: {artifact}")
    return artifact


def table_row(label: str, artifact: dict[str, Any]) -> str:
    best = artifact["best_of_n"]
    rates = [
        f"{best[str(n)]['solve_rate']:.4f}" if str(n) in best else "—" for n in TABLE_N
    ]
    largest = max(best, key=int) if best else None
    cost = f"{best[largest]['episodes_per_puzzle']:.2f}" if largest else "—"
    return (
        f"| {label} | {artifact['deterministic_solve_rate']:.4f} | "
        + " | ".join(rates)
        + f" | {cost} |"
    )


def table(args: argparse.Namespace) -> None:
    """Markdown rows straight from the artifacts, so no number is copied by hand."""
    header = "| policy | deterministic | " + " | ".join(f"best-of-{n}" for n in TABLE_N)
    lines = [header + " | episodes/puzzle @max |", "|" + "---|" * (len(TABLE_N) + 3)]
    for label in args.labels:
        path = (
            args.output_dir
            / f"{ARTIFACT_PREFIX}_{label}_{args.size}x{args.size}_{args.split}.json"
        )
        if not path.exists():
            lines.append(f"| {label} | (no artifact) |")
            continue
        lines.append(table_row(label, json.loads(path.read_text(encoding="utf-8"))))
    # The table is this command's product, meant to be pasted, not a log line.
    sys.stdout.write("\n".join(lines) + "\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("score", help="Score a checkpoint or a baseline.")
    policy = run.add_mutually_exclusive_group(required=True)
    policy.add_argument("--run-id", help="A run under --model-root.")
    policy.add_argument("--baseline", choices=sorted(POLICIES))
    puzzles = run.add_mutually_exclusive_group(required=True)
    puzzles.add_argument(
        "--eval-set", type=Path, help="A frozen split (see `eval_set.py`)."
    )
    puzzles.add_argument("--dataset", help="A pickled dataset under --dataset-root.")
    run.add_argument("--split", default="test", help="With --dataset only.")
    run.add_argument("--size", type=int, required=True)
    run.add_argument("--checkpoint", default=FINAL_CHECKPOINT_NAME)
    run.add_argument(
        "--label", default=None, help="Artifact label; needed with --checkpoint."
    )
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    run.add_argument("--limit", type=int, default=None, help="Subsample, for timing.")
    run.add_argument("--device", default="auto")
    run.add_argument("--model-root", type=Path, default=MODEL_ROOT)
    run.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    run.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    run.add_argument("--overwrite", action="store_true")

    rows = commands.add_parser("table", help="Markdown rows from written artifacts.")
    rows.add_argument("labels", nargs="+")
    rows.add_argument("--size", type=int, required=True)
    rows.add_argument("--split", default="test")
    rows.add_argument("--output-dir", type=Path, default=ARTIFACT_DIR)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.command == "table":
        table(args)
        return
    if args.checkpoint != FINAL_CHECKPOINT_NAME and args.label is None:
        parser.error(
            "--checkpoint needs --label: the artifact is named after the label"
        )
    if args.max_attempts < 0:
        parser.error("--max-attempts cannot be negative")
    score(args)


if __name__ == "__main__":
    main()
