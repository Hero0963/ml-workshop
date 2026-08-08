# .claude/session-brief.py
"""ml-workshop — SessionStart 簡報：印出 repo 現況與各子專案下一步，讓新 session 不必自己摸索。

只做唯讀操作（git log / status ＋ 讀 ai-collab/roadmap.md）。由 .claude/settings.json 掛在 SessionStart。
純標準函式庫，不依賴任何 venv。
"""

import re
import subprocess
import sys
from pathlib import Path

# Windows 主控台預設 cp950，中文輸出會 UnicodeEncodeError；強制 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent

# 有 ai-collab/roadmap.md 的子專案會被掃描；沒有的就跳過（不報錯）
SUBPROJECTS = [
    "linkedin-zip-challenge",
    "board-game-rl",
    "deep-learning-karpathy",
    "lingua-tutor",
    "more_simple_reinforcement_learning",
    "notes",
]

MAX_NEXT_STEPS = 3
MAX_DIRTY_SHOWN = 5


def git(*args: str) -> str:
    """跑一次 git，失敗就回空字串（例如不是 git repo）。"""
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def next_steps(roadmap: Path) -> list[str]:
    """抓 roadmap.md 的『## 下一步』區塊裡的頂層編號項（`1. **...**`）。"""
    try:
        text = roadmap.read_text(encoding="utf-8")
    except OSError:
        return []

    section = next(
        (
            s
            for s in re.split(r"^## ", text, flags=re.MULTILINE)
            if s.startswith("下一步")
        ),
        None,
    )
    if section is None:
        return []

    items = [ln for ln in section.splitlines() if re.match(r"^\d+\.\s+\*\*", ln)]
    return [
        re.sub(r"\*\*", "", re.sub(r"^\d+\.\s+", "", ln)).strip()
        for ln in items[:MAX_NEXT_STEPS]
    ]


def main() -> None:
    out = ["── ml-workshop · 機器學習實作練功房 ──"]

    branch = git("rev-parse", "--abbrev-ref", "HEAD") or "(非 git repo)"
    last = git("log", "-1", "--format=%h %s (%ad)", "--date=short") or "(無 commit)"
    out.append(f"分支 {branch}｜最近 commit：{last}")

    dirty = [ln for ln in git("status", "--porcelain").splitlines() if ln]
    if dirty:
        shown = "、".join(ln[3:] for ln in dirty[:MAX_DIRTY_SHOWN])
        tail = " …" if len(dirty) > MAX_DIRTY_SHOWN else ""
        out.append(f"工作區有 {len(dirty)} 個未提交變更：{shown}{tail}")
    else:
        out.append("工作區乾淨")

    for name in SUBPROJECTS:
        steps = next_steps(ROOT / name / "ai-collab" / "roadmap.md")
        if steps:
            out.append(f"下一步（{name}/ai-collab/roadmap.md）：")
            out.extend(f"  {i}. {s}" for i, s in enumerate(steps, 1))

    out.append(
        "venv：一律 cd <子專案> 再 uv run（根 .venv 是 py3.9 devtools）｜規範正本 AGENTS.md"
    )
    print("\n".join(out))


if __name__ == "__main__":
    main()
