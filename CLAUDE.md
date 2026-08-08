# CLAUDE.md — ml-workshop

> Claude Code 每次 session 自動讀本檔。本檔用 `@import` 載入完整內容，**避免重複維護**：
> repo 級規範改 `AGENTS.md`、Python 程式碼規範改 `rules.md`，本檔只放 import ＋ Claude Code 專屬備註。

## 操作指南（repo 級正本）
@AGENTS.md

## Python 程式碼與工具鏈規範
@rules.md

## Claude Code 專屬備註

- 已設 **SessionStart hook**：每次自動跑 `.claude/session-brief.py`，印出分支／最近 commit／工作區狀態／各子專案下一步（設定在 `.claude/settings.json`）。
- 首次遇到 `@import` 會跳一次核可對話框，按「允許」即可，之後每次自動載入 `AGENTS.md` 與 `rules.md`。
- **進子專案工作要再讀該子專案的 `AGENTS.md`**（例如 `linkedin-zip-challenge/AGENTS.md`）；從子專案目錄直接開 Claude Code 也可以，那裡有自己的 `CLAUDE.md`。
- 助理的跨 session 記憶在 `~/.claude/projects/D--it-project-github-sync-ml-workshop/memory/`（索引 `MEMORY.md`）。
- **venv 陷阱**：repo 根的 `.venv` 是 py3.9 devtools，不是拿來跑子專案的。一律 `cd <子專案>` 再 `uv run`（見 `AGENTS.md §5`）。
