# CLAUDE.md — linkedin-zip-challenge

> 從**這個子專案目錄**開 Claude Code 時自動讀本檔。內容用 `@import` 載入，避免重複維護。
> 從 **repo 根**開的話讀的是 `../CLAUDE.md`（載入 repo 級 `../AGENTS.md`），要動這個子專案時再讀本目錄的 `AGENTS.md`。

## 操作指南（本子專案正本）
@AGENTS.md

## Claude Code 專屬備註

- **第一站是 `ai-collab/roadmap.md`**：現況、下一步、已定案不要再重開的決策。
- **本專案自 2025-10-30 起休眠約 9 個月**——開發前先 `uv sync` ＋ `uv run pytest` 建立基線，不要假設環境還是好的。
- **venv**：Python 3.11，一律 `cd linkedin-zip-challenge` 再 `uv run`；repo 根的 `.venv` 是 py3.9 devtools，不能拿來跑這個專案。
- `ai-collab/dev_log.md` 有 600+ 行，**不要整份讀**，用關鍵字搜需要的段落。
- SessionStart hook 設在 repo 根的 `.claude/settings.json`（從子專案目錄開時不會自動跑，可手動 `python ../.claude/session-brief.py`）。
- ⚠ **那份簡報不能當現況的依據**（2026-08-29 實測）：hook 用**絕對路徑**指向 `ml-workshop/.claude/session-brief.py`，
  而腳本內部又是 `git -C <腳本所在的 repo>`，所以**分支、工作區狀態、roadmap 三者全部來自 `ml-workshop` 那個 checkout**。
  在別的 worktree 工作時它會報 `ml-workshop` 的分支；`ml-workshop` 沒 `git pull` 時它報的是舊的下一步。
  **現況一律以 `ai-collab/roadmap.md` 與該 track 的 handover 為準。**
