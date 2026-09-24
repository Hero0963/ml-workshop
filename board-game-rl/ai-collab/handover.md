# 交接文件 (Handover)

> **Last Updated:** 2026-09-24
> **上一位 Agent 的最後工作**（2026-09-24）: 重新建立基線、補 Q-Learning 測試、建立 `roadmap.md` 與 `notes/`；
> **M1 純 MCTS 做到能在 Gradio 對弈**（本人已實際下過）。M1 剩 S3／S4／S6，見下方「下次開工」。

---

## 快速上手（新 Agent 必讀）

請依序閱讀以下文件：

1. `ai-collab/roadmap.md` — **現況與下一步**（第一站）
2. **本文件** — 已完成的功能、已知限制、環境確認清單
3. `ai-collab/rules.md` — 開發規範（別踩雷）
4. `ai-collab/project_guide.md` — 架構與啟動方式
5. `ai-collab/notes/` — 討論與決策的來龍去脈（為什麼這樣做、否決了什麼）
6. `ai-collab/dev_log.md` — 完整開發歷程
7. `README.md` — 對外文件

---

## 專案現況

### 已完成

| 功能 | 狀態 | 說明 |
|------|------|------|
| 井字遊戲 Engine | 完成 | `games/tic_tac_toe/engine.py`，純規則判定 |
| Gymnasium Env | 完成 | `games/tic_tac_toe/env.py`，標準 RL 介面 |
| Alpha-Beta Agent | 完成 | 完美解 + 預計算快取 |
| Q-Learning Agent | 完成 | **不敗**，150K 場訓練，D4 對稱查表 |
| DQN Agent | 完成 | MLP (9→128→128→9)，vs AB 零敗，vs Random 後手 ~2.5% 敗率 |
| MCTS Agent | **可對弈，實驗未做** | `agents/mcts_agent.py`（UCT ＋ 隨機推演，通用）＋ `games/base.py`（`GameRules` 介面）＋ `games/tic_tac_toe/rules.py`；API／Gradio 暫定 2,000 次模擬 |
| Web UI (Gradio) | 完成 | 五種對手分頁（2026-09-24 加 MCTS）+ **先後手選擇** |
| FastAPI 後端 | 完成 | `/predict` 端點，agent 快取 |
| 平行訓練 | 完成 | Q-Learning 多核 CPU，6.3x 加速 |
| 教學文件 | 完成 | Alpha-Beta / Q-Learning / Bellman / 訓練方法論 / CFR / 路線圖 |
| 單元測試 | 完成 | 69 個 tests 全通過（2026-09-24 補上 Q-Learning 12 個、規則 11 個、MCTS 17 個、API 1 個）|

### DQN Agent 技術細節

```
網路: MLP 9 → 128 → 128 → 9 (PyTorch, ~18K params)
訓練: 100,000 場 × 混合對手 (Random/Self/AB/Hybrid)
技術: Experience Replay (50K buffer) + Target Network (500 步同步)
驗證: vs AB 先後手 零敗 / vs Random 先手 零敗 / vs Random 後手 ~2.5% 敗率
關鍵技術:
  - Board Normalization (與 Q-Learning 相同)
  - Epsilon-greedy → 0.01 (探索衰減)
  - MSE Loss + Gradient Clipping
  - Best model 定期驗證 + 自動儲存 (>=)
```

### DQN vs Q-Learning 對比

| | Q-Learning (查表) | DQN (神經網路) |
|---|---|---|
| vs AB | 零敗 | 零敗 |
| vs Random | **零敗** | 後手 ~2.5% 敗率 |
| 可擴展性 | 僅限小遊戲 | **可擴展到大遊戲** |
| 訓練時間 | ~14 分鐘 | ~14 分鐘 |

### 已知限制

1. **Gradio CSS Warning**: Gradio 6.0 將 `css` 參數從 `Blocks()` 移到 `launch()`，有 warning 但不影響功能
2. **DQN vs Random 後手敗率**: 神經網路近似的本質限制，可透過增加 Random 對手比例、D4 對稱增強等方式改善

---

## 下一步

**正本在 [`roadmap.md`](roadmap.md)**（2026-09-24 起；本檔不再重複維護，避免兩份不同步）。
目前進行中的是 M1 純 MCTS，計畫書：[`plans/2026-09-24_m1-pure-mcts.md`](plans/2026-09-24_m1-pure-mcts.md)。

### 下次開工（M1 接續）

1. 在 worktree `ml-workshop/.claude/worktrees/board-game-rl-review-4ded4f` 工作（分支 `claude/board-game-rl-review-4ded4f`，`.venv` 已建好）；
   若換了 checkout，先 `cd board-game-rl && uv sync --locked`。
2. 跑基線：`uv run --with pytest pytest tests -q`（2026-09-24：69 passed）。
3. 從計畫書的 **S3 模擬次數掃描** 開始：寫 `scripts/eval_mcts.py`，找出不輸給 Alpha-Beta 的最小模擬次數 $N^*$，
   再把 `api/inference.py` 的 `MCTS_SIMULATIONS`（暫定 2,000）換成 $N^*$ 加餘裕。每秒約 67,000 次模擬，掃描預估幾分鐘。
4. 接著 S4（空盤面根節點統計）、S6（`docs/` 的 MCTS 教材、`project_guide.md`、`README.md`、`docs/learning_path.md`）。
5. 想跟 MCTS 下棋：在 `board-game-rl/` 執行 `uv run --locked python src/board_game_rl/ui/gradio_app.py`，開 `http://localhost:7860` 的「MCTS」分頁
   （Claude Code 的瀏覽器窗格可用 repo 根 `.claude/launch.json` 的 `board-game-rl-gradio`）。

---

## Future Work

### 同一套 API、兩種前端（Gradio ＋ Svelte）

2026-09-24 本人決定先列為 future work，當天只用 Gradio 跟 MCTS 下棋。
完整評估（thread-the-grid 的做法、本專案的差距、推薦做法）見
[`notes/2026-09-24-review-and-m1-decisions.md`](notes/2026-09-24-review-and-m1-decisions.md) §6。重點：

- **目標**：一個 FastAPI 埠同時提供 `/api`、`/ui`（Gradio）、第二個前端（Svelte＋Vite 建置的靜態檔），兩個前端都走 HTTP 呼叫同一組 API。
  這是 thread-the-grid 已經在用的架構（`thread-the-grid/src/app/main.py`）。
- **現況的差距**：Gradio 目前直接 import `get_optimal_move`，沒經過 API；FastAPI 與 Gradio 是兩個 process（`start.sh`）；
  勝負判定只在 Gradio 端；對手清單是寫死的字串。
- **推薦**：勝負判定放後端（API 回傳 agent 的那一步＋下完後的盤面＋勝負）；API 提供對手清單；Gradio 改走 HTTP。
- **會動到的檔**：`api/`、`ui/`、`api/inference.py`，另開一條 track 並寫自己的計畫書。

---

## 環境確認清單

接手時請確認（在 worktree 裡工作的話，`.venv` 不會跟著來，要先 `uv sync --locked`）：
- [ ] `uv sync --locked` — 依 lockfile 建環境，不重新解析版本
- [ ] `uv run pytest tests/ -v` — 69 tests 全通過（`pytest` 列入 dev 相依之前，改用 `uv run --with pytest pytest tests/ -v`）
- [ ] `uv run python scripts/play.py` — 可以在終端對戰
- [ ] `models/q_table.json` 存在 (3,441 states)
- [ ] `models/alphabeta_cache.json` 存在 (4,520 states)
- [ ] `models/dqn_model.pth` 存在 (~79 KB)
