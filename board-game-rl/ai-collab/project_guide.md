# 專案指南 (Project Guide)

> **Last Updated:** 2026-04-11

本文件紀錄 `board-game-rl` 專案的整體架構、核心設計與啟動方式。

## 核心設計理念 (Domain-Driven Design)

為了支援未來的多種棋類遊戲（如圍棋、中國象棋），專案採用「按遊戲領域分類」的設計，實現演算法與遊戲規則的高度解耦：

1. **Games (`games/`)**: 每個遊戲擁有獨立目錄（如 `tic_tac_toe/`）。
   - `engine.py`: 純粹的遊戲規則判定
   - `env.py`: 遵循 [Gymnasium](https://gymnasium.farama.org/) 介面的訓練環境
   - `alphabeta_agent.py`: 依賴特定規則的搜尋演算法
2. **Agents (`agents/`)**: 放置**通用的 AI 演算法**（如 `QLearningAgent`, `RandomAgent`）。這些 Agent 只看 `observation` 與 `legal_actions`，可直接套用於不同遊戲。
3. **API & UI**: 對外的預測服務與對弈介面。

## 專案結構

```text
board-game-rl/
├── src/board_game_rl/
│   ├── games/tic_tac_toe/      # 井字遊戲專屬 (Engine, Env, Alpha-Beta)
│   ├── agents/                 # 通用 AI 大腦 (Q-Learning, DQN, Random)
│   ├── api/                    # FastAPI 後端 (支援多 Agent 預測)
│   ├── ui/                     # Gradio 前端 (分頁對弈介面)
│   └── utils/                  # 共用工具 (Logger 等)
├── scripts/                    # 訓練、預計算、對戰腳本
├── models/                     # 訓練好的模型與 JSON 快取表
├── docs/                       # 演算法教材 (Markdown)
├── tests/                      # 單元測試 (DDD 鏡像結構)
└── ai-collab/                  # AI 協作文件 (規範、日誌、交接)
    └── reports/                # 訓練報告
```

## 已實作功能

### 1. Q-Learning Agent（不敗）
- **棋盤正規化**: 單一 Q-Table 同時覆蓋先手 (X) 與後手 (O)
- **D4 對稱查表**: 8 種旋轉/翻轉，3,441 種狀態等效覆蓋全部盤面
- **混合對手訓練**: Random / Self-Play / Alpha-Beta / Hybrid (Random→AB)
- **平行訓練**: 多核 CPU 加速，自動產出 Markdown 訓練報告
- **驗證結果**: vs Alpha-Beta、Random、Self 均零敗

### 2. Alpha-Beta Agent（完美解）
- Minimax + Alpha-Beta Pruning，井字遊戲的數學最優解
- 預計算快取 `models/alphabeta_cache.json`（O(1) 查表）

### 3. DQN Agent（Deep RL）
- PyTorch MLP (9→128→128→9)，Q-value function approximation
- Experience Replay Buffer (50K) + Target Network (500 步同步)
- vs Alpha-Beta 零敗，vs Random 後手 ~2.5% 敗率
- 訓練報告: `ai-collab/reports/dqn_training_report_2026-04-11.md`

### 4. Web UI 對弈介面
- Gradio 分頁式 UI，支援 Alpha-Beta / DQN / Q-Learning / Random 四種對手
- 支援先後手選擇（先手 X / 後手 O）

## 啟動方式

### 本機直接啟動
```bash
cd board-game-rl

# Web UI (localhost:7860)
uv run python src/board_game_rl/ui/gradio_app.py

# 終端對戰
uv run python scripts/play.py

# 訓練 Q-Learning
uv run python scripts/train_q_learning.py

# 執行測試
uv run pytest tests/ -v
```

### Docker 啟動
```bash
docker compose up -d
# Gradio: localhost:7860, FastAPI: localhost:8000
```

## 開發規範
- **Plan First**: Agent 在修改代碼前須先提出計畫
- **DDD Structure**: 新增遊戲時須遵循 `games/` 目錄規範
- 詳細規範見 `ai-collab/rules.md`
