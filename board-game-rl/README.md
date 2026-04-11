# Board Game RL (棋盤遊戲強化學習)

> **Last Updated:** 2026-04-11

本專案 `board-game-rl` 是一個專注於**棋類遊戲的強化學習演算法**研究與實作的實驗室。我們透過建立標準化的環境，探索從傳統搜尋演算法到現代強化學習在不同棋類遊戲中的應用。

## 核心架構 (Domain-Driven Design)

專案採用高度解耦的「領域驅動設計」，確保演算法的通用性與遊戲擴充的彈性：

1. **Games**: 獨立封裝井字遊戲邏輯與 Gymnasium 環境。
2. **Agents**: 實作通用的 AI 演算法（Q-Learning, DQN, Random），可無痛遷移至不同棋類。
3. **Multi-Agent UI**: 提供分頁式 Gradio 介面，可同時與不同層級的 AI 對弈，支援先後手選擇。

## 快速開始

### 1. 啟動 Web UI 對弈介面
```bash
# 本機直接啟動 (Gradio UI: 7860)
cd board-game-rl
uv run python src/board_game_rl/ui/gradio_app.py

# 或透過 Docker (含 FastAPI: 8000)
docker compose up -d
```

### 2. 訓練 AI

```bash
cd board-game-rl

# Q-Learning (查表法，150K 場，不敗)
uv run python scripts/train_q_learning.py

# DQN (神經網路，100K 場)
uv run python scripts/train_dqn.py
```

### 3. 終端對戰
```bash
cd board-game-rl
uv run python scripts/play.py
```

## 目前成果

| Agent | 方法 | vs Alpha-Beta | vs Random | 可擴展性 |
|-------|------|---------------|-----------|----------|
| **Q-Learning** | 查表 (3,441 states) | 零敗 | **零敗** | 僅限小遊戲 |
| **DQN** | 神經網路 (MLP 18K params) | 零敗 | 後手 ~2.5% 敗率 | **可擴展** |
| **Alpha-Beta** | Minimax + 剪枝 | — | — | 完美解 |

## 演算法教材

專案內含精心編寫的雙版本教學文件（高中生版 + 專業版，含圖解與實際推演）：

- [從零到不敗：Q-Learning 訓練方法論](docs/q_learning_unbeatable_tutorial_2026-03-28.md)
- [深入了解 Alpha-Beta 剪枝法](docs/alphabeta_tutorial_2026-03-22.md)
- [深入了解 Q-Learning 演算法](docs/q_learning_tutorial_2026-03-22.md)
- [貝爾曼方程式：看透未來的魔法](docs/bellman_equation_tutorial_2026-03-22.md)
- [反事實遺憾最小化 (CFR)](docs/cfr_tutorial.md)
- [學習路線圖：從井字遊戲到麻將](docs/learning_path.md)

## 未來展望 (Future Work)
- [x] ~~**DQN (Deep Q-Network)**: 將查表法升級為神經網路大腦~~
- [ ] **MCTS (蒙地卡羅樹搜尋)**: 實作 AlphaZero 的核心搜尋演算法
- [ ] **AlphaZero**: MCTS + Policy/Value Network
- [ ] **新遊戲擴充**: Connect Four (四子棋)
