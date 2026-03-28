# Board Game RL (棋盤遊戲強化學習)

> **Last Updated:** 2026-03-28

本專案 `board-game-rl` 是一個專注於**棋類遊戲的強化學習演算法**研究與實作的實驗室。我們透過建立標準化的環境，探索從傳統搜尋演算法到現代強化學習在不同棋類遊戲中的應用。

## 🌟 核心架構 (Domain-Driven Design)

專案採用高度解耦的「領域驅動設計」，確保演算法的通用性與遊戲擴充的彈性：

1. **Games**: 獨立封裝井字遊戲邏輯與 Gymnasium 環境。
2. **Agents**: 實作通用的 AI 演算法（如 Tabular Q-Learning），可無痛遷移至不同棋類。
3. **Multi-Agent UI**: 提供分頁式 Gradio 介面，可同時與不同層級的 AI 對弈。

## 🚀 快速開始

### 1. 啟動 Web UI 對弈介面
```bash
# 本機直接啟動 (Gradio UI: 7860)
cd board-game-rl
uv run python src/board_game_rl/ui/gradio_app.py

# 或透過 Docker (含 FastAPI: 8000)
docker compose up -d
```

### 2. 訓練不敗 AI (Q-Learning)
```bash
cd board-game-rl
uv run python scripts/train_q_learning.py
```
- 150,000 場混合對手訓練（Random / Self-Play / Alpha-Beta / Hybrid）
- 棋盤正規化：一張 Q-Table 同時覆蓋先手與後手
- D4 對稱查表：3,441 種狀態等效覆蓋全部 ~5,000 種盤面
- 支援多核平行訓練（自動偵測 CPU 核數）
- 自動驗證 + 產出 Markdown 訓練報告

### 3. 終端對戰
```bash
cd board-game-rl
uv run python scripts/play.py
```

## 🏆 目前成果
- **Q-Learning Agent**: 經過 150K 場訓練，達到**不敗 (Unbeatable)** — 對 Alpha-Beta、Random、Self 三種對手均零敗
- **Alpha-Beta Agent**: 基於 Minimax + Alpha-Beta Pruning 的完美解，含預計算快取
- **平行訓練**: 16 核加速 6.3 倍

## 📚 演算法教材
專案內含精心編寫的雙版本教學文件（含圖解與實際推演）：
- [從零到不敗：Q-Learning 訓練方法論](docs/q_learning_unbeatable_tutorial_2026-03-28.md) **NEW**
- [深入了解 Alpha-Beta 剪枝法](docs/alphabeta_tutorial_2026-03-22.md)
- [深入了解 Q-Learning 演算法](docs/q_learning_tutorial_2026-03-22.md)
- [貝爾曼方程式：看透未來的魔法](docs/bellman_equation_tutorial_2026-03-22.md)

## 🔮 未來展望 (Future Work)
- [ ] **DQN (Deep Q-Network)**: 將查表法升級為神經網路大腦。
- [ ] **MCTS (蒙地卡羅樹搜尋)**: 實作 AlphaZero 的核心搜尋演算法。
- [ ] **新遊戲擴充**: 引入圍棋 (Go) 或中國象棋引擎。
