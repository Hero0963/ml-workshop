# Board Game RL (棋盤遊戲強化學習)

> **Last Updated:** 2026-03-22

本專案 `board-game-rl` 是一個專注於**棋類遊戲的強化學習演算法**研究與實作的實驗室。我們透過建立標準化的環境，探索從傳統搜尋演算法到現代強化學習在不同棋類遊戲中的應用。

## 🌟 核心架構 (Domain-Driven Design)

專案採用高度解耦的「領域驅動設計」，確保演算法的通用性與遊戲擴充的彈性：

1. **Games**: 獨立封裝井字遊戲邏輯與 Gymnasium 環境。
2. **Agents**: 實作通用的 AI 演算法（如 Tabular Q-Learning），可無痛遷移至不同棋類。
3. **Multi-Agent UI**: 提供分頁式 Gradio 介面，可同時與不同層級的 AI（完美大師、學習中的小白、隨機亂走）對弈。

## 🚀 快速開始

### 1. 環境安裝與啟動
專案完整支援 Docker 容器化開發與 GPU 加速：
```bash
# 啟動所有服務 (Gradio UI: 7860, FastAPI: 8000)
docker compose up -d
```

### 2. 訓練你的 AI (Q-Learning)
```bash
# 進入容器讓 AI 自我對弈兩萬盤
docker compose exec play uv run python scripts/train_q_learning.py
```

## 📚 演算法教材
專案內含精心編寫的雙版本教學文件（含圖解與實際推演）：
- [深入了解 Alpha-Beta 剪枝法](docs/alphabeta_tutorial_2026-03-22.md)
- [深入了解 Q-Learning 演算法](docs/q_learning_tutorial_2026-03-22.md)
- [貝爾曼方程式：看透未來的魔法](docs/bellman_equation_tutorial_2026-03-22.md)

## 🔮 未來展望 (Future Work)
- [ ] **DQN (Deep Q-Network)**: 將查表法升級為神經網路大腦。
- [ ] **MCTS (蒙地卡羅樹搜尋)**: 實作 AlphaZero 的核心搜尋演算法。
- [ ] **新遊戲擴充**: 引入圍棋 (Go) 或中國象棋引擎。
