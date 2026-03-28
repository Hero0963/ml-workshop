# 開發日誌 (Dev Log)

## 2026-03-07: 初始化 Board Game RL
- **任務**: 建立棋類強化學習研究環境
- **進度**:
  - 建立 Git 分支 `research/board-game-rl`
  - `uv init --python 3.13` 初始化獨立環境
  - 完成 `engine.py` 與 `tic_tac_toe_env.py`
- **下一步**: 容器化 + Web UI

## 2026-03-14: 容器化、Web UI、Alpha-Beta Agent
- **任務**: 完整開發者體驗 + 第一個 AI Agent
- **進度**:
  - Dockerfile (CUDA 12.6) + docker-compose + Hot-Reload
  - FastAPI (`/predict`) + Gradio UI (分頁對弈)
  - Alpha-Beta Agent (Minimax + Pruning)，含視覺化 Log
  - 單元測試 3 個通過
  - CFR / Alpha-Beta 教學文件

## 2026-03-22: Q-Learning 基礎版 + DDD 重構
- **任務**: 實作基礎 Q-Learning Agent
- **進度**:
  - Tabular Q-Learning Agent (Epsilon-Greedy + Bellman Update)
  - 訓練腳本 20K 場 vs Random
  - DDD 架構重構（`games/` vs `agents/` 解耦）
  - Q-Learning / Bellman Equation 教學文件

## 2026-03-28: Q-Learning 不敗 + 平行訓練 + D4 對稱
- **任務**: 讓 Q-Learning Agent 達到「跟誰下都不輸」的不敗水準
- **進度**:
  - **棋盤正規化**: `player` 參數 + `_normalize_obs()`，一張 Q-Table 通吃先後手
  - **訓練腳本重寫**: 混合對手 (Random / Self-Play / Alpha-Beta / Hybrid)
  - **Hybrid 對手設計**: Random 開局 1-3 步 → Alpha-Beta 收尾（Developer 提出的 idea）
  - **平行訓練**: `ProcessPoolExecutor`，16 核 6.3x 加速，Q-Table 平均合併
  - **D4 對稱查表**: 8 種旋轉/翻轉，推論時自動匹配，3,441 狀態覆蓋全部盤面
  - **驗證**: vs AB / Random / Self 各 5,000 場 x 先後手 = 30,000 場，全部零敗
  - 更新 `inference.py` 傳入 `player` + `play.py` 改用 Q-Agent
  - 教學文件: `docs/q_learning_unbeatable_tutorial_2026-03-28.md`
  - 訓練報告: `ai-collab/reports/training_report_2026-03-28.md`
- **關鍵決策**:
  - 直接跟 Alpha-Beta 訓練行不通（確定性對手只覆蓋 ~165 狀態）
  - Hybrid 對手是最有效的解法（Random 提供覆蓋 + AB 提供品質）
  - 訓練不需 GPU，純 CPU + dict 查表
- **下一步**: 見 `ai-collab/handover.md`
