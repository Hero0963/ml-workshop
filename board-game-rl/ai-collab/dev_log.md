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

## 2026-04-11: DQN (Deep Q-Network) 實作
- **任務**: 將查表法升級為神經網路，完成 Stage 1 的 Deep RL 里程碑
- **進度**:
  - **DQN Agent** (`agents/dqn_agent.py`): MLP (9→128→128→9) + Experience Replay + Target Network
  - **訓練腳本** (`scripts/train_dqn.py`): 混合對手訓練 + 即時 log + 定期驗證 + best model 自動儲存
  - **單元測試**: 12 個 DQN 測試（網路、Buffer、Agent、save/load）
  - **API/UI 整合**: inference.py 支援 DQN、Gradio 新增 DQN 分頁 + 先後手選擇
  - **inference.py 快取**: 修復每次 API call 重新載入模型的效能問題
  - **訓練結果**: 100K 場，vs Alpha-Beta 先後手零敗，vs Random 後手 2.5% 敗率
- **關鍵決策**:
  - 井字遊戲狀態空間小，DQN 的泛化能力反而不如 Q-Learning 的精確查表
  - vs Random 的敗率是 neural approximation 的本質限制，可透過增加 Random 對手比例改善
  - Best model 儲存用 `<=`（不嚴格大於），確保後期更穩定的模型能覆蓋早期版本
- **下一步**: 見 `ai-collab/handover.md`
