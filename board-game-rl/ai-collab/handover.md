# 交接文件 (Handover)

> **Last Updated:** 2026-03-28
> **上一位 Agent 的最後工作**: 實現不敗 Q-Learning Agent

---

## 快速上手（新 Agent 必讀）

請依序閱讀以下文件：

1. **本文件** — 了解專案現況與下一步
2. `ai-collab/rules.md` — 開發規範（別踩雷）
3. `ai-collab/project_guide.md` — 架構與啟動方式
4. `ai-collab/dev_log.md` — 完整開發歷程
5. `README.md` — 對外文件

---

## 專案現況

### 已完成

| 功能 | 狀態 | 說明 |
|------|------|------|
| 井字遊戲 Engine | 完成 | `games/tic_tac_toe/engine.py`，純規則判定 |
| Gymnasium Env | 完成 | `games/tic_tac_toe/env.py`，標準 RL 介面 |
| Alpha-Beta Agent | 完成 | 完美解 + 預計算快取 |
| Q-Learning Agent | 完成 | **不敗**，150K 場訓練，D4 對稱查表 |
| Web UI (Gradio) | 完成 | 三種對手分頁對弈 |
| FastAPI 後端 | 完成 | `/predict` 端點 |
| 平行訓練 | 完成 | 多核 CPU，6.3x 加速 |
| 教學文件 | 完成 | Alpha-Beta / Q-Learning / Bellman / 訓練方法論 |
| 單元測試 | 完成 | 12 個 tests 全通過 |

### Q-Learning Agent 技術細節

```
訓練: 150,000 場 × 混合對手 (Random/Self/AB/Hybrid)
Q-Table: 3,441 states × D4 symmetry = 覆蓋全部 ~5,000 種盤面
驗證: vs AB/Random/Self × 先後手 × 5,000 場 = 30,000 場零敗
關鍵技術:
  - Board Normalization (player=-1 時翻轉棋盤)
  - D4 Symmetry Lookup (8 種旋轉/翻轉查表)
  - Hybrid Opponent (Random 開局 → AB 收尾)
```

### 已知限制

1. **Gradio CSS Warning**: Gradio 6.0 將 `css` 參數從 `Blocks()` 移到 `launch()`，目前有 warning 但不影響功能
2. **inference.py 每次建新 Agent**: 每次 API call 都重新 load Q-table，效率不高（可改為單例模式）
3. **Q-Learning 只適用小遊戲**: Tabular 方法無法處理大狀態空間（圍棋等）

---

## 下一步 (Suggested Next Steps)

### 優先順序建議

#### 1. DQN (Deep Q-Network) — 推薦優先
將查表法升級為神經網路，為擴展到更大遊戲奠定基礎。

**為什麼重要**: Tabular Q-Learning 在井字遊戲已達到極限（不敗）。下一個有意義的挑戰是用 neural network 做 function approximation，這是通往 AlphaZero 的必經之路。

**具體任務**:
- 實作 DQN Agent (`agents/dqn_agent.py`)，用 PyTorch 建立簡單 MLP
- 實作 Experience Replay Buffer
- 實作 Target Network (Double DQN)
- 訓練腳本 + GPU 支援
- 與 Tabular Q-Learning 做效能對比

**預期學習**: Loss function (MSE)、Backpropagation、GPU training、neural network generalization

#### 2. MCTS (蒙地卡羅樹搜尋)
AlphaZero 的核心搜尋演算法。

**具體任務**:
- 實作 MCTS Agent (UCB1 selection + random rollout)
- 與 Alpha-Beta 做對比
- 教學文件

#### 3. 新遊戲擴充
測試 DDD 架構的擴展能力。

**候選遊戲**: Connect Four (四子棋) — 狀態空間比井字大但仍可控，適合驗證 DQN

---

## 環境確認清單

接手時請確認：
- [ ] `uv run pytest tests/ -v` — 12 tests 全通過
- [ ] `uv run python scripts/play.py` — 可以在終端對戰
- [ ] `models/q_table.json` 存在 (3,441 states)
- [ ] `models/alphabeta_cache.json` 存在 (4,520 states)
