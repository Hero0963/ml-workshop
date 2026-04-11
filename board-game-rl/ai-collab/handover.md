# 交接文件 (Handover)

> **Last Updated:** 2026-04-11
> **上一位 Agent 的最後工作**: 實作 DQN Agent + UI 先後手選擇

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
| DQN Agent | 完成 | MLP (9→128→128→9)，vs AB 零敗，vs Random 後手 ~2.5% 敗率 |
| Web UI (Gradio) | 完成 | 四種對手分頁 + **先後手選擇** |
| FastAPI 後端 | 完成 | `/predict` 端點，agent 快取 |
| 平行訓練 | 完成 | Q-Learning 多核 CPU，6.3x 加速 |
| 教學文件 | 完成 | Alpha-Beta / Q-Learning / Bellman / 訓練方法論 / CFR / 路線圖 |
| 單元測試 | 完成 | 28 個 tests 全通過 |

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

## 下一步 (Suggested Next Steps)

### 優先順序建議

#### 1. MCTS (蒙地卡羅樹搜尋) — 推薦優先
AlphaZero 的核心搜尋演算法。先在井字遊戲上實作驗證，再擴展到大遊戲。

**具體任務**:
- 實作 MCTS Agent (UCB1 selection + random rollout)
- 與 Alpha-Beta 做對比
- 教學文件（高中生版 + 專業版）

#### 2. AlphaZero (MCTS + Neural Network)
結合 MCTS 搜尋 + Policy/Value 雙頭網路，實現完整的 AlphaZero 架構。

**具體任務**:
- 實作 dual-head ResNet (同時輸出 policy 分佈 + value)
- 將 MCTS 與神經網路結合 (PUCT selection)
- Self-play 訓練流程

#### 3. 新遊戲擴充
測試 DDD 架構的擴展能力。

**候選遊戲**: Connect Four (四子棋) — 狀態空間比井字大但仍可控，適合驗證 DQN/MCTS

---

## 環境確認清單

接手時請確認：
- [ ] `uv run pytest tests/ -v` — 28 tests 全通過
- [ ] `uv run python scripts/play.py` — 可以在終端對戰
- [ ] `models/q_table.json` 存在 (3,441 states)
- [ ] `models/alphabeta_cache.json` 存在 (4,520 states)
- [ ] `models/dqn_model.pth` 存在 (~79 KB)
