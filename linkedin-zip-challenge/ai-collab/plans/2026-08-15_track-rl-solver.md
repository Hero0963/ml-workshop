# 任務計畫書 — Track RL：重啟 RL solver（讓 agent 自己通關）

> **給接手這條 track 的 agent。這是你的作戰計畫，不是背景資料。**
> 建立日期：2026-08-15（Asia/Taipei）｜對應 roadmap 第 3 項｜姊妹 track：[track-vlm-parser](2026-08-15_track-vlm-parser.md)
> **完整的失敗根因分析與設計推理不在本檔**，在 [`../reports/2026-08-15_rl-restart-plan.html`](../reports/2026-08-15_rl-restart-plan.html)（用瀏覽器開）。
> **開工前一定要讀那份報告的第 2 節**——不讀就會重蹈 2025-10 的覆轍。

---

## 0. 一句話目標

**訓練一個能對「沒看過的新題目」直接產出通關路徑的策略網路**，取代 2025-10 那個在確定性評估下必定卡死的 DQN。

**明確不做**：

- ❌ 不碰 `src/core/vl_models/`（另一條 track）
- ❌ 不試圖用 RL 打敗 CP-SAT（不會贏，也不是目的）
- ❌ **不要再靠調 reward 權重去修迴圈**（2025-10-12 與 10-13 各撞一次牆，根因不在權重）
- ❌ 不從 `models/dqn_*.pth` 續訓（那是失敗策略的權重，砍掉重練）
- ❌ 不自己 `uv add`

---

## 1. 開工前必讀（照順序）

1. 本檔
2. **[`../reports/2026-08-15_rl-restart-plan.html`](../reports/2026-08-15_rl-restart-plan.html) 第 2 節「機制性根因」** ← 最重要，講清楚為什麼舊版必然失敗
3. [`../roadmap.md`](../roadmap.md) — 已定案不要重開的決策
4. [`../../AGENTS.md`](../../AGENTS.md) — 子專案規範正本
5. `../dev_log.md` 的 2025-10-12／10-13／10-15 三則（用日期搜，不要整份讀）
6. 教材：`../../../more_simple_reinforcement_learning/chap_05_ppo`、`chap_02_dqn`

---

## 2. Worktree 環境建置 ★

新 worktree 只會拿到進版控的檔案。**不會跟過來**的：

| 項目 | 狀態 | 怎麼辦 |
|------|------|--------|
| `.env` | ❌ 不進版控 | 從主工作樹複製（RL 用不到 ollama 設定，但 app 啟動會讀） |
| `.venv` | ❌ 不進版控 | 各 worktree 各自 `uv sync`（`torch==2.4.1+cu121`，數 GB） |
| `datasets/rl_datasets/` | ❌ 不進版控，**且主工作樹裡是空的** | A1 階段自己重新生成題庫 |
| `models/` | ❌ 不進版控 | 主工作樹有 2025-10 的 4 個舊 checkpoint，**不要續訓** |
| `logs/sb_tensorboard/` | ❌ 不進版控 | 自己產 |

```powershell
# --- 由本人執行 ---
cd D:\it_project\github_sync\ml-workshop
git worktree add ..\zip-rl -b feat/rl-masked-ppo main
Copy-Item .\linkedin-zip-challenge\.env ..\zip-rl\linkedin-zip-challenge\.env

# --- agent 從這裡開始 ---
cd ..\zip-rl\linkedin-zip-challenge
uv sync
uv run pytest                 # 基線：2026-08-08 紀錄為 46 passed，先確認全綠
uv run ruff check .
```

> **`uv run` 一律先 `cd linkedin-zip-challenge`**；repo 根的 `.venv` 是 py3.9 devtools。
> RL 全程**本機**跑，不需要 Colab。GPU 幾乎閒著，瓶頸在環境步進的 CPU。

---

## 3. ⚠ A0 之前必須先拍板的相依決策

`MaskablePPO` 在 `sb3-contrib`，而：

```
sb3-contrib 2.9.0  requires  stable_baselines3>=2.9.0,<3.0
stable-baselines3 2.9.0  requires  torch>=2.8
本專案 pyproject.toml     pins     torch==2.4.1  (+cu121 自訂 index)   ← 衝突
本專案 uv.lock            locks    stable-baselines3 2.7.0, gymnasium 1.2.1
```

三個選項（詳見報告 §5.2），**由本人決定，agent 不自己 `uv add`**：

| 選項 | 做法 | 代價 |
|------|------|------|
| **A（推薦）** | 升 `torch` 到 ≥2.8（換 cu124/cu126 index），SB3 與 sb3-contrib 一起上 2.9 | 動到刻意的 pin（`AGENTS.md §5`），要重跑 46 測試回歸 |
| B | 找與 SB3 2.7.0 相容的 sb3-contrib 版本 | PyPI 上未確認到 2.7.0，需實測 |
| **C（備案）** | 不裝 contrib，自寫 masked PPO | 多 350–450 行；但零衝突且學習價值最高（masking 本身約 40 行） |

**在拍板前先做 A0**——A0 不需要任何新套件。

---

## 4. 已定案，不要重開

| 決策 | 理由 |
|------|------|
| **不用調 reward 權重修迴圈** | 根因是觀測空間存在 2-cycle（二值 path ＋ 步數不在觀測裡），與權重無關 |
| **物理非法動作用 masking，不用扣分** | 舊版 `-10` 在跟 `+1000` 搶尺度，而且「一直撞牆」本身就是確定性策略的不動點 |
| **速度用折扣因子 γ 表達，不用每步扣分** | 舊版每步 `-1` 累積到 `-72`，淹掉終局訊號 |
| **shaping 位能用「覆蓋率」不用「距離」** | 距離位能會懲罰「為填角落而暫時遠離下一個號碼」，而那正是解 Zip 必須做的 |
| **允許倒車是手段、一筆畫是目標** | 三階段課程逐步收緊，最後才 mask 已訪格 |
| **每個 episode 抽新題目** | 舊版只 overfit 單一地圖；泛化才是 RL 的價值 |
| **RL 不取代 CP-SAT** | 價值在攤提式推論（毫秒級前向）與學習方法本身 |

---

## 5. 階段任務

### A0 — 環境健全性（0.5 天）★ 先做，且不需要新套件

**這一步若當年存在，第一天就能排除「環境本身不可解」。**

- [ ] 寫 pytest：把 `generate_puzzle()` 回傳的 **ground-truth 解答路徑**逐步餵進現有 `PuzzleEnv`，斷言最後 `terminated=True` 且拿到成功獎勵
- [ ] 寫一支小腳本驗證 2-cycle 假說：讓 agent 在兩個已訪格間震盪，印出 `_get_obs()` 的 hash 是否只在兩個值間交替（報告 §2.1 的論證目前是**靜態程式碼分析，未實驗復現**）

**Done**：測試通過；2-cycle 有實驗證據（或推翻它——若推翻，立刻回報，整份設計要重審）。

### A1 — env v2（2–3 天）

新增 `src/core/rl/rl_env_v2.py`（**不要改壞舊的 `rl_env.py`**，留著對照）。

**觀測（padding 到 8×8）**：

| # | Channel | 值 |
|---|---------|-----|
| 0 | `valid_mask` | 1＝可走格（含 padding 與障礙處理） |
| 1 | `wall_right` | 1＝該格右側有牆 |
| 2 | `wall_down` | 1＝該格下方有牆 |
| 3 | **`visit_count`** | `min(v,4)/4` ← 打開 2-cycle 的關鍵 |
| 4 | `visit_recency` | `(t − 最後造訪步)/budget`，未訪為 1 |
| 5 | `agent_pos` | one-hot |
| 6 | `wp_next` | one-hot（下一個目標） |
| 7 | **`wp_future`** | 未收集的 waypoint 標 `k/N` ← 修「看不到未來 waypoint」 |
| 8 | `wp_done` | 已收集標 1 |

**全域純量（6 維）**：`coverage`、**`time_used = 已用步數/budget`（嚴格單調，數學上排除觀測循環）**、`wp_progress`、`revisit_ratio`、`last_action` one-hot(4)、`grid_shape`。

**動作與 mask**：4 方向；遮掉 ①出界 ②障礙 ③有牆 ④**順序錯的 waypoint 格** ⑤（僅 Phase 3）已訪格。
⚠ **邊界情況**：Phase 3 可能四個方向全被 mask（死路）→ 環境必須直接 `terminated=True`，**不要讓全 False 的 mask 進到取樣階段**（`MaskablePPO` 會出錯）。

**Reward**：

```
gamma = 0.99                  # 「越快越好」由折扣表達
success = +1.0                # 唯一大獎
failure（超時／死路） = 0.0
覆蓋 shaping（Phase 1-2）: F = lambda * (gamma * Phi(s') - Phi(s)),  Phi = 已訪格數/可走格數,  lambda = 0.5
重走成本: c * (visit_count_before - 1)   # c: 0 -> 0.005 -> 改用 mask
budget = k × 可走格數          # k: 3 -> 2 -> 1
```

- [ ] 實作 env v2 ＋ 單元測試（mask 正確性、全 mask 邊界、reward 邊界）
- [ ] 離線預生成 **50k 題**存成資料集（用 `generate_rl_dataset.py`），固定 seed 切 train/val/test
      ⚠ **出題器有 `timeout_per_attempt=20s`，絕對不要放在 rollout 迴圈裡**
- [ ] 量兩個 baseline：**masked random**（合法動作均勻亂選）與 **greedy**（永遠靠近下一個 waypoint）

**Done**：測試全綠；兩個 baseline 的成功率依尺寸落盤。**沒有 baseline 就不准進 A2**——否則無法證明模型學到東西。

### A2 — Phase 1「先學會走到終點」（1–2 天）

4×4 無牆、允許自由重走、budget 3×格數、shaping λ=0.5。
加**反向 curriculum**：出題器回傳 `(Puzzle, solution_path)`，讓 agent 從「解答倒數第 3 格」起步，成功率達標後往前推到倒數第 6、10 格……逐步逼近真正起點。**這是解決「從來沒有正樣本」的關鍵，成本近乎為零。**

**Done**：軟成功率 ≥ 90% 且**顯著高於 masked random**；學習曲線與 seed 落盤。

### A3 — Phase 2「學會少繞路」（2–3 天）

5×5→6×6 加牆；重走成本 `c` 從 0 線性升到 0.005；budget 2×格數；λ 0.5→0.2。權重**接續**A2，不重新初始化。

**Done**：軟成功率 ≥ 85% 且 `revisit_ratio < 0.2`。

### A4 — Phase 3「一筆畫」（2–4 天）

6×6→7×7 完整規則；**mask 已訪格**；budget＝格數−1；shaping 關掉（只剩 +1 與 γ）。

**Done**：在 1,000 題 held-out 上，**deterministic 的合法一筆畫通關率**明確優於兩個 baseline。
⚠ **評估一律 `deterministic=True`**——訓練期的高分不算數，這正是 2025 年被騙的地方。
⚠ **軟成功率與合法一筆畫率要分開報**：Phase 1/2 允許重走，成功路徑**不是合法的 Zip 解**。

### A5 — 落地（1 天）

- [ ] 掛成 API 第 10 種 solver（`src/app/routers/solver.py` 的 `SOLVERS` ＋ schema）
- [ ] 更新 `roadmap.md`、`dev_log.md`，出一份 `../reports/` 訓練報告

**Done**：`uv run pytest` 全綠；API 可選 RL solver；報告含四類指標與 baseline 對照。

### A6 —（選修）自寫 PPO 對拍 / AlphaZero-lite

有了 A4 這個「已知能成功」的參考點之後再造輪子，才學得到東西。與 `board-game-rl` 的 MCTS→AlphaZero 規劃共用學習投資。

---

## 6. 評估協定（每次都照這個報）

| 指標 | 說明 |
|------|------|
| **主**：deterministic 通關率 | 1,000 題 held-out，依尺寸分列；**同時報兩種推論模式**：(a) 允許重走 (b) 強制 mask 已訪格 ←(b) 才是產品可用數字 |
| 效率 | 平均步數／理論最少步數；`revisit_ratio`；合法一筆畫率 |
| 成本 | 單題決策延遲 vs DFS／CP-SAT 求解時間（RL 唯一可能贏的維度） |
| 對照 | masked random、greedy —— **每次都要一起報** |

---

## 7. 與 VLM track 的協作

| 面向 | 約定 |
|------|------|
| 程式碼 | 你動 `src/core/rl/`；VLM 動 `src/core/vl_models/`、`src/app/`、`src/ui/`。**A5 會動到 `src/app/routers/solver.py`，動之前先確認 VLM track 沒有同時在改** |
| 共用模組 | `src/core/utils.py`、`src/core/puzzle_generation/` **只讀不改**；真要改先提出 |
| 文件 | dev_log 各自加自己的 `## YYYY-MM-DD` 區塊；roadmap 只改自己那一項；衝突時兩邊都保留 |
| 相依 | `pyproject.toml`／`uv.lock` 序列化，由本人統一處理（你這條 track 的 §3 決策優先度較高，因為它擋住 A2） |
| Docker | 容器名與埠全機唯一，不要同時起 stack（你這條大多用不到 Docker） |

---

## 8. 回報與紅線

- 回報要有：**改了哪些檔**、實際指令與**輸出關鍵行**、文件是否更新、**逐項確認的 done 條件**。沒跑就說沒跑。
- **長時間訓練前先確認**（訓練是小時級）；訓練成品放 `models/`、資料放 `datasets/`，都不進版控。
- git：commit／push／PR 都需當次授權；不在 `main` 開發；禁 force push。
- **不永久刪除任何檔案**：移進 `../../../soft-delete/<時間戳>/<原相對路徑>` 並回報還原方式。
- 舊的 `rl_env.py`、`train_single_cnn_sb.py`、`models/dqn_*.pth` **保留當對照，不要刪、不要覆寫**。

---

## 9. 風險

1. **A0 推翻 2-cycle 假說**（低但影響大）：若實驗顯示觀測不循環，整份設計的前提要重審 → 立刻回報。
2. **相依決策拖延**（中）：A2 之後就需要 MaskablePPO；若遲遲未拍板，改走選項 C 自寫。
3. **Phase 3 死路率過高**（中）：一筆畫的 mask 會讓失敗變常態 → 靠 curriculum 接續訓練與反向 curriculum 緩解；必要時在 Phase 2.5 加入「部分覆蓋給部分分數」的終局獎勵。
4. **CPU 爭用**（低）：VLM track 的資料生成也吃多核，兩邊錯開跑。
5. **超參未經調校**（確定）：報告給的 PPO 超參與 λ／c／γ 只是合理起點，不是最佳值。
