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
| A | 升 `torch` 到 ≥2.8（換 cu124/cu126 index），SB3 與 sb3-contrib 一起上 2.9 | 動到刻意的 pin（`AGENTS.md §5`），要重跑 46 測試回歸 |
| **B（2026-08-15 查證後改為推薦）** | `uv add sb3-contrib==2.7.1`，**完全不動 torch** | 停在 contrib 2.7.x |
| C（備案） | 不裝 contrib，自寫 masked PPO | 多 350–450 行；學習價值最高，但計畫書 A6 已排在 A4 之後做（要先有能動的參考點） |

### ✅ 2026-08-15 查證：選項 B 可行（原本標「未確認、需實測」）

實際查 PyPI metadata（非記憶；`uv run python` 直接讀 JSON API，並下載 wheel 檢查內容）：

| 套件 | 版本 | 發佈日 | requires |
|------|------|--------|----------|
| `sb3-contrib` | **2.7.1** | 2025-12-05 | `stable_baselines3>=2.7.0,<3.0` |
| `sb3-contrib` | 2.8.0 | 2026-04-01 | `stable_baselines3>=2.8.0,<3.0` |
| `sb3-contrib` | 2.9.0（最新） | 2026-06-15 | `stable_baselines3>=2.9.0,<3.0` |
| `stable-baselines3` | 2.7.0（本專案 lock 的版本） | — | `torch>=2.3,<3.0` ← **torch 2.4.1 滿足** |
| `stable-baselines3` | 2.8.0 | — | `torch>=2.3,<3.0` ← 也滿足 |
| `stable-baselines3` | 2.9.0 | — | `torch>=2.8,<3.0` ← 這一版才衝突 |

`sb3-contrib==2.7.1` 的 wheel 內含 `sb3_contrib/ppo_mask/ppo_mask.py` 與 `sb3_contrib/common/maskable/`
（buffers／distributions／policies／evaluation／callbacks），頂層 `__init__.py` 有 `from sb3_contrib.ppo_mask import MaskablePPO`。
**結論：衝突只存在於 SB3 2.9；停在 2.7/2.8 完全不必動 `torch` pin。**

### ✅ 2026-08-15 已安裝（本人當次授權後由 agent 執行）

```powershell
cd D:\it_project\github_sync\zip-rl\linkedin-zip-challenge
uv add sb3-contrib==2.7.1 --index-strategy unsafe-best-match
```

⚠ **`--index-strategy` 這個旗標是必要的**：專案把 `index-strategy = "unsafe-best-match"` 寫在 `[tool.uv.pip]`，
那只對 `uv pip` 生效，`uv add`／`uv lock` 不吃，少了旗標會解不動（`requests` 在 cu121 index 上版本不夠新）。

安裝後實測確認：`torch 2.4.1+cu121` 與 `stable-baselines3 2.7.0` **都沒有被動到**，
`from sb3_contrib import MaskablePPO` 可用，`uv run pytest` 全綠。變動只有 `pyproject.toml` ＋ `uv.lock` ＋ 一個新套件。

---

## 4. 已定案，不要重開

| 決策 | 理由 |
|------|------|
| **不用調 reward 權重修迴圈** | 根因是觀測空間存在 2-cycle（二值 path ＋ 步數不在觀測裡），與權重無關 |
| **物理非法動作用 masking，不用扣分** | 舊版 `-10` 在跟 `+1000` 搶尺度，而且「一直撞牆」本身就是確定性策略的不動點 |
| **速度用折扣因子 γ 表達，不用每步扣分** | 舊版每步 `-1` 累積到 `-72`，淹掉終局訊號 |
| **shaping 位能用「覆蓋率」不用「距離」** | 距離位能會懲罰「為填角落而暫時遠離下一個號碼」，而那正是解 Zip 必須做的 |
| **★ 全程一筆畫，倒車不開放**（2026-08-15 修訂） | 見下方修訂說明 |
| **每個 episode 抽新題目** | 舊版只 overfit 單一地圖；泛化才是 RL 的價值 |
| **RL 不取代 CP-SAT** | 價值在攤提式推論（毫秒級前向）與學習方法本身 |

### ★ 2026-08-15 修訂：從「三階段放寬→收緊」改為「全程一筆畫 ＋ 反向 curriculum」

**原案**（保留備查）：「允許倒車是手段、一筆畫是目標」，Phase 1 自由重走 → Phase 2 重走收費 → Phase 3 才 mask 已訪格。

**改動理由**（本人 2026-08-15 拍板）：

1. **一筆畫在構造上必定可解**：出題器是先畫 Hamiltonian path 再挖題，加牆時只從「不在解答路徑上的邊」挑
   （`src/core/puzzle_generation/puzzle_generator.py:44-58`、`:117-121`），所以每題至少有一條合法一筆畫解。
2. **禁止重踩 ⇒ 2-cycle 在定義上不可能發生**（要形成 A↔B 迴圈必須重踩已訪格）。
   於是原案為了打開 2-cycle 而設的 `visit_count`、`visit_recency` 兩個 channel **失去存在理由，直接砍掉**。
3. **原案的 Phase 1/2「成功」不是合法 Zip 解**，等於先訓練一個不是目標的任務，再靠 Phase 3 掰回來（distribution shift）。
   A0 才剛實測到「環境獎勵的東西不是遊戲規則」有多致命（見 [reports/2026-08-15_a0-env-v1-findings.md](../reports/2026-08-15_a0-env-v1-findings.md)）。
4. **稀疏訊號改由反向 curriculum 解**（原案 §4.6 本來就有這個機制，與一筆畫完全相容）：
   從「解答倒數第 k 格」起步，成功率達標後把 k 往前推。兩個機制原本重疊，保留成本較低的那個。
5. 少掉三條 anneal 排程（λ、重走成本 `c`、budget），超參數量下降。

**保留的備案**：若反向 curriculum 把起點推遠後成功率崩掉（死路率過高），**先試**「部分覆蓋給部分分數」的終局獎勵（§9.3），
**再考慮**回頭開放倒車。是否啟用由 A1 量到的 baseline 數據決定，不是憑感覺。

---

## 5. 階段任務

### A0 — 環境健全性（0.5 天）★ 先做，且不需要新套件

**這一步若當年存在，第一天就能排除「環境本身不可解」。**

- [ ] 寫 pytest：把 `generate_puzzle()` 回傳的 **ground-truth 解答路徑**逐步餵進現有 `PuzzleEnv`，斷言最後 `terminated=True` 且拿到成功獎勵
- [ ] 寫一支小腳本驗證 2-cycle 假說：讓 agent 在兩個已訪格間震盪，印出 `_get_obs()` 的 hash 是否只在兩個值間交替（報告 §2.1 的論證目前是**靜態程式碼分析，未實驗復現**）

**Done**：測試通過；2-cycle 有實驗證據（或推翻它——若推翻，立刻回報，整份設計要重審）。

### A1 — env v2（2–3 天）

新增 `src/core/rl/rl_env_v2.py`（**不要改壞舊的 `rl_env.py`**，留著對照）。

**觀測（padding 到 8×8）** — 2026-08-15 修訂版，一筆畫下不需要重走相關的 channel：

| # | Channel | 值 |
|---|---------|-----|
| 0 | `valid_mask` | 1＝可走格（含 padding 與障礙處理） |
| 1 | `wall_right` | 1＝該格右側有牆 |
| 2 | `wall_down` | 1＝該格下方有牆 |
| 3 | `visited` | 1＝已走過（一筆畫下不可能 >1，故二值即可；action mask 只看四鄰，policy 仍需要全域已訪圖來規劃） |
| 4 | `agent_pos` | one-hot |
| 5 | `wp_next` | one-hot（下一個目標） |
| 6 | **`wp_future`** | 未收集的 waypoint 標 `k/N` ← 修「看不到未來 waypoint」 |
| 7 | `wp_done` | 已收集標 1 |

> ~~`visit_count`／`visit_recency`~~ **已刪**：它們的唯一用途是打開重走造成的 2-cycle，而一筆畫下重走不存在。

**全域純量**：`coverage = 已訪格/可走格`（一筆畫下等同「已用步數比例」，嚴格單調）、`wp_progress`、`last_action` one-hot(4)、`grid_shape`(2)。
~~`revisit_ratio`~~ 已刪（恆為 0）；~~`time_used`~~ 與 `coverage` 在一筆畫下是同一個量的仿射變換，只留一個。

**動作與 mask**：4 方向；遮掉 ①出界 ②障礙 ③有牆 ④**順序錯的 waypoint 格** ⑤**已訪格（全程都遮，不再分階段）**。
⚠ **邊界情況**：四個方向全被 mask（死路）→ 環境必須直接 `terminated=True`，**不要讓全 False 的 mask 進到取樣階段**（`MaskablePPO` 會出錯）。
一筆畫下死路是**常態**而非例外，這條邊界測試是 A1 的必要項。

**Reward**：

```
gamma = 0.99                  # 「越快越好」由折扣表達
success = +1.0                # 唯一大獎（走完全部可走格且依序收完 waypoint）
failure（死路／超時） = 0.0
覆蓋 shaping: F = lambda * (gamma * Phi(s') - Phi(s)),  Phi = 已訪格數/可走格數,  lambda 預設 0.2
budget = 可走格數 - 1          # 一筆畫的理論步數，不再分階段放寬
```

> 一筆畫下每步必定新增一格，所以 Φ 是「時間的嚴格遞增函數」，這個 shaping 實質上是**存活獎勵**
> （活得越久＝覆蓋越多）。potential-based 形式保證不改變最優策略集合，但 λ 太大會讓「苟活」蓋過「完成」，
> 故預設值調低到 0.2。**λ 未經調校**，A2 要做敏感度檢查。

- [ ] 實作 env v2 ＋ 單元測試（mask 正確性、全 mask 邊界、reward 邊界）
- [ ] 離線預生成 **50k 題**存成資料集（用 `generate_rl_dataset.py`），固定 seed 切 train/val/test
      ⚠ **出題器有 `timeout_per_attempt=20s`，絕對不要放在 rollout 迴圈裡**
- [ ] 量兩個 baseline：**masked random**（合法動作均勻亂選）與 **greedy**（永遠靠近下一個 waypoint）

**Done**：測試全綠；兩個 baseline 的成功率依尺寸落盤。**沒有 baseline 就不准進 A2**——否則無法證明模型學到東西。

> **三個 stage 全程都是一筆畫**（已訪格永遠被 mask）。curriculum 的兩個軸改成
> **①「起點離終點多遠」（反向 curriculum）② 盤面尺寸與約束**，不再放寬遊戲規則。
> 因此**每一次「成功」都是一條合法的 Zip 解**，訓練指標與產品指標從第一天就對齊，
> 不再需要「軟成功率 vs 合法一筆畫率」兩套帳。

### A2 — Stage 1「反向 curriculum 起步」（1–2 天）

4×4 無牆。出題器回傳 `(Puzzle, solution_path)`，**答案本來就在手上**：讓 agent 從「解答倒數第 k 格」起步
（該處之前的格子預先標成已訪），k 從 3 開始，成功率達標後推到 6、10……逐步逼近真正起點 k＝全長。
**這是解決「從來沒有正樣本」的關鍵，成本近乎為零。**

- [ ] k 的推進要**自動化**（達標才升級）並記錄每個 k 的成功率曲線
- [ ] 記錄**死路失敗率**隨 k 的變化——這是判斷要不要啟用備案（§4 修訂說明）的依據

**Done**：k 推到全長時通關率 ≥ 90% 且**顯著高於 masked random**；各 k 的學習曲線與 seed 落盤。

### A3 — Stage 2「加牆與尺寸」（2–3 天）

5×5→6×6，加牆與障礙。權重**接續** A2，不重新初始化；反向 curriculum 的 k 在每次升尺寸時可回退再推。

**Done**：通關率 ≥ 85%（deterministic，合法一筆畫）；死路率落盤。

### A4 — Stage 3「完整尺寸與泛化」（2–4 天）

6×6→7×7 完整規則。shaping λ 可退到 0（只剩 +1 與 γ），驗證策略不依賴 shaping。

**Done**：在 1,000 題 held-out 上，**deterministic 的合法一筆畫通關率**明確優於兩個 baseline。
⚠ **評估一律 `deterministic=True`**——訓練期的高分不算數，這正是 2025 年被騙的地方。

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
| **主**：deterministic 通關率 | 1,000 題 held-out，依尺寸分列。全程一筆畫（2026-08-15 修訂）⇒ **只有一種推論模式**，通關率即產品可用數字，不再需要「軟成功率／合法率」兩套帳 |
| 失敗歸因 | **死路率**（四方向全被 mask）vs **超時率**——一筆畫下死路是主要失敗模式，要分開看 |
| 效率 | 平均步數／理論最少步數（一筆畫下成功即等於最少步數，故此欄主要看失敗局走多遠） |
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
