# 任務計畫 — RL track 收尾：多尺寸模型 ＋ Docker 一鍵啟動 ＋ A5 掛 API

> 建立於 2026-09-11（Asia/Taipei）｜分支 `feat/rl-a2-training`｜worktree `zip-rl`
> 本檔是**這次收尾任務的正本**。session 中斷後接手：先讀本檔 §0 現況，再看 §5 檢查點。
> 背景與既有結論一律去 [`../handover-rl-solver.md`](../handover-rl-solver.md)，本檔不重複。

---

## 0. 現況快照（每完成一步就更新這一節）

| 項目 | 狀態 |
|---|---|
| ruff 版本統一 | ✅ commit `fccddc2` |
| ② 6×6 best-of-32／64 | ✅ 0.8500／0.8835。**判定基準定為 best-of-32** |
| T1.1 4/5/6 資料集 | ✅ `seed20300000_n20000_456`（11m16s）|
| T1.2 `Goal` 支援多尺寸 | ✅ `sizes: tuple` ＋ `board_label` ＋ 5 個測試 |
| T1.3 多尺寸模型 | ✅ `bc_multi_456`，465.3s |
| T1.5 對照組 ＋ ⓪ value-coef | ✅ 四臂全跑完；value-coef 差 −0.0083（雜訊內）|
| T1.4 三尺寸 best-of-32 | ✅ 六個組合全部落盤在 `logs/rl_probes/` |
| T2 Docker 一鍵啟動 | ✅ 四個缺陷修完，新增 `start.py`，五項驗收實測通過 |
| T3 A5：RL solver 掛 API | ✅ 上線，三個尺寸都解得出來，7×7 回 400 |
| T4 文件與結案報告 | ✅ 報告、notes（4 份）、部署指南、兩份 README、roadmap／dev_log／handover／AGENTS |
| T5 refactor：solver registry | ✅ 三份重複清單收成一份 ＋ 4 個測試 |

**全部檢查點（§5）已完成。剩下的是 commit（需當次授權）。**
**下一個 session 的主線**：BC → PPO 微調（`--init-from` 尚未實作）。

---|---|
| ruff 版本統一 | ✅ 已 commit `fccddc2` |
| ② 6×6 best-of-32／64 | ✅ 已量完，見 §1。**判定基準定為 best-of-32**（本人 2026-09-11 決定）|
| T1.1 4/5/6 資料集 | ✅ `seed20300000_n20000_456`（11m16s）。4×4 15,439／5×5 15,996／6×6 16,000 train |
| T1.2 `Goal` 支援多尺寸 | ✅ `sizes: tuple`、`board_label`、5 個新測試；新增 `goal3_multi` ＋ 3 個單尺寸對照 goal |
| T1.3 訓練多尺寸模型 | 🚧 執行中（`bc_multi_456`，44s／epoch）|
| T1.5 對照組 ＋ ⓪ value-coef | 🚧 排在同一條訓練鏈（`bc_ctrl_4x4`／`_valuefree`／`_5x5`／`_6x6`）|
| T1.4 三尺寸 best-of-32 評估 | ⬜ 等訓練完 |
| T2 Docker 一鍵啟動 | 🚧 檔案已改完（見 §2 實際發現），prod image 建置中 |
| T3 A5：RL solver 掛 API | ✅ `solver_service.py` ＋ router ＋ 6 個測試。**測試 261 → 272 passed** |
| T4 文件與結案報告 | 🚧 `notes/` 已建（4 份），報告與部署指南待寫 |

---

## 1. ② 的結果（2026-09-11 實測，這次收尾的判定依據）

`bc_6x6` × 6×6 held-out test（2,000 題）× `--max-attempts 64`：

| N | 1 | 2 | 4 | 8 | 16 | **32** | **64** |
|---|---|---|---|---|---|---|---|
| solve | 0.4285 | 0.5600 | 0.6630 | 0.7460 | 0.8105 | **0.8500** | **0.8835** |
| 嘗試／題 | 1.00 | 1.57 | 2.39 | 3.59 | 5.35 | 8.05 | 12.25 |

deterministic 0.4620（與已發表值逐位元相同，確認評估路徑等價）。耗時 664.6s。

**判讀**：

- **best-of-32 ＝ 0.8500，剛好等於門檻**（1,700／2,000），**best-of-64 ＝ 0.8835 才有餘裕**。
- ⚠ **best-of-N 的評估雜訊約 ±0.01**：同一個模型、同一個測試集，只因為抽樣 rng 串流不同
  （前一次 `--max-attempts 16` 在第 16 次就停，這次跑到 64，下游題目拿到的 seed 因而不同），
  N=16 就從 **0.8000 變成 0.8105**。⇒ **0.8500 是踩線，不能當成穩定達標**；
  **可以宣稱達標的是 best-of-64。**
- 事前預測（hazard 0.035 → N=32 約 0.887）**偏樂觀**，實際 hazard 繼續衰減，落在 0.015–0.025 情境。
  這條 track 第三次證明「外推不可信」。

⇒ **goal2_6x6 在推論預算 N=64（平均 12.25 次／題、約 0.5 秒／題）下達標。**

---

## 2. 這次要做的四條線

### T1 — 多尺寸模型（本人 2026-09-11 授權「確定要做」）

**目標**：一個模型同時吃 4×4／5×5／6×6，best-of-32 判定。

| 步驟 | 內容 | 估時 |
|---|---|---|
| T1.1 | 生資料集 `--count 20000 --sizes 4,5,6 --name <新名>`（現行包只有 4 和 6；5×5 只存在於舊小包）| **11 分**（機器）|
| T1.2 | `Goal.size: int` → 多尺寸。`goal.size` 全專案只用在 3 處：`train_maskable_ppo.py:410` 過濾、`:642` log、`train_behaviour_cloning.py:357` metadata。**env 不用改**（`_load_sample()` 每局重讀 height／width，觀測固定 8×8，已讀碼確認）。新增 `goal3_multi` ＋ 混合尺寸不變量測試 | **30–45 分** |
| T1.3 | BC 訓練 `goal3_multi`（約 117 萬個 (state, action) 對，是現行 6×6 的 2.1 倍）| **8–9 分**（機器）|
| T1.4 | 三個尺寸各跑 `probe_cross_size.py --max-attempts 64` | **40 分**（機器）|
| T1.5 | 對照組：單尺寸 5×5 BC（否則無法判斷混合是變好還是變差）| 訓 4 分 ＋ 評 10 分 |

**done 條件**：
1. 一張表：混合模型 × {4,5,6} × {det, best-of-32, best-of-64}，對照 `bc_4x4`／`bc_6x6`／單尺寸 5×5。
2. 明確回答「一個模型能不能取代三個」，並標出哪些差異在 ±0.04 雜訊內。
3. 6×6 沒有變差（若變差 > 0.04，記錄並保留兩個模型的方案）。

**已知風險（要先寫下來，不能事後補）**：混合尺寸**很可能改善 4×4／5×5**（`bc_6x6` 沒看過 4×4
就有 0.546），但**沒有機制理由能改善 6×6**——6×6 的瓶頸是長程規劃（單步 93.9% → 需 98.9%），
更小的盤面不提供長程訊號。也可能因容量攤薄而略降。**6×6 的 seed 雜訊從沒量過。**

### T2 — Docker 一鍵啟動既有服務

**目前讀碼發現的兩個缺陷（尚未實測，跑之前不要當成結論）**：

1. `.devcontainer/Dockerfile`（prod）**沒有 `CMD`**，`docker-compose.yml` 也沒有 `command:`
   ⇒ `docker compose up` 會建好 image 但**不會啟動 app**。
2. `run_docker_dev.py` 第 3 步用 `docker compose exec`（**沒有 `-d`**）跑 `python -m src.app.main`，
   `subprocess.run(check=True)` 會一直等 ⇒ **後面的 healthcheck 永遠跑不到**。

| 步驟 | 內容 |
|---|---|
| T2.1 | 先實測現況：`docker compose up -d` 後容器狀態與 `/api/echo/health`，確認上面兩點 |
| T2.2 | Dockerfile 加 `CMD`（uvicorn），compose 補 `healthcheck` |
| T2.3 | `models/` 掛唯讀 volume 進 app 容器（RL checkpoint 不進 image；`models/` 不進版控）|
| T2.4 | `run_docker_dev.py` 改成「起服務 → 等 health → 印網址」，不再自己 exec 前景程序 |
| T2.5 | app 容器**不要求 GPU**（RL 推論走 CPU 即可；ollama 才需要 GPU）|

**done 條件**：`docker compose -f docker-compose.dev.yml up -d` 之後，
① `/api/echo/health` 回 200；② `/ui` 開得起來；③ `/api/solver/solve` 三種傳統 solver 可用；
④ `/api/vision/solve` 能打到 ollama（模型已在 volume 裡：`zip-qwen35-4b-p4c`、`qwen3.5`、`qwen2.5vl`、`gemma4`）；
⑤ `/api/solver/solve` 的 RL solver 可用。**每一項都要貼實際輸出。**

### T3 — A5：RL solver 掛進 API

| 步驟 | 內容 |
|---|---|
| T3.1 | 新增 `src/core/rl/solver_service.py`：lazy 載 checkpoint（含快取）、`puzzle_data` → `PuzzleSample`、rollout（deterministic ＋ best-of-N）、回傳 `solution_path` |
| T3.2 | `routers/solver.py` 的 `SOLVERS` 加一項；**模型檔不存在 → 503**、**尺寸沒有對應模型 → 400**，不要讓它變 500 |
| T3.3 | 測試：有模型／無模型／不支援尺寸／解不出來 四條路徑 |

**⚠ 跨 track**：`src/app/routers/solver.py` 依協作約定屬 VLM track。
2026-09-11 實測 `zip-vlm` worktree 只有 `ai-collab/commands.txt` 是 dirty，
**沒有人在改 `src/app/`** ⇒ 現在動是安全的。

**⚠ 分布外**：RL 訓練資料的牆是 0 或 2–5 道，VLM 從真實截圖讀出來的可到 10+ 道 ⇒
RL solver 對 vision 來的題目要標示為實驗性。

### T4 — 文件

| 產出 | 內容 |
|---|---|
| `reports/2026-09-11_rl-wrap-up.md` | 4／5／6 能力總表、best-of-N 成本、瓶頸的定量說明、跨尺寸矩陣、「這題該不該用 RL」的結論 |
| `ai-collab/deployment-guide.md` | Docker 怎麼起、每個服務是什麼、埠、常見失敗排查 |
| `roadmap.md`／`dev_log.md`／`handover-rl-solver.md` | 現況、做了什麼、接手要知道的 |
| `README.md` | 加一節「Run everything with Docker」（英文）|

---

## 3. 執行順序與資源

**不要同時跑**：資料生成（16 workers）、BC 訓練（GPU＋9 執行緒）、長評估。本機上限 75%。
Docker build 主要吃網路與 IO，可以和資料生成並行。

```
T1.1 生資料（背景 11 分）
  └─ 同時做 T3（寫 RL solver service，純寫碼）
T1.2 改 Goal ＋ 測試
T1.3 訓練（背景 9 分）
  └─ 同時做 T2（Docker，會 build image）
T1.4 評估（背景 40 分）
  └─ 同時做 T4（文件）
最後：跑完整測試 ＋ pre-commit ＋ commit（需當次授權）
```

**估計**：機器時間約 1 小時 40 分（多數可背景）；需要人盯的工作約 2–3 小時。
⇒ **不會在一個 session 內做完，§5 的檢查點就是交接點。**

---

## 4. 「6×6 要怎麼訓練才可能真的變好」——給未來的自己

已量到的事實：6×6 每局約 **14.22 次真正的選擇**，單步 93.9%，`0.939^14.22 = 0.4086`
（實測 0.4095）。要 0.85 需要單步 **98.86%**，即**錯誤率砍 5.4 倍**。
而完美的**一步**前瞻 oracle 只值 +0.0285 ⇒ **缺的是長程規劃，不是單步感知**。

在這個診斷下，**有機制理由**的訓練方向只有三類（都沒做）：

1. **搜尋進訓練迴圈（AlphaZero 式）**：用 MCTS／beam search 產生比策略本身更強的
   target，再把它蒸餾回策略。這正是「策略單步不夠準、但搜尋能補」的標準解，
   而本專案已經量到搜尋確實有效（best-of-64 把 0.462 變成 0.8835）。
   ⇒ **把推論期已經買到的東西搬進訓練期。**
2. **DAgger**：BC 只看得到專家軌跡上的狀態，一走偏就沒有訓練訊號；DAgger 讓策略自己走、
   在它走到的狀態上問專家（這裡的專家是現成的 CP-SAT solver，免費且完美）。
   直接對應「compounding error」這個已知限制。
3. **BC → PPO 微調**（handover 的①）：唯一能推翻「這題不該用 RL」的實驗。

**沒有機制理由的**（已被證偽或被上界關掉）：加資料、加拓撲特徵、割點、GNN、調 PPO 超參、
單純加深網路、單純把 PPO 推到全長。

---

## 5. 檢查點（交接就從這裡接）

- [ ] C1：新資料集產生完成，manifest 的 `sizes` 是 `[4,5,6]`，digest 記錄在 dev_log
- [ ] C2：`goal3_multi` 可訓練，混合尺寸測試通過，`pytest` 全綠
- [ ] C3：多尺寸模型三個尺寸的 best-of-64 數字落盤在 `logs/rl_probes/`
- [ ] C4：`docker compose up -d` 後五項 done 條件逐項貼輸出
- [ ] C5：RL solver 在 `/api/solver/solve` 回得出答案（貼 curl 輸出）
- [ ] C6：四份文件更新完成
- [ ] C7：pre-commit 綠 ＋ 測試綠 ＋ commit（**需當次授權**）
