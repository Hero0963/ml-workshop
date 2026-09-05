# 交接文件 — RL Track（一筆畫 solver）

> **接手這條 track 的 agent／developer 從這一份開始讀，讀完就能動手。**
> 最後更新：2026-09-05（Asia/Taipei）｜分支 `feat/rl-a2-training`｜worktree `zip-rl`｜對應 roadmap 第 3 項
> ⚠ **要讀就讀 `zip-rl` 這份**：本檔在每個 worktree 都有一份，別的 worktree 拿到的是那條分支上次 commit 的版本
> （`zip-vlm` 的副本停在 2026-08-15，還在說「A2 尚未開始」）。
> 其他文件是延伸閱讀，本檔會標明什麼時候該去翻哪一份。
> 姊妹 track 的交接文件：[`handover-vlm-parser.md`](handover-vlm-parser.md)

---

## 0. 一句話現況

**A0／A1／A2 都跑完了，兩個瓶頸假說也依序被實驗證實：「在背答案」加資料解掉了，「預算不夠」加預算也真的推得動。**
**兩個 goal 仍未達門檻**，而現在卡的是第三個問題——**推到全長的成本是小時級，值不值得投**。

> ★ **接手前一定要知道的七件事：**
>
> **① 三輪訓練都做完了**，這是完整成績（held-out test、`deterministic=True`）：
>
> | goal | 資料集 | 步數 | curriculum | train | test | 落差 | 門檻 |
> |---|---|---|---|---|---|---|---|
> | 4×4 | A2（1,360） | 1M | 全長 | 0.950 | 0.788 | +0.162 | 0.90 ❌ |
> | 4×4 | **新（15,419）** | 1M | 全長 | 0.927 | **0.877** | **+0.050** | 0.90 ❌ |
> | 6×6 | A2（1,360） | 5M | k=33/36 | 0.502 | 0.253 | +0.250 | 0.85 ❌ |
> | 6×6 | **新（16,000）** | 5M | k=27/36 | 0.352 | **0.344** | **+0.009** | 0.85 ❌ |
> | 6×6 | 新（16,000） | **8M**（續訓） | **k=30/36** | — | **0.409** | — | 0.85 ❌ |
>
> 對照組 4×4 greedy 0.117／random 0.089、6×6 greedy 0.004／random 0.000，**跨三輪完全相同**
> ⇒ 評估協定是對齊的，這些數字可以跨輪比較。
> 2026-09-05 那一輪的完整分析在
> [`reports/2026-09-05_rl-budget-and-design-notes.md`](reports/2026-09-05_rl-budget-and-design-notes.md)。
>
> **② 落差收斂的形狀是對的**：held-out 上升、**訓練集反而下降**（4×4 0.950→0.927、6×6 0.502→0.352），
> 那是「停止背答案」的特徵，不是運氣。難度對照也做過（兩個沒學過任何一邊的 baseline 打同一組
> train／test，四項全顯示 held-out 一樣難或略易），所以「test 比較難」已排除。
>
> **③ ✅ 「加預算有效」已驗證（2026-09-05），但代價在指數上升。** 續訓 3M 步（5M → 8,011,776，749.5s）：
> curriculum **k=27 → 30**、held-out **0.344 → 0.409**、死路率 0.656 → 0.591。
> 上一輪在 k=27 卡了 2.3M 步沒過門檻，續訓後**56.9 萬步就晉級** ⇒ 當時的判讀是對的。
>
> **但每級晉級成本約以 ×2 成長**：812,944 → 1,307,280 → **2,867,776**。
> k=30 已花 2,436,944 步未晉級，而其 rollout 成功率**仍單調爬升**（0.679 → **0.764**，門檻 0.90），
> 形狀與 k=27 晉級前完全相同 ⇒ **不是停滯，是還沒到**。
> 外推推到全長還要 **15–30M 步 ≈ 1–2 小時**——⚠ **這個外推不可信**（§3.8：上一輪同樣推法差了三倍），
> 只當量級參考。**還沒排除的替代解釋**：k 越大越接近整盤，可能存在學不動的天花板 ⇒ 成本發散而非 ×2。
> **最便宜的判別證據是下一級 30→33 實際花多少**（落在 4–8M 內則 ×2 模型成立）。
>
> ⚠ **不要把 curriculum 的 rollout 成功率當能力指標**——它疊了三層有利條件：起點是
> **k（前面已預先走好）**、題目是**訓練集**、策略是**隨機取樣**。
> 同一個模型從真正起點、在 held-out、deterministic 只有 **0.409**。
> 那個 0.764 唯一的用途是**決定 curriculum 要不要晉級**與**判斷加預算有沒有用**。
> **拿它當成績就是 2025 年那次「被訓練期高分騙過」的翻版。**
>
> **④ ⚠ 有一個沒人記錄的設計偏離**：restart 報告的階段表指定 shaping λ 在**一筆畫階段是 0**
> （「只剩 +1 與 γ」），而「全程一筆畫」等於全程都是那個階段——但 `PuzzleEnvV2` 一直是
> `shaping_lambda=0.2`，實算佔一局總分 **14%**，**從來沒有關掉跑過**。一次對照就有答案。
>
> **⑤ 門檻（0.90／0.85）不是硬標準**，見 §7.20：它沒有推導、繼承自已廢棄的設計，而且**沒指定推論模式**
> ——同一個模型 deterministic 0.870、best-of-2 0.903。**報告時兩個數字都要給。**
>
> **⑥ 改設定只改 `src/core/rl/train_config.py`。** goal（盤面／牆策略／步數預算／done 門檻）、PPO 超參、
> 網路、curriculum、資源上限全在那一個檔。訓練腳本只負責執行一個 goal。
> **資料集現在預設是 `seed20300000_n20000_4-6`；要重現 A2 的數字加 `--dataset main_n1700_456`。**
>
> **⑦ ★★ 這條 track 的產出是「做中學」，不是指標。** 2026-09-05 本人明確定案
> （restart plan §8 早就寫過：「不是為了比 CP-SAT 快或準（不會贏），是為了學 masking／PPO／curriculum／MCTS／RLVR」）。
> **這會改變「什麼叫做完」**：沒達標不等於失敗，**「知道為什麼沒達標」本身就是產出**；
> 反過來，為了衝分數而犧牲可解釋性（同時改多個變數、只留贏的那次、拿訓練期高分當成績）**與目標相反**。
> ⇒ **設計理由與機制解釋要落盤進 `reports/`，不能只留在對話裡**——它和數字同等是交付物。
> 已落盤的一份：[`reports/2026-09-05_rl-budget-and-design-notes.md`](reports/2026-09-05_rl-budget-and-design-notes.md) §5
> （8 個 channel 就是 AlphaGo 式的 feature planes、為什麼選 MaskablePPO、best-of-N 為何就是「policy 引導搜尋」的雛形、
> 以及圍棋與 Zip 的兩個本質差異）。

A0 的結論仍然是整條 track 的前提：**2025-10 的舊環境不是「難學」，是「餵標準答案也不會過關」**，
而且它的獎勵與 Zip 規則反相關。所以 env v2 是重寫，不是修補。

---

## 1. 開工前照這個順序讀

| 順序 | 檔案 | 什麼時候讀 |
|---|---|---|
| 1 | **本檔** | 一定 |
| 2 | [`plans/2026-08-15_track-rl-solver.md`](plans/2026-08-15_track-rl-solver.md) | 一定。作戰計畫：分階段 done 條件、協作約定、紅線。**§4 有 2026-08-15 的 curriculum 修訂說明** |
| 3 | [`reports/2026-08-15_a0-env-v1-findings.md`](reports/2026-08-15_a0-env-v1-findings.md) | 一定。舊環境為什麼不能用、實驗數據、對 v2 設計的具體要求 |
| 4 | [`reports/2026-08-15_rl-restart-plan.html`](reports/2026-08-15_rl-restart-plan.html)（瀏覽器開） | 要調超參、改觀測或考慮路線 B 才讀。§4.8 有 PPO 起手超參，§7 是 GRPO 路線 |
| 5 | [`roadmap.md`](roadmap.md) 的「已定案不要再重開的決策」表 | 一定，快速掃過 |
| 6 | [`../AGENTS.md`](../AGENTS.md) | 一定。子專案規範正本（venv、驗證、紅線、回報格式） |
| 7 | `dev_log.md` 的 RL 區塊：`## 2026-08-15`（A0／A1）、`## 2026-08-29`（A2）、`## 2026-09-05`（續訓） | 想看做了什麼、量到什麼時翻 |
| 8 | [`reports/2026-09-05_rl-budget-and-design-notes.md`](reports/2026-09-05_rl-budget-and-design-notes.md) | 最新一輪的數字在 §1–3；**§5 是設計筆記**（8 channel 與 AlphaGo feature planes、為何選 MaskablePPO、best-of-N 與搜尋），想搞懂「為什麼這樣設計」就讀它 |

> ⚠ **不要整份讀 `dev_log.md`**（**1,977 行**，2026-09-05 實測），用日期或關鍵字搜。

### 這條 track 的工作守則（都是實際犯錯後定下來的，照做）

1. **本機資源最多用到 75%**（本人常設要求：「讓我還能做別的事」）。上限要涵蓋**每一個會開工作的地方**——
   訓練的 `torch.set_num_threads`／GPU 記憶體，**以及資料生成的 `multiprocessing.Pool`**。
   預算只寫在 `train_config.DEFAULT_CPU_FRACTION` 一處。⚠「N% 的核心數」不等於「N% CPU」，
   **要在工作跑的時候實際取樣過才算設好**（§7.13、§7.14）。
2. **下結論前先跑掉能推翻它的對照。** 例：訓練集／held-out 的落差要先用「沒學過任何一邊」的 baseline
   證明兩邊難度相同，才能叫泛化問題（§7.15）。**還沒排除的要明講。**
3. **進行中的 log 不能當結論。** 曲線還在跑就外推，本次錯了兩次（§7 開頭那兩則）。
4. **長時間或吃資源的工作開跑前先問本人**；跑到一半發現超標，**先停再修**，不要跑完再說。
5. **告一段落就更新文件再 commit**（本人的節奏）：`dev_log.md` 記做了什麼與量到什麼、
   `roadmap.md` 記現況與下一步、本檔記接手要知道的、助理 memory 記跨 session 的教訓。
   **不要只留在對話裡。** commit 需要本人當次授權，單獨說 commit **不含** push。
6. **資料集用 digest 認，不用指令認**（§7 與 `generate_dataset_v2` 的 docstring）。新資料集的
   `--base-seed` 要離既有的夠遠，否則只是同一亂數流位移。
7. **★ 解釋要跟數字一起交付**（2026-09-05 定案，見 §0 ⑦）。被問到「為什麼這樣設計」時，
   答案要**去讀原始決策文件**（restart plan／計畫書／程式碼），不是憑印象重新編一套理由；
   查完就**寫進 `reports/`**。判準很簡單：**下一個 session 會不會需要再問一次同樣的問題**——
   會的話就該落盤。

---

## 2. 環境建置

worktree `D:\it_project\github_sync\zip-rl` 已存在且已 `uv sync`。若要從零重建：

```powershell
cd D:\it_project\github_sync\ml-workshop
git worktree add ..\zip-rl feat/rl-a2-training
Copy-Item .\linkedin-zip-challenge\.env ..\zip-rl\linkedin-zip-challenge\.env   # .env 不進版控，缺它 app 啟動會出錯
cd ..\zip-rl\linkedin-zip-challenge
uv sync
```

**驗證基線**（開工第一件事，不要假設環境是好的）：

```powershell
cd D:\it_project\github_sync\zip-rl\linkedin-zip-challenge
uv run pytest        # 通過數見下；xfailed 應為 8
uv run ruff check .  # 期待 All checks passed!
```

- ⚠ **通過數取決於這條 branch 帶了哪些 commit，不要當固定值**：
  `feat/rl-masked-ppo` 在 A1 當時是 **76 passed**；`main` 於 2026-08-29 併入 VLM track 後是
  **214 passed**（多出來的都是 `src/core/vl_models/`、`src/app/` 與 `src/ui/` 的測試，與 RL 無關）；
  **本分支 `feat/rl-a2-training` 在 2026-09-05 實測是 242 passed, 8 xfailed in 20.44s**（多的是 A2 的測試）。
  **拉了 main 之後數字變大是正常的**，不是壞掉。真正的基線用法是：開工先跑一次記下來，之後拿它比較。
- **8 個 xfailed 是刻意的**，不是壞掉：它們釘住 env v1 的缺陷（`xfail(strict=True)`），**若哪天變成 XPASS 會失敗**，代表有人改了 `rl_env.py`，那時要回頭更新 A0 報告。
- **venv 陷阱**：一律 `cd linkedin-zip-challenge` 再 `uv run`。repo 根的 `.venv` 是 py3.9 devtools，跑不動這個子專案。
- 相依已就緒：`torch 2.4.1+cu121`、`stable-baselines3 2.7.0`、`sb3-contrib 2.7.1`（含 `MaskablePPO`）、`tensorboard`。
  **A2 全程沒有新增任何套件**（2026-08-29 實測確認）。
- ⚠ **分支換了**：A2 起在 `feat/rl-a2-training`（從 `main` 開，因為 `feat/rl-masked-ppo` 已整條併進 main）。
  同一個 worktree `zip-rl`，重建方式相同，只是 `git worktree add ..\zip-rl feat/rl-a2-training`。

**重建資料集**（`datasets/` 不進版控，新 worktree 不會有）：

```powershell
uv run python -m src.core.rl.generate_dataset_v2 --count 1700 --sizes 4,5,6 --timeout 0.5 --name main_n1700_456
```

約 45 秒，產生 5,100 題（train/val/test = 8:1:1，依尺寸各自切）。加 `--sizes 7` 可補 7×7（100 題約 35 秒）。

---

## 3. 已驗證的事實（實測過，不要重測，也不要憑記憶推翻）

全部證據在 `reports/2026-08-15_a0-env-v1-findings.md`，原始數據可用
`uv run python -m src.core.rl.diagnose_env_v1` 重跑（輸出到 gitignore 的 `logs/rl_diagnostics/`）。

1. **舊 env（`rl_env.py`）合法解無法終止：0/7。** `reset()` 把起點當成待收集的 waypoint 1，收集判定只在移動後執行，
   合法一筆畫不重踩起點 ⇒ 索引永遠停在 0 ⇒ 終局不可達。
2. **成功獎勵只發給違規路徑**：同一條解答前面加一步「踩回起點」，6/6 終止並拿 +999.01（整局 +2359～+4946），
   而合法解只有 −35～−48。**舊環境能教出的最高分策略在定義上就是作弊。**
3. **2-cycle 假說成立**：兩個已訪格間來回 8 步只有 2 個相異觀測；用那 2 個狀態建的確定性策略跑 69 步到 truncated，全程只碰 2 格。
4. **舊 env 的非法移動不計入步數預算**（`truncated` 寫死 False）：撞牆 82 次、預算 72，從未回報 truncation。
5. **出題器有 parity 限制**：5×5 起點掃描 —— `(r+c)` 偶數 13/13 成功、奇數 0/12 全滅。奇數盤會有一定比例回 `None`。
6. **出題器的 20 秒 timeout 是純浪費**：7×7 會成功的搜尋最慢 0.415s（0.5s cutoff 下）。改成 0.5s 後
   5,100 題從外推的 ~23 小時降到 **45 秒**。這是呼叫端參數，**不必改共用模組**。
7. **Baseline（510 題 held-out × 20 局，`logs/rl_baselines/`）**：

   | policy | 4×4 | 5×5 | 6×6 |
   |---|---|---|---|
   | masked random | 8.8% | 0.9% | 0.0% |
   | greedy（往下一個號碼靠） | 10.2% | 3.7% | 0.8% |

   失敗中 **90–100% 是死路**（不是超時）。**greedy ＝距離型 shaping 的天花板，6×6 就崩掉**——
   這是報告 §2.2「距離位能與真目標不同構」的實驗證據。

**A2 新增（2026-08-29 實測）**

8. **反向 curriculum 本身有效，而且晉級成本可預期地上升。** 每往前推一級所需步數：
   4×4 是 416 → 13,600 → 19,872 → 28,496 → 89,680（到全長）；
   6×6 是 416 → 960 → 41,584 → 53,024 → 57,312 → 130,432 → 314,208 → 445,168 → 690,736 → 662,064（到 k=33）。
   **但不能用前段外推後段**——跑到一半用 ×1.29 推「1.3M 步到全長」，實際差三倍。
9. **兩個 goal 都在背答案。** deterministic 打訓練集 vs held-out：4×4 **0.947 / 0.788**、6×6 **0.553 / 0.253**。
   訓練集的 deterministic 成績等於訓練 rollout 曲線 ⇒ 落差是**泛化**，不是 argmax／取樣的差別。
10. **訓練成本與資源**：`DummyVecEnv` ＋ 16 env 約 **4,000–4,200 fps**；4×4 1M 步 248s、6×6 5M 步 1,203s。
    GPU 峰值只有 **83.8 MiB**，整個訓練程序約吃 1 個核心 ⇒ **瓶頸是單執行緒的 Python env step**，
    不是 GPU、不是網路。

**2026-09-05 新增**（完整分析見 [`reports/2026-09-05_rl-budget-and-design-notes.md`](reports/2026-09-05_rl-budget-and-design-notes.md)）

11. **`--resume` 續訓確實帶回 curriculum**（實測 log：`Resuming ... at 5005312 steps, k=27`）。
    ⚠ 但**要看 log 確認，不要相信程式碼**——陷阱 #10 的失敗模式就是靜默從 `k_start` 重來且每條曲線都正常。
    另外 `--timesteps N --resume` 是**再加 N 步**不是總數：SB3 在 `reset_num_timesteps=False` 時做
    `total_timesteps += self.num_timesteps`（`base_class.py:416`）。
12. **`--resume` 會覆寫同 run-id 的 `model_final.zip`／`train_state.json`／`eval_test.json`。**
    續訓前先把上一輪的複製一份（本次留下 `*_at5005312.*`），否則上一輪的成績就只剩文件裡的數字、
    沒有可重跑的權重。`progress.jsonl` 是 append，曲線歷史會自動保留。
13. **GPU utilization 高不代表 GPU 是瓶頸。** 實測訓練中 36–40%、**訓練結束後 0–5%**（同一張卡上還有桌面程式，
    所以**一定要跑結束後的對照**才能歸因）。但 `utilization.gpu` 量的是「有 kernel 在執行的時間比例」，
    不是算力佔用：117 萬參數的網路高頻發射小 kernel 就長這樣。
    佐證：兩輪 fps 幾乎相同（3,968 vs 4,002）。**顯存**才是真的沒用到——實測權重 4.31 MiB、
    batch 512 的 fwd+bwd+Adam 峰值 115.58 MiB（對照：7B 模型光 fp16 權重就 13.0 GiB）；
    參數約 90% 集中在 `Linear(4104→256)`，三層 conv 只有約 7.7 萬；
    **rollout buffer 根本不上顯卡**（SB3 用 `np.zeros`，`buffers.py:392`，每個 minibatch 才搬）。

---

## 4. 已定案的設計決策（不要重開）

| 決策 | 理由 |
|---|---|
| **全程一筆畫，倒車不開放**（2026-08-15 本人拍板改的） | 一筆畫在構造上必定可解；禁止重踩 ⇒ 2-cycle 定義上不可能；每次成功都是合法 Zip 解。詳見計畫書 §4 修訂說明 |
| **稀疏訊號靠反向 curriculum 解，不靠放寬規則** | 答案本來就在手上，從「解答倒數第 k 格」起步逐步往前推 |
| **合法性定義以 `src/core/solvers/dfs.py` 為準** | `dfs.py:96-105`（全覆蓋＋號碼依序）、`dfs.py:72-77`（站在號碼 1 上即算收集）。**solver 一直是對的，錯的是舊 env** |
| **reward 冰湖式**：成功 +1、其餘 0、速度由 γ 表達 | 舊版每步 −1 累積到 −72，淹掉終局訊號 |
| **v1 的 `rl_env.py`、舊訓練腳本、`models/dqn_*.pth` 保留當對照** | 不刪、不覆寫、不續訓 |
| **`src/core/utils.py`、`src/core/puzzle_generation/` 只讀不改** | 共用模組，要改先提出（VLM track 也在用） |
| **不用調 reward 權重修迴圈** | 2025-10 已試過 0.1→0.01，根因不在權重 |
| **RL 不取代 CP-SAT** | 價值在攤提式推論與學習方法本身 |

---

## 5. 程式地圖（這條 track 加了什麼）

| 檔案 | 用途 |
|---|---|
| `src/core/rl/rl_env_v2.py` | **主角**。`PuzzleEnvV2`：一筆畫 env、`action_masks()`、反向 curriculum、死路終止 |
| `src/core/rl/action_space.py` | 共用動作編碼（0:Up 1:Down 2:Left 3:Right）與 `path_to_actions()` |
| `src/core/rl/generate_dataset_v2.py` | 決定性資料集產生器，**保留 solution path**（舊的 `generate_rl_dataset.py:59` 會丟掉） |
| `src/core/rl/baselines.py` | masked random ／ greedy 兩個對照組與評估器 |
| `src/core/rl/diagnose_env_v1.py` | A0 的六個 probe，可重跑產生證據 JSON |
| `src/core/rl/train_config.py` | **A2 新增。改設定只改這裡**：`GOALS` 定義每個訓練目標（盤面、牆策略、步數預算、done 門檻），以及 `PPOSettings`／`NetworkSettings`／`CurriculumSettings`／`ResourceSettings` |
| `src/core/rl/train_maskable_ppo.py` | **A2 新增**。只負責執行一個 goal：`GridScalarExtractor`、curriculum callback、checkpoint ＋ `train_state.json`、評估。CLI 是 `--goal <key>` |
| `src/core/tests/rl/test_train_maskable_ppo.py` | **A2 新增**。18 個測試，含「SB3 會把 grid 攤平」與 curriculum state round-trip 兩個防呆 |
| `src/core/tests/rl/test_rl_env_v2.py` | 21 個測試：mask 四規則、死路邊界、reward 邊界、ground-truth 重播 |
| `src/core/tests/rl/test_rl_env_v1_diagnosis.py` | 釘住 v1 缺陷（8 個 strict xfail ＋ 對照測試） |
| `ai-collab/reports/2026-08-15_a0-env-v1-findings.md` | A0 完整報告 |
| `ai-collab/reports/2026-09-05_rl-budget-and-design-notes.md` | **2026-09-05 新增**。續訓結果 ＋ curriculum 成本曲線 ＋ GPU 實測 ＋ **§5 設計筆記**（做中學的落盤處）|

**沒有動到**：`src/core/rl/` 的舊檔案、`src/core/vl_models/`（VLM track 的地盤）、
`src/core/utils.py`、`src/core/puzzle_generation/`、`src/app/`、`src/ui/`。

### env v2 的介面速覽

```python
from src.core.rl.rl_env_v2 import PuzzleEnvV2, PuzzleSample

env = PuzzleEnvV2(samples, reverse_curriculum_k=None, shaping_lambda=0.2, gamma=0.99)
obs, info = env.reset()          # obs = {"grid": (8,8,8) float32, "scalars": (8,) float32}
mask = env.action_masks()        # (4,) bool —— MaskablePPO 直接吃這個方法名
obs, reward, terminated, truncated, info = env.step(action)
env.set_reverse_curriculum_k(6)  # 訓練中調整起點距離；k >= 2
```

- **8 個 grid channel**：valid／wall_right／wall_down／visited／agent／wp_next／wp_future／wp_done
- **8 個純量**：coverage、waypoint 進度、last_action one-hot(4)、height/8、width/8
- **reward 的實際大小**（`shaping_lambda=0.2`、`gamma=0.99`，2026-08-29 實算）：
  每步 shaping ＝ `0.2 × (0.99 × 覆蓋率_後 − 覆蓋率_前)`，
  4×4 每步 **+0.0123 → +0.0105**、整局累積 **+0.171**；6×6 每步 **+0.0054 → +0.0036**、整局 **+0.158**。
  成功另外 **+1.0** ⇒ **shaping 只佔一局總分的 14%**，訊號主體仍是終局那個 +1。
  死路與超時**沒有任何懲罰**（就是 0），「越快越好」由 γ 表達而不是每步扣分。
- **失敗只有兩種**：死路（四方向全被 mask，`info["dead_end"]`）與超時（一筆畫下幾乎不會發生，是防呆）
- **非法動作**：直接終止並回 `info["invalid_action"]`（讓 `check_env` 能跑，也讓「忘了套 mask」立刻現形）

---

## 6. 下一步

**A2 做完了**（結果見 §0）。這一節寫的是還沒做的，順序是有理由的——**不要跳過第一項**。

### ✅ 資料集實驗已完成，結論是「換瓶頸了」

新資料集 `seed20300000_n20000_4-6`（4×4 train 15,419／6×6 train 16,000，約 11 倍；切分前已去重、
三個 split 兩兩交集 0、`--verify` 通過）。兩個 goal 都用它重訓過，結果見 §0 ①。

**落差從 +0.162／+0.250 收斂到 +0.050／+0.009 ⇒「資料太少在背答案」成立且已解掉。**
**不要再加資料**——6×6 的落差只剩 +0.009，沒有東西可以再 overfit。

### ✅ ① 加預算已完成（2026-09-05），結論是「有效，但貴」

續訓 3M 步（5M → 8,011,776，749.5s / 4,002 fps）：curriculum **k=27 → 30**、held-out **0.344 → 0.409**。
詳見 §0 ③ 與 [`reports/2026-09-05_rl-budget-and-design-notes.md`](reports/2026-09-05_rl-budget-and-design-notes.md)。
**這一項的 done 條件只達成一半**：held-out 有跟著動（✅），但 curriculum 沒推到全長（❌，停在 k=30/36）。

重跑或再續訓的指令（`--timesteps` 是**再加**多少步，不是總數）：

```powershell
uv run python -m src.core.rl.train_maskable_ppo --goal goal2_6x6 --run-id goal2_6x6_bigdata `
  --resume --timesteps 3000000 --eval-split test --eval-episodes 20
```

⚠ 續訓前**先備份** `model_final.zip`／`train_state.json`／`eval_test.json`（§3.12）。

### ★ 接下來做這兩件（依序）

**① `shaping_lambda=0` 的對照 —— 先做這個，它最便宜**

restart 報告的階段表指定一筆畫階段的 shaping λ **是 0**（「只剩 +1 與 γ」），
而 `PuzzleEnvV2` 一直用 0.2（佔一局總分 14%），**從沒關掉跑過**。
⇒ **目前所有數字都是在一個與規格不符、且沒人驗證過的設定下量到的。**
改 `train_config.py` 的 `Goal.shaping_lambda=0.0`，開一個**新 run-id** 跑 4×4（約 4 分鐘）。
**Done**：知道 shaping 是在幫忙還是扯後腿——兩個答案都有價值。

**② 依 ① 的結果，決定要不要把 6×6 推到全長**

外推還需要 **15–30M 步 ≈ 1–2 小時**（⚠ 外推不可信，見 §0 ③）。**這是小時級，開跑前要本人授權。**
真正要回答的問題是「×2 成長」還是「成本發散」：
**最便宜的判別證據是下一級 30→33 實際花多少**——落在 4–8M 內則 ×2 模型成立，
顯著超出就該改路（調超參／加網路容量／改推論策略），而不是繼續砸預算。

⚠ **兩件事不要同時改**（shaping ＋ 預算一起動就分不出誰造成什麼）。

### 之後才輪到這些

| 想做的事 | 為什麼要等 |
|---|---|
| 調 PPO 超參（`ent_coef`、`lr`、`n_steps`） | 超參確實是**未調校**的起手值，但 `shaping_lambda` 的對照更便宜也更該先做（現在等於在一個沒人驗證過的 reward 設定上調超參）|
| `shaping_lambda` **敏感度掃描**（0.1／0.3…） | 先跑 §6 的 λ=0 對照，確認它到底有沒有用，再談掃描 |
| A3（5×5、加牆、權重接續） | 6×6 還在 k=30/36，沒到全長 |
| A4（7×7、held-out 1,000 題） | 7×7 資料集也還沒生（`--sizes 7`，100 題 35 秒，已不是瓶頸） |
| A5（掛成 API 第 10 種 solver） | ⚠ 會動 `src/app/routers/solver.py`，動之前先確認 VLM track 沒在改。
  另外**牆的分布不同**：RL 訓練資料的牆是 0 或 2–5 道，VLM 讀出來的真實題目可到 10+ 道 ⇒ 分布外 |

**⚠ 長時間訓練（小時級）開跑前要先問本人。** 目前的規模是分鐘級（4×4 1M ≈ 4 分、6×6 5M ≈ 20 分），
資源上限已寫進 `ResourceSettings`（CPU 執行緒與 GPU 記憶體各 75%），實測只用約 1 個核心與 84 MiB GPU。
訓練成品放 `models/`、資料放 `datasets/`，都不進版控。

---

## 7. 陷阱清單（我踩過的）

1. **pre-commit 的 `ruff` 是釘 v0.4.8，與專案 venv 的 0.14.1 格式化結果不同。**
   commit 時 hook 會改檔並中止；**重新 `git add` 同一批檔案再 commit 一次**即可（不要用 `--no-verify`）。
2. **`uv run` 要在子專案目錄下**；用 `python <script.py>` 直接跑會 `ModuleNotFoundError: No module named 'src'`，
   要嘛 `uv run python -m src.core.rl.<module>`，要嘛帶 `PYTHONPATH=.`。
3. **出題器會回 `None`**（parity）。任何生成迴圈都要處理，不能假設一定拿得到題目。
4. **`timeout_per_attempt` 不要用預設的 20 秒**，用 0.5 秒（見 §3.6）。
5. **`MaskablePPO` 不能接受全 False 的 mask**——env 已在死路時先 `terminated=True`，改 env 時不要破壞這個保證
   （`test_dead_end_terminates_before_an_all_false_mask_is_sampled` 在守這件事）。
6. **`dev_log.md` 與 `roadmap.md` 兩條 track 會同時改**，rebase 時常在 `## 2026-08-15` 區塊衝突：**兩邊都保留**。
7. **不要從 `models/dqn_*.pth` 續訓**——那是失敗策略的權重。

**A2 新增的五個（2026-08-29 實際踩到）**

8. **★ SB3 會靜默把 grid 攤平，不會報錯。** `MultiInputPolicy` 靠 `is_image_space` 決定要不要用 CNN，
   而它要求 `uint8` 0–255；我們是 `float32` 0–1 ⇒ `CombinedExtractor` 把 8×8×8 攤成 512 維丟給 MLP，
   **盤面幾何整個消失，而且完全沒有警告**。反過來強制走圖片路徑（`normalized_image=True`）才會崩：
   `Calculated padded input size per channel: (1 x 1). Kernel size: (4 x 4)`（NatureCNN 開頭是 8×8 stride-4）。
   **一律用 `GridScalarExtractor`**，`test_sb3_would_flatten_the_grid` 在守這件事。
9. **★ Gymnasium 1.x 移除了 wrapper 的屬性穿透。** `Monitor(env).action_masks()` 直接 `AttributeError`；
   `VecEnv.env_method` 沒事只是因為 SB3 走 `get_wrapper_attr`。讀 mask 一律用 `read_action_masks()`。
   **附帶教訓**：我為了預防這件事寫的探針沒抓到它，因為探針只建構、從沒呼叫那個 lambda——
   **只驗建構不驗熱路徑的探針，證明力比看起來低**。
10. **★ 續訓不會帶 curriculum。** SB3 的 `.zip` 只有 policy 與 optimizer，沒有 `reverse_curriculum_k`，
    所以只載模型會**靜默從 `k_start` 重來，而且每條曲線都正常**。`train_state.json` 存在 checkpoint 旁邊，
    `--resume` 會讀它。改動這一塊之後**要實際存檔→續訓→比對 k 與 timesteps**，不要只看程式碼。
11. **★ `SubprocVecEnv` 在這個 env 比 `DummyVecEnv` 慢**（3,221 vs 3,880 fps，100k 步實測）。
    env step 是很輕的單執行緒 Python（整個訓練程序只吃約 1 個核心），Windows 的行程間通訊成本大於平行收益。
    **restart plan §4.8 寫的 `SubprocVecEnv` 對這個 env 是錯的**；預設已改成 `dummy`。
12. **checkpoint 一個 14 MB，而且刻意不刪。** 控制磁碟用**間隔**（`Goal.checkpoint_every`）而不是保留上限——
    VLM track 的 `save_total_limit=2` 刪掉中段 checkpoint，讓一個實驗永遠做不成。每次跑抓 25–50 個。
13. **★ 資源上限要涵蓋「所有會開工作的地方」，不是只有訓練。** 本人要求「CPU 最多 75%，我還要用電腦」，
    我把它接到訓練（torch 執行緒＋GPU 記憶體）卻漏了 `generate_dataset_v2` 的 `Pool(processes=None)`
    ——那等於 `os.cpu_count()`＝24 個 worker，**實際吃到 99–100%，是本人先發現的**。
    預算現在只寫在 `train_config.DEFAULT_CPU_FRACTION` 一處，訓練與生成共用。
    **談定上限後，先把所有 `Pool`／`set_num_threads`／`n_envs`／子行程都找一遍再宣稱已設限。**
14. **★「N% 的核心數」不等於「N% CPU」。** `int(24 × 0.75)`＝18 個 worker 實測仍是 **74–82%**，
    因為父程序（發任務、收結果）與 OS 也吃。保留 2 個給父程序、用 16 個之後平均 **68%**。
    **上限沒有在工作跑的時候取樣過，就不算設好了——算出來不等於量過。**
15. **★ 訓練集／held-out 的落差要先排除「兩邊難度不同」才能叫 overfit。** 用兩個 baseline
    （它們沒從任何一邊學到東西）打同一組 train／test 就能分辨。本次四項全顯示 held-out 一樣難或略易，
    模型自己的落差才有意義。**沒有這個對照就不要宣稱泛化問題。**

16. **★★ 多個 worker 同時往同一個 stderr「管線」寫 log 會整組卡死。** `generate_puzzle` 每次嘗試印一行，
    40k 的建置會從 16 個程序吐出 **19.8 MB／約 9 萬行**到同一個繼承來的 stderr；那個 stderr 是管線時就死鎖。
    **症狀長得像「很慢」**：所有 worker 的 CPU 停在幾乎相同的值、全部閒置、父程序也閒置、十幾分鐘沒有輸出——
    **那是大家一起被同一件事擋住，不是還在算**（累積 CPU 已經超過整份工作需要的量就是鐵證）。
    三個對照：接管線＋worker 印 log **卡死 >120s**；同樣的 pool 直接寫檔 **2.2s**；接管線但 worker 不印 log **2.1s**。
    - **修法**：`logger.disable("src.core.puzzle_generation.puzzle_generator")`，
      **必須寫在 worker 函式 `_generate_one` 裡面**。⚠ 寫在 `build_dataset`（父程序）**沒有用**——
      Windows 的 `spawn` 子程序不會執行到那行，我這樣改過一次，log 大小 19,787,873 bytes **完全沒變**。
    - `vl_models/dataset_builder.py:291` 早就這樣做了，註解一模一樣，docstring 還記著它們也在 Windows 上卡死過。
      **同一個共用產生器的 log 已經害了兩條 track。**
17. **★ 小盤面會生出重複題，而且會跨 split 洩題。** 要 20,000 題時 4×4 只有 **97.2% 唯一**，
    且 **111 題同時出現在 train 和 test（held-out 的 5.57%）**——那會直接灌水你要量的泛化指標。6×6 是 100% 唯一。
    產生器現在**切分前依 fingerprint 去重**，三個 split 在構造上不重疊。
    ⚠ 舊那包（1,700/尺寸）回頭驗過 **train∩test = 0**，所以 A2 的結論沒被影響。
    **要更多資料時，一定要同時問「其中有多少是新的」。**
18. **★ 修好了沒有，要用「效果」判斷，不是看 diff。** 本次兩個修正在 diff 上都對、實際都沒生效
    （CPU 上限算出 18 個 worker 卻量到 74–82%；logger 關在不會生 worker 的程序裡）。
    兩次都只要**一個量測**就當場拆穿：取樣 CPU、看 log 檔多大。

19. **★★ `deterministic=True` 的重跑是位元級相同——重試等於零收益；但改成抽樣就會變好。**
    實測 300 題 4×4：換 reset seed 兩次評估的動作序列 **300/300 完全相同**。
    改成 `deterministic=False` 抽樣後（Zip 的解**可驗證**，所以 best-of-N 是正當手段，不是猜）：

    | 推論方式 | solve |
    |---|---|
    | deterministic | 0.870 |
    | 抽樣 1 次 | 0.863 |
    | 抽樣 best-of-2 | **0.903** |
    | 抽樣 best-of-4 | 0.930 |
    | 抽樣 best-of-16 | **0.967** |

    成功時的**中位嘗試次數是 1**，額外預算幾乎全花在難題尾巴。
20. **★★ `target_solve_rate` 這條線沒有推導，而且沒有指定推論模式——不要當它是硬標準。**
    同一個模型 deterministic 0.870（未達標）、best-of-2 0.903（達標）⇒ **「有沒有過」目前可以靠改推論設定調整**。
    0.90 的來歷：計畫書與 restart 報告都只寫結果不寫理由，而且它原本是**已廢棄的三階段設計**裡的
    「軟成功率」**階段升級門檻**。它現在還跟 `CurriculumSettings.promote_threshold`（也是 0.9，但量的是
    **訓練集上、隨機策略**的 rollout）撞名撞值，兩者意義完全不同。
    **往後兩個數字都要報**：deterministic（拿來比較訓練設定，沒有推論預算這個變數）
    ＋ best-of-N（附 N 與平均嘗試次數，那才是「當 solver 有多好用」）。
21. **★★ curriculum 的 rollout 成功率**不是**能力指標，差距很大。** 它疊了三層有利條件：
    起點是 **k 格（前面已預先走好）**、題目是**訓練集**、策略是**隨機取樣**。
    實例：6×6 在 k=27 的 rollout 是 **0.820**，同一個模型從真起點、held-out、deterministic 是 **0.344**。
    rollout 的用途只有兩個：**決定 curriculum 晉級**、**判斷加預算有沒有用**。
    **拿它當成績＝重演 2025 年「被訓練期高分騙過」那次失敗。**
    ⚠ **4×4 不受這個影響**：它的 curriculum 已推到全長，所以最終 rollout 與評估量的是同一件事
    （只差訓練集／held-out 與取樣／argmax）。**「沒看過的 4×4 一次走完」的正確數字是 0.877。**

**2026-09-05 新增**

22. **★ 交接文件在每個 worktree 都有一份，別的 worktree 那份是舊的。** 本檔進版控，
    所以 `zip-vlm`／`ml-workshop` 看到的是**那條分支上次 commit 的版本**——2026-09-05 當天
    `zip-vlm` 的副本停在 2026-08-15，還在寫「下一步是 A2 第一次訓練」，而 A2 早就跑完兩輪。
    **從哪個 worktree 開 session，讀到的 `CLAUDE.md`／`AGENTS.md`／handover 就是哪一份。**
    要動 RL 就從 `zip-rl` 開，或至少確認你讀的是 `zip-rl` 那份。
23. **★ `--timesteps N --resume` 是「再加 N 步」不是「總數」**（SB3 `base_class.py:416`：
    `reset_num_timesteps=False` 時 `total_timesteps += self.num_timesteps`）。
    把它當總數會得到一個比預期長得多的 run。

---

## 8. 與 VLM track 的協作約定

| 面向 | 約定 |
|---|---|
| 程式碼 | 我動 `src/core/rl/`；VLM 動 `src/core/vl_models/`、`src/app/`、`src/ui/` |
| ⚠ 交會點 | **A5 會動 `src/app/routers/solver.py`**（掛第 10 種 solver），動之前先確認 VLM track 沒有同時在改 |
| 共用模組 | `src/core/utils.py`、`src/core/puzzle_generation/` **只讀不改**，真要改先提出 |
| 相依 | `pyproject.toml`／`uv.lock` 序列化處理；新套件由本人授權 |
| 文件 | dev_log 各自加自己的 `###` 區塊；roadmap 只改自己那一項；衝突時兩邊都保留 |
| CPU | VLM 的資料生成也吃多核，長工作錯開跑 |

---

## 9. 已知缺口與尚未決定的事

- **7×7 資料集還沒生**。當初因為「太慢」被本人喊停，但 §3.6 的修正之後 100 題只要 35 秒，**已經不是瓶頸**，
  A4 之前補生即可（`--sizes 7`）。
- ~~**資料集規模**：若 A2 出現明顯 overfit 再回頭加大~~ → **已完成並結案（2026-08-29）**：
  加大 11 倍後落差從 +0.162／+0.250 收斂到 **+0.050／+0.009**。**這條線走完了，不要再加資料。**
- **⚠ `shaping_lambda=0.2` 與設計規格不符**：restart 報告的階段表指定一筆畫階段是 **0**（「只剩 +1 與 γ」），
  而實作一直是 0.2（佔一局總分 14%），**從未關掉跑過**。這是 §6 的第二件事，一次對照就有答案。
- **只訓過一個 seed**。所有結論都建立在單一 seed 上，跨 seed 重複還沒做。
- **網路架構已實作**：`GridScalarExtractor`（3 層 3×3 conv、padding=1、不 pooling、policy 共 1,170,949 參數）。
  ⚠ **SB3 預設不會用 CNN**，原因見 §7.8。
- **6×6 還沒推到全長**（目前 **k=30/36**，2026-09-05 續訓後）。「是預算問題不是能力問題」**已有一次實驗支持**
  （加預算真的推動了一級），但**尚未證明能推到底**：每級成本約 ×2 成長，而「成本發散」這個替代解釋還沒排除。
  判別方式見 §0 ③。`--resume` 可續，備份要先做（§3.12）。
- **⚠ 一輪訓練約 12 分鐘、推到全長估 1–2 小時**，後者是**小時級 ⇒ 開跑前要本人授權**（守則 4）。
- **出題器的 parity 根治**（奇數盤只從多數色挑起點）要動共用模組，**已提報但未做**，由本人決定。
