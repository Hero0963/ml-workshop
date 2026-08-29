# 交接文件 — RL Track（一筆畫 solver）

> **接手這條 track 的 agent／developer 從這一份開始讀，讀完就能動手。**
> 最後更新：2026-08-29（Asia/Taipei）｜分支 `feat/rl-a2-training`｜worktree `zip-rl`｜對應 roadmap 第 3 項
> 其他文件是延伸閱讀，本檔會標明什麼時候該去翻哪一份。
> 姊妹 track 的交接文件：[`handover-vlm-parser.md`](handover-vlm-parser.md)

---

## 0. 一句話現況

**A0／A1／A2 都跑完了。訓練機制是好的，但兩個 goal 都沒過門檻——而且根因已查明：不是訓練不夠，是資料太少，模型在背答案。下一步是加大資料集再重訓，不是調參。**

> ★ **接手前一定要知道的四件事：**
>
> **① A2 首次訓練已完成（2026-08-29）**，成績與門檻如下（held-out test，`deterministic=True`）：
>
> | goal | 步數 | 耗時 | curriculum | test | 門檻 | greedy／random |
> |---|---|---|---|---|---|---|
> | 4×4 | 1M | 248s | 推到全長（152k 步） | **0.788** | 0.90 ❌ | 0.102／0.088 |
> | 6×6 | 5M | 1,203s | 只到 **k=33/36** | **0.253** | 0.85 ❌ | 0.007／0.001 |
>
> 兩個 baseline **完全重現 2026-08-15 的數字** ⇒ 評估協定已對齊，這些數字可以直接和舊表比。
>
> **② 根因是泛化，不是訓練量。** 把最終策略用 deterministic 同時打訓練集與 held-out：
> 4×4 **0.947 vs 0.788**、6×6 **0.553 vs 0.253**。訓練集的 deterministic 成績等於訓練曲線，
> 所以落差**不是** argmax／取樣的差別。**每個尺寸 1,360 題訓練資料不夠**（§9 那個未決問題有答案了）。
> **先加步數不加資料，只會背得更熟。**
> **對照已做**：兩個 baseline（沒看過任何一邊）打同一組 train／test，四項全部顯示 test **一樣難或略易**
> （4×4 random 0.0753/0.0876、greedy 0.1038/0.1018；6×6 random 0.0000/0.0009、greedy 0.0041/0.0074），
> 所以「test 比較難」這個解釋被排除。模型自己的落差 z ＝ **+4.32／+5.64**。
> ⚠ **還沒證明的部分**：train 只抽 170/1,360、只訓一個 seed ⇒「資料不夠」是**最合理解釋、不是唯一解**
> （網路容量與缺正則化都還沒排除）。**真正的證明是加大資料集後落差縮小**——所以那是 §6 的 done 條件。
>
> **③ 6×6 沒有卡住，是預算不夠。** k=33 的成功率在剩下 2.6M 步一路單調爬 0.518 → **0.800**（門檻 0.90），
> 死路率同步降到 0.200。再約 1.2M 步有機會過關，而且 `--resume` 會帶著 curriculum 續訓，不必重跑。
>
> **④ 改設定只改 `src/core/rl/train_config.py`。** goal（盤面／牆策略／步數預算／done 門檻）、PPO 超參、
> 網路、curriculum、資源上限全在那一個檔。訓練腳本只負責執行一個 goal。

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
| 7 | `dev_log.md` 的 `## 2026-08-15` → RL Track A0／A1 兩則 | 想看做了什麼、量到什麼時翻 |

> ⚠ **不要整份讀 `dev_log.md`**（1,700+ 行），用日期或關鍵字搜。

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

---

## 2. 環境建置

worktree `D:\it_project\github_sync\zip-rl` 已存在且已 `uv sync`。若要從零重建：

```powershell
cd D:\it_project\github_sync\ml-workshop
git worktree add ..\zip-rl feat/rl-masked-ppo
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
  **214 passed**（多出來的都是 `src/core/vl_models/`、`src/app/` 與 `src/ui/` 的測試，與 RL 無關）。
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
- **失敗只有兩種**：死路（四方向全被 mask，`info["dead_end"]`）與超時（一筆畫下幾乎不會發生，是防呆）
- **非法動作**：直接終止並回 `info["invalid_action"]`（讓 `check_env` 能跑，也讓「忘了套 mask」立刻現形）

---

## 6. 下一步

**A2 做完了**（結果見 §0）。這一節寫的是還沒做的，順序是有理由的——**不要跳過第一項**。

### ★ 先做這個：加大資料集，然後重訓

**根因是泛化不是訓練量**（§0 ②），所以在資料變多之前，加步數、調超參、改網路都只會讓模型把
同一批 1,360 題背得更熟。

- 生成很便宜：`generate_dataset_v2.py` 產 5,100 題只要 45 秒，加大到每尺寸 10,000–20,000 題是分鐘級。
  ```powershell
  uv run python -m src.core.rl.generate_dataset_v2 --count 20000 --sizes 4,6 --timeout 0.5 --name main_n20000_46
  ```
- 改用新資料集只要動 `train_config.py` 的 `Goal.dataset`，不必碰訓練腳本。
- **Done 條件**：deterministic 的「訓練集 vs held-out」落差明顯縮小（現在是 4×4 0.947→0.788、
  6×6 0.553→0.253）。**落差沒縮小就不要往下走**——那代表瓶頸不在資料量。

### 然後：把兩個 goal 的預算補足

- **6×6 用 `--resume` 續訓**，不要重跑：`--goal goal2_6x6 --resume --timesteps 3000000`。
  它會從 `train_state.json` 讀回 k=33 與 5M 的步數繼續。約 12 分鐘。
- 4×4 已經推到全長，缺的是泛化，所以它等資料集。

### 之後才輪到這些

| 想做的事 | 為什麼要等 |
|---|---|
| 調 PPO 超參（`ent_coef`、`lr`、`n_steps`） | 超參是**未調校**的起手值沒錯，但現在的瓶頸是資料，調了也量不準 |
| `shaping_lambda` 敏感度（現在 0.2，未調校） | 同上 |
| A3（5×5、加牆、權重接續） | 6×6 都還沒到全長 |
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
- ~~**資料集規模**：判斷是這個網路用不到 50k，若 A2 出現明顯 overfit 再回頭加大~~
  → **已有答案（2026-08-29）：overfit 出現了，要加大。** 每尺寸 1,360 題訓練資料不夠，
  deterministic 的訓練集／held-out 落差是 4×4 0.947/0.788、6×6 0.553/0.253。**這是下一步第一件事**（§6）。
- **`shaping_lambda=0.2` 未經調校**，γ、λ 都只是合理起點。**敏感度檢查要等資料集加大之後做**，
  現在量到的差異會被泛化落差蓋過去。
- **網路架構已實作**：`GridScalarExtractor`（3 層 3×3 conv、padding=1、不 pooling、policy 共 1,170,949 參數）。
  ⚠ **SB3 預設不會用 CNN**，原因見 §7.8。
- **6×6 還沒推到全長**（停在 k=33/36），但那是預算問題不是能力問題，`--resume` 可續。
- **出題器的 parity 根治**（奇數盤只從多數色挑起點）要動共用模組，**已提報但未做**，由本人決定。
