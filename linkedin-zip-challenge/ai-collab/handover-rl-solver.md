# 交接文件 — RL Track（一筆畫 solver）

> **接手這條 track 的 agent／developer 從這一份開始讀，讀完就能動手。**
> 最後更新：2026-08-15（Asia/Taipei）｜分支 `feat/rl-masked-ppo`｜worktree `zip-rl`｜對應 roadmap 第 3 項
> 其他文件是延伸閱讀，本檔會標明什麼時候該去翻哪一份。
> 姊妹 track 的交接文件：[`handover-vlm-parser.md`](handover-vlm-parser.md)

---

## 0. 一句話現況

**A0（環境健全性）與 A1（env v2 ＋ 資料集 ＋ baseline）已完成並落盤；下一步是 A2 —— 第一次真正的訓練（4×4 ＋ 反向 curriculum ＋ MaskablePPO）。**

A0 的結論改變了整條 track 的前提：**2025-10 的舊環境不是「難學」，是「餵標準答案也不會過關」**，
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

> ⚠ **不要整份讀 `dev_log.md`**（800+ 行），用日期或關鍵字搜。

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
uv run pytest        # 期待 76 passed, 8 xfailed
uv run ruff check .  # 期待 All checks passed!
```

- **8 個 xfailed 是刻意的**，不是壞掉：它們釘住 env v1 的缺陷（`xfail(strict=True)`），**若哪天變成 XPASS 會失敗**，代表有人改了 `rl_env.py`，那時要回頭更新 A0 報告。
- **venv 陷阱**：一律 `cd linkedin-zip-challenge` 再 `uv run`。repo 根的 `.venv` 是 py3.9 devtools，跑不動這個子專案。
- 相依已就緒：`torch 2.4.1+cu121`、`stable-baselines3 2.7.0`、`sb3-contrib 2.7.1`（含 `MaskablePPO`）。**不需要再裝任何東西就能做 A2。**

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

## 6. 下一步：A2（Phase 1 訓練）

**目標**：4×4 無牆，反向 curriculum 從 k=3 起，達標後往前推到真正起點。
**Done 條件**：k 推到全長時通關率 ≥ 90% **且顯著高於 masked random 的 8.8%**；各 k 的學習曲線與 seed 落盤。

要做的事：

1. 寫 `src/core/rl/train_maskable_ppo.py`（新檔，**不要覆寫舊的 `train_single_cnn_sb.py`**）。
2. **⚠ 預期會踩的坑（未驗證，只是預判）**：SB3 的 `MultiInputPolicy` 對 image-like 空間預設用 `NatureCNN`，
   它有最小尺寸要求，**8×8 很可能過不了**，需要自訂 `BaseFeaturesExtractor`（3 層 3×3 conv、padding=1、不 pooling
   → flatten → 接純量 → MLP 256 → policy(4) ＋ value(1)，報告 §4.7）。**動手前先實跑一次確認，不要照抄我的猜測。**
3. **反向 curriculum 的推進要自動化**：用 callback 監看近期成功率，達標就 `env.set_reverse_curriculum_k(k+3)`，
   並把每個 k 的成功率與 step 數記進 tensorboard／JSON。**同時記死路率隨 k 的變化**——那是判斷要不要啟用備案的依據。
4. PPO 超參起手值見報告 §4.8（`n_envs=16`、`n_steps=512`、`lr=3e-4`、`gamma=0.99`、`ent_coef=0.01`）。**未調校**。
5. **評估一律 `deterministic=True`**，並與 baseline 一起報。訓練期的高分不算數——那正是 2025 年被騙的地方。

**⚠ 長時間訓練（小時級）開跑前要先問本人。** 訓練成品放 `models/`、資料放 `datasets/`，都不進版控。

之後：A3（5×5→6×6 加牆）→ A4（6×6→7×7 完整規則、held-out 1000 題）→ A5（掛成 API 第 10 種 solver）。

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
- **資料集規模**：計畫書原寫 50k，目前是 5,100 題。判斷是這個網路（10⁵–10⁶ 參數）用不到 50k，
  不夠再補生。**若 A2 出現明顯 overfit 再回頭加大。**
- **`shaping_lambda=0.2` 未經調校**，γ、λ 都只是合理起點，A2 要做敏感度檢查。
- **網路架構尚未實作**（見 §6.2 的預判）。
- **訓練完全還沒開始**——目前沒有任何 RL 模型權重存在。
- **出題器的 parity 根治**（奇數盤只從多數色挑起點）要動共用模組，**已提報但未做**，由本人決定。
