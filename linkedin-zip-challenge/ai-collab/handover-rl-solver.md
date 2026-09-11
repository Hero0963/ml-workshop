# 交接文件 — RL Track（一筆畫 solver）

> **接手這條 track 從這一份開始讀。這裡只寫「開工前必須知道的」。**
> 細節一律不重複——量過的數字與踩過的坑在 [`rl-traps-and-facts.md`](rl-traps-and-facts.md)，
> 其餘去哪查看 **§6 總目錄**。
> 最後更新：2026-09-12（Asia/Taipei）｜分支 `feat/rl-a2-training`｜worktree `zip-rl`｜roadmap 第 3 項
>
> ⚠ **要讀就讀 `zip-rl` 這份**：本檔在每個 worktree 都有一份，別的 worktree 拿到的是那條分支
> 上次 commit 的版本（`zip-vlm` 的副本停在 2026-08-15，還在說「A2 尚未開始」）。
> 姊妹 track：[`handover-vlm-parser.md`](handover-vlm-parser.md)

---

## 1. goal 與判定標準（先讀這個）

**訓練出能解「沒看過的題目」的策略**：`goal1_4x4` ≥ **0.90**、`goal2_6x6` ≥ **0.85**（held-out test），
最終 **A5 掛成 API 的第 10 種 solver**。

⚠ 同樣是定案的：**這條 track 的產出是「做中學」，不是指標。**
沒達標不等於失敗，**「知道為什麼沒達標」本身就是產出**；反過來，為了衝分數而犧牲可解釋性
（同時改多個變數、只留贏的那次、拿訓練期高分當成績）**與目標相反**。

> ⚠ **goal 原本寫在這份文件的第 520 行，而前面是一面十幾條的資訊牆——2026-09-05 實際造成過發散**
> （接手的 agent 讀完那面牆，開始優化最後一條提到的東西）。所以它現在在最前面。
> **但要誠實：知道 goal 並不足以防發散**——當天那個 agent 讀過也引用過 goal，照樣多花 30 分鐘
> 去確認一個試跑已經否定的東西。**真正能擋住的是 §2 那張死路地圖**，不是自我提醒。

### 判定用哪個推論模式（★ 2026-09-05 定案，此前一直是空白）

**門檻用 best-of-N 判定，但 deterministic 與 best-of-N 兩個數字都要報**（附 N 與平均嘗試次數）。

- **為什麼**：0.90／0.85 從來沒指定推論模式（同一個模型 deterministic 0.870 未達標、best-of-2 0.903 達標），
  而 A5 的產品形態就是 solver、**Zip 的解可驗證** ⇒ 花推論預算是正當手段，不是猜。
- ⚠ **但 deterministic 仍是比較「訓練設定」的唯一公平尺**（它沒有預算這個變數）。
  **拿 best-of-N 比兩種訓練方法＝拿「多花計算」冒充「變聰明」。**

### 現況（2026-09-12）

**判定基準是 best-of-32**（2026-09-11 本人定案：練習用專案，不在 N=32 vs 64 上著墨）。

| goal | deterministic | best-of-32 | 門檻 | 狀態 |
|---|---|---|---|---|
| 4×4 | **0.9404**（`bc_multi_456`）| **0.9917**（1.55 次／題）| 0.90 | **✅ 達標** |
| 6×6 | **0.5205**（`bc_multi_456`）｜舊 `bc_6x6` 0.4620 | `bc_6x6` = **0.8500**（8.05 次／題）| 0.85 | **✅ 踩線達標** |

⚠ **0.8500 剛好等於門檻**（1,700／2,000），而 best-of-N 的**評估雜訊約 ±0.01**
（同模型同測試集，只因抽樣 rng 串流不同，N=16 就從 0.8000 變 0.8105）
⇒ 講「達標」要連著這句話，best-of-64 = 0.8835 才是有餘裕的那個。

**★ 現在服務的是一個吃三種尺寸的模型 `bc_multi_456`**：4×4／5×5／6×6 三個盤面**都**贏過
同資料訓練的單尺寸專用模型（+0.0362／+0.0365／+0.0315，deterministic），訓練成本還略低。
**5×5 從此有模型**。完整對照在 [收尾報告](reports/2026-09-12_rl-wrap-up.md) §2；
兩種訓練 × 每個 N 的舊對照在 [BC 報告](reports/2026-09-05_rl-behaviour-cloning.md) §3。

**A5 已完成**：`RL (behaviour cloning)` 是 `/api/solver/solve` 的第 4 種 solver，
Docker 一鍵起得來（[部署指南](deployment-guide.md)）。

**PPO 到底能不能成功？** 4×4 **已經成功**（best-of-4 = 0.9238）。
6×6 **沒有證據支持它能到 0.85**：curriculum 卡在 k=30/36、每級成本約 ×2、8M 步只有 0.4095，
而外推不可信。**唯一沒做的 PPO 實驗是「推到全長會停在哪」，1–2 小時，要授權。**
詳見 [BC 報告](reports/2026-09-05_rl-behaviour-cloning.md) §6。

⇒ **唯一還開著的缺口是 6×6。不是在攻 6×6 的工作，開跑前先說明為什麼要做。**

**什麼算做完**：4×4 已達標 ⇒ 後續判定只看 6×6。
**若 6×6 窮盡目前路線仍到不了 0.85 ⇒ A5 只上 4×4，6×6 標為實驗性並寫清楚為什麼**——那份說明就是交付物。

### 判讀規則（違反這條的數字不要報告成進步）

**訓練 seed 的雜訊地板：4×4 實測 ±0.02–0.04**（3 seed 臂內全距 0.0380，而 greedy baseline 跨 6 個 run
只差 0.002 ⇒ **雜訊幾乎全在訓練 seed，不在評估**）。**任何改動效果要大於 ±0.04 才算數。**
⚠ **6×6 的雜訊從來沒量過**（一輪約 20 分鐘）——在那之前，6×6 的 0.0x 差異一律標「**有動但未確證**」。

---

## 2. ⛔ 已關掉的線（都有實驗證據，**不要重開**）

| 不做 | 為什麼 | 證據 |
|---|---|---|
| 繼續加資料 | 落差只剩 +0.009，沒東西可再 overfit | dev_log `2026-08-29` |
| 把「**當前狀態**的拓撲性質」放進觀測 | 策略根本沒在用它（失敗中亮燈比例 62.1% → 64.6%）| [連通性報告](reports/2026-09-05_rl-connectivity-feature.md) |
| **動作條件版**連通性、**割點** | 一步前瞻的 oracle **上界**只有 +0.0156（4×4），而那是**整族**的上界 | [oracle 報告](reports/2026-09-05_rl-lookahead-oracle.md) §5 |
| **GNN** | 主要論據被上面兩個否掉；只剩「跨尺寸泛化」沒被測到 | 同上 §7.5 |
| **policy 排序 DFS 用在 6×6** | 100 節點以上輸給 best-of-N | 同上 §10 |
| **帶重啟的 DFS** | k1／k2 輸給 best-of-N **和** policy-DFS ⇒ 這一族的極值在「完全不回溯」端點 | 同上 §11 |
| 調 PPO 超參、`shaping_lambda` 掃描 | 能買到的量級大機率在雜訊裡；λ=0 vs 0.2 已是 null | [預算與設計筆記](reports/2026-09-05_rl-budget-and-design-notes.md) §6 |
| 單純把 6×6 推到全長 | 只會把 0.409 推高一點，**不會靠近 0.85** | 同上 §1–3 |
| 調 reward 權重解迴圈 | 2025-10 試過；根因在 env，已重寫 | [A0 報告](reports/2026-08-15_a0-env-v1-findings.md) |

**還開著、但目前沒有證據支持的**：加深 conv trunk（已證偽的是「加拓撲特徵」，**不是**「加深度」）。

**★ 2026-09-12 更正一條**：舊版這裡寫「GNN 的跨尺寸泛化測不到，因為 `Linear(4104→256)` 綁死 8×8 padding」
——**寫反了**。padding 到 8×8 正是**讓**跨尺寸可測的原因（純量帶 `height/8`、`width/8`，
`_load_sample()` 每局重讀高寬），所以現成 checkpoint 直接就能餵別的尺寸。已經量完：
**泛化真的存在但單向**（6×6 模型在沒看過的 4×4 拿 0.5456，4×4 模型在 6×6 只有 0.0105），
而且**混合尺寸訓練在三個盤面都贏單尺寸專用模型**。⇒ GNN 的「跨尺寸」論據現在不是「沒測過」，
而是「**不用 GNN 也做得到**」。

**★ 已經量到的唯一瓶頸**：**6×6 單步正確率 93.9%，達標需要 98.9%**——要把單步錯誤率**砍 5.4 倍**。
這是「換等級」不是「改良」，任何新想法先對照這個數字。
（模型大小**不是**瓶頸，有三個實測支持，見 [oracle 報告](reports/2026-09-05_rl-lookahead-oracle.md) §7 的 AlphaGo 定量對照。）

---

## 3. 現在該做什麼

**★ 2026-09-11 本人定案：「我知道有別種 solver，但這個專案就是要用 RL 做。」**
⇒ 收尾已完成（多尺寸模型、A5 掛 API、Docker），**主線變成「真的用 RL」**。

### 下一步，依優先序

| 順位 | 做什麼 | 修的是什麼 | 成本 | 正牌 RL？ |
|---|---|---|---|---|
| **1** | **BC → PPO 微調**（`--init-from` **尚未實作**）| BC 的天花板（只會資料集那一條解）＋ 走偏後沒訊號 | 實作 1–2h ＋ 訓練小時級（**要授權**）| ✅ |
| 2 | **ExIt／拒絕抽樣微調** | 同上，但用「搜尋」而不是「獎勵」：拿 best-of-N **解開的軌跡**當新標籤重訓 | 半天 | 介於 |
| 3 | **DAgger** | 走偏後沒訊號（專家＝CP-SAT，現成免費）| 半天 | ❌ 仍是監督式 |
| 4 | **AlphaZero 式** | 長程規劃本身 | 1–2 天 | ✅ |

**名詞不懂就去 [`notes/01-rl-methods-explained.md`](notes/01-rl-methods-explained.md)**，那裡有白話對照與「為什麼這題 RL 的優勢用不到」。

**微調的具體約束**：跑**全長、不用 curriculum**（`CurriculumState(current_k=None)`，
`_maybe_promote` 對 `None` 是 no-op）。
**前置封鎖已解除**：value head 會不會傷到策略，2026-09-12 量完了——
`--value-coef 0.5` vs `0` 差 **−0.0083**（在 ±0.02–0.04 雜訊內）⇒ **沒有證據支持它有害**，
可以直接用帶 critic 的 checkpoint 去微調。

### 還沒做、也記下來的

- **6×6 的 seed 雜訊仍然沒量過**（用 BC 量只要約 16 分鐘：3 seed × 228s ＋ 評估）。
  在那之前 6×6 的 0.0x 差異一律標「有動但未確證」。
- **多尺寸為什麼有效沒有機制實驗**。事後解釋是「小盤面提供密度更高的長程訊號＝免費的 curriculum」，
  要驗證得另外設計臂（例如 4+6 不含 5）。
- **app image 有 22.9 GB**（基底是 CUDA devel 但 app 不用 GPU），換 slim 基底可大幅縮小，未驗證系統相依。

### ★ BC 還算 RL 嗎？——不算，而且這件事對 goal 有意義

**BC 是模仿學習（IL）裡最簡單的一種，演算法上就是監督式學習**（cross-entropy、固定 `(X, y)`、
無獎勵、無探索、無信用分配）。RL 的定義特徵是「用獎勵優化自己產生的軌跡」，BC 三樣都沒有。

**但**：整條 pipeline 仍是 RL 的（env、遮罩、評估協定沒換），**BC 的定位是暖啟動不是替代品**，
下一步接 PPO 微調就是 RL；而且「先監督再 RL」正是原始 AlphaGo 的做法。

⚠ **誠實的部分**：這個問題**獎勵極稀疏、專家示範免費且完整、解可驗證、mask 後平均分支只有 1.5**
⇒ **RL 的三個典型優勢在這裡全部用不到。** 目前最好的模型仍然是監督式訓練出來的。
**BC → PPO 微調是唯一能推翻這個結論的實驗。**
完整論證見 [BC 報告](reports/2026-09-05_rl-behaviour-cloning.md) §4 與 [`notes/01-rl-methods-explained.md`](notes/01-rl-methods-explained.md)。

### 資料集完整性（2026-09-05 實測複驗，不是引用）

**現行 `seed20300000_n20000_4-6` 乾淨**：三個 split 內部重複 0、兩兩交集 0、digest `--verify` 全過。
分尺寸：**4×4 train 15,419／val 1,927／test 1,928**、**6×6 16,000／2,000／2,000**，train∩test 都是 0。
**id 是內容推導的**（`sample_fingerprint()` 的 canonical JSON），`PuzzleSample` **沒有 id 欄位**。
⚠ 舊的 `main_n1700_456` 有**訓練集內部 4 筆重複 ＋ train∩val 1 筆**（`train∩test` 仍是 0，不影響已發表數字）。
**教訓：「驗過 train∩test」不等於「三個 split 兩兩不交」。**

**授權門檻**：小時級的工作（例如把 6×6 推到全長，估 1–2 小時）**開跑前要問本人**。
分鐘級的訓練（4×4 約 4 分、6×6 約 20 分）可自行執行，**但跑完要回報**。

---

## 4. 環境與驗證（照做就能開工）

worktree `D:\it_project\github_sync\zip-rl` 已存在且已 `uv sync`。從零重建：

```powershell
cd D:\it_project\github_sync\ml-workshop
git worktree add ..\zip-rl feat/rl-a2-training
Copy-Item .\linkedin-zip-challenge\.env ..\zip-rl\linkedin-zip-challenge\.env   # .env 不進版控
cd ..\zip-rl\linkedin-zip-challenge
uv sync
```

**開工第一件事，建立基線**（不要假設環境是好的——本專案曾休眠 9 個月）：

```powershell
cd D:\it_project\github_sync\zip-rl\linkedin-zip-challenge
uv run pytest        # 2026-09-12 實測 276 passed, 8 xfailed
uv run ruff check .  # 期待 All checks passed!
```

- ⚠ **通過數取決於這條 branch 帶了哪些 commit，不是固定值。** 用法是「開工先跑一次記下來，之後拿它比較」。
- **8 個 xfailed 是刻意的**（`xfail(strict=True)` 釘住 env v1 的缺陷）；變成 XPASS 代表有人改了 `rl_env.py`。
- **venv 陷阱**：一律 `cd linkedin-zip-challenge` 再 `uv run`。repo 根的 `.venv` 是 py3.9 devtools，跑不動。
- 相依已就緒（`torch 2.4.1+cu121`、`stable-baselines3 2.7.0`、`sb3-contrib 2.7.1`、`tensorboard`）；**沒有新套件**。

**重建資料集**（`datasets/` 不進版控）：

```powershell
uv run python -m src.core.rl.generate_dataset_v2 --count 1700 --sizes 4,5,6 --timeout 0.5 --name main_n1700_456
```

⚠ 現行**預設資料集是 `seed20300000_n20000_4-6`**（4×4 train 15,419／6×6 16,000）；
重現 A2 的舊數字要加 `--dataset main_n1700_456`。**資料集用 digest 認，不用指令認。**

**本機資源上限 75%**（本人常設要求）。預算只寫在 `train_config.DEFAULT_CPU_FRACTION` 一處，訓練與生成共用。
⚠「N% 的核心數」不等於「N% CPU」——**要在工作實際跑的時候取樣過才算設好**。

---

## 5. 程式地圖

| 檔案 | 用途 |
|---|---|
| `src/core/rl/rl_env_v2.py` | **主角**。一筆畫 env、`action_masks()`、反向 curriculum、死路終止 |
| `src/core/rl/train_config.py` | **改設定只改這裡**：goal（盤面／牆／步數／門檻）、PPO、網路、curriculum、資源上限 |
| `src/core/rl/train_maskable_ppo.py` | PPO 訓練：`GridScalarExtractor`、curriculum callback、checkpoint、評估 |
| `src/core/rl/train_behaviour_cloning.py` | **2026-09-05 新增**。監督式暖啟動，產出與 PPO **同架構、可互換**的 checkpoint（BC 是**第三個** env 建構點，見陷阱 #24）|
| `src/core/rl/solver_service.py` | **2026-09-12 新增。唯一的服務端**：checkpoint 載入（lazy＋快取）、`puzzle` → env、best-of-N rollout。缺模型 503、尺寸不支援 400 |
| `src/core/solvers/registry.py` | **solver 清單的唯一正本**（API／截圖端點／Gradio 都 import 它）|
| `src/core/rl/baselines.py` | masked random／greedy 對照組 ＋ `evaluate()`。**評估 env 只能從 `make_eval_env()` 建** |
| `src/core/rl/generate_dataset_v2.py` | 決定性資料集產生器，**保留 solution path** |
| `src/core/rl/action_space.py` | 共用動作編碼（0:Up 1:Down 2:Left 3:Right）與 `path_to_actions()` |
| `src/core/rl/diagnose_env_v1.py` | A0 的六個 probe，可重跑產生證據 JSON |
| `src/core/tests/rl/` | env 21 個、PPO 訓練 18＋個、**BC 10 個**、v1 診斷 8 個 strict xfail |

**env v2 介面速覽**

```python
env = PuzzleEnvV2(samples, reverse_curriculum_k=None, shaping_lambda=0.2, gamma=0.99,
                  connectivity_features=False)   # True -> scalars 8 -> 10（舊 checkpoint 載不動）
obs, info = env.reset()          # obs = {"grid": (8,8,8) float32, "scalars": (8,) float32}
mask = env.action_masks()        # (4,) bool —— MaskablePPO 直接吃這個方法名
```

- **8 個 grid channel**：valid／wall_right／wall_down／visited／agent／wp_next／wp_future／wp_done
- **reward 冰湖式**：成功 +1、其餘 0，「越快越好」由 γ 表達；死路與超時**沒有懲罰**
- **失敗只有兩種**：死路（四方向全被 mask）與超時（一筆畫下幾乎不會發生，是防呆）

**不要動**：`src/core/utils.py`、`src/core/puzzle_generation/`（共用模組，VLM track 也在用）、
`src/core/rl/` 的 v1 舊檔與 `models/dqn_*.pth`（留作對照）。

---

## 6. ★ 去哪裡查什麼

| 想知道什麼 | 去哪 |
|---|---|
| **量過的數字、踩過的坑**（30 個陷阱 ＋ 28 條已驗證事實 ＋ 已定案的設計決策 ＋ 實驗編年史） | [`rl-traps-and-facts.md`](rl-traps-and-facts.md) ← **動手前掃一遍標題** |
| **名詞看不懂、判讀規則、推論怎麼跑**（做中學筆記）| [`notes/`](notes/)：[概念](notes/01-rl-methods-explained.md)、[判讀紀律](notes/02-reading-the-numbers.md)、[推論與上線](notes/03-inference-and-serving.md)、[決策紀錄](notes/2026-09-11-decisions-rl-track.md) |
| **怎麼把服務起起來** | [`deployment-guide.md`](deployment-guide.md)（Docker 一鍵、驗收指令、常見失敗）|
| 某個實驗**怎麼做、為什麼是那個結論** | [`reports/`](reports/)：**[收尾報告](reports/2026-09-12_rl-wrap-up.md)（最新，多尺寸／value-coef／A5／Docker）**、[A0 env 診斷](reports/2026-08-15_a0-env-v1-findings.md)、[預算與設計筆記](reports/2026-09-05_rl-budget-and-design-notes.md)（**§5 是 AlphaGo 對照與設計理由**）、[連通性特徵](reports/2026-09-05_rl-connectivity-feature.md)、[oracle 上界／best-of-N／搜尋](reports/2026-09-05_rl-lookahead-oracle.md)、**[行為克隆](reports/2026-09-05_rl-behaviour-cloning.md)**（**§4 是「BC 還算不算 RL」的完整論證**、§5 資料集複驗、§6 PPO 能不能成功）|
| **某天做了什麼、量到什麼**（逆時序全記錄，1,900+ 行） | [`dev_log.md`](dev_log.md) ⚠ **不要整份讀**，用日期或關鍵字搜 |
| 專案整體現況、兩條 track 的優先序 | [`roadmap.md`](roadmap.md) |
| 原始作戰計畫、分階段 done 條件、A0–A6 路線 | [`plans/2026-08-15_track-rl-solver.md`](plans/2026-08-15_track-rl-solver.md) ＋ [restart plan（HTML，瀏覽器開）](reports/2026-08-15_rl-restart-plan.html) |
| 專案架構、啟動方式、模組職責 | [`project_guide.md`](project_guide.md) |
| 子專案規範（venv、驗證、紅線、回報格式） | [`../AGENTS.md`](../AGENTS.md)；repo 級 [`../../AGENTS.md`](../../AGENTS.md)、程式風格 [`../../rules.md`](../../rules.md) |
| **某個決定當時為什麼那樣下** | `git log --oneline -- ai-collab/` ＋ 對應 commit message（本 track 的 commit message 寫的是**結論與理由**，不是檔案清單） |
| 程式實際怎麼跑 | 直接讀 `src/core/rl/`——每個檔開頭的 docstring 都寫了它為什麼存在 |
| 常用指令 | [`commands.txt`](commands.txt) |

---

## 7. 工作守則（都是實際犯錯後定下來的）

1. **下結論前先跑掉能推翻它的對照**，還沒排除的要明講。
   例：訓練／held-out 的落差，要先用「沒學過任何一邊」的 baseline 證明兩邊難度相同，才能叫泛化問題。
2. **進行中的 log 不能當結論**——曲線還在跑就外推，這條 track 錯過兩次。
3. **一次只改一件事**；效果小於 ±0.04 不要當成進步。
4. **長時間或吃資源的工作開跑前先問**；跑到一半發現超標，**先停再修**。
5. **告一段落就更新文件再 commit**：`dev_log.md` 記做了什麼、`roadmap.md` 記現況與下一步、本檔記接手要知道的、
   較大的任務出 `reports/`。**不要只留在對話裡。** commit 需**當次授權**，單獨說 commit **不含** push。
6. **解釋要跟數字一起交付**。被問「為什麼這樣設計」時去**讀原始決策文件**，不要憑印象重編理由；
   查完就寫進 `reports/`。判準：**下一個 session 會不會需要再問一次同樣的問題**。

---

## 8. 與 VLM track 的協作約定

| 面向 | 約定 |
|---|---|
| 程式碼 | 我動 `src/core/rl/`；VLM 動 `src/core/vl_models/`、`src/app/`、`src/ui/` |
| ⚠ 交會點 | **A5 會動 `src/app/routers/solver.py`**，動之前先確認 VLM track 沒有同時在改 |
| 共用模組 | `src/core/utils.py`、`src/core/puzzle_generation/` **只讀不改**，要改先提出 |
| 相依 | `pyproject.toml`／`uv.lock` 序列化處理；新套件由本人授權後手動 `uv add` |
| 文件 | dev_log 各加自己的 `###`；roadmap 只改自己那一項；衝突時兩邊都保留 |
| CPU | VLM 的資料生成也吃多核，長工作錯開跑 |

⚠ **A5 還有一個沒解的分布問題**：RL 訓練資料的牆是 0 或 2–5 道，
而 VLM 從真實截圖讀出來的可到 10+ 道 ⇒ **分布外**。
