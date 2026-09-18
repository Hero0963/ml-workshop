# 專案收尾計畫與進度（2026-09-19）

> **這份是收尾工作的進度正本。** context 被壓縮後從這裡接回來：先看「§3 進度」找到停在哪一階段，再看「§2 已查明的事」。
> 分支 `docs/publish-weights-future-work`（worktree `zip-vlm`，已快轉到 `main` 的 `ab6634b`）。
> 本人當次指示：停止一切開發與實驗，只做 fix issue ＋ 收尾；`main`（本機與 remote）要是最新進度；
> 文件齊全；陌生人 `git clone` 後能用 Docker 跑起來；出一份最終報告（驗收角度：作者／repo 訪客／二次開發者）。
> 自主規劃、自己定義 done、自己驗收。

---

## 1. 範圍與 done 條件

| # | 階段 | done 條件 | 驗證方式 |
|---|---|---|---|
| S0 | 盤點 worktree／branch／git | 每個 worktree 的分支、未 commit 改動、是否已併進 `main` 都查清楚 | `git worktree list`、`merge-base --is-ancestor` |
| S1 | 進度檔 | 本檔存在並隨階段更新 | — |
| S2 | 收攏散落的改動 | `zip-solvers`、`zip-infra` 未 commit 的內容移植到本分支（不動那兩個 worktree 本身） | diff 對照 |
| S3 | 修 issue | 陌生人照 README 起服務時會撞到的問題修掉；修不掉的寫進文件 | 全新 clone ＋ `python start.py` 實跑 |
| S4 | Survey | 權重託管方式、GPT-6 computer use、RL 後續路線／AlphaZero 對照、「只准走一次」規則的評估；全部附一手來源 | 每個外部事實有連結與查證日期 |
| S5 | 文件 | README（英／中）、roadmap、handover ×3、deployment-guide、project_guide、dev_log 反映「已收尾」的真實狀態，沒做完的寫成 future work | 逐檔對照、連結檢查 |
| S6 | 最終報告 | `reports/2026-09-19_project-wrap-up.md`，三種讀者各有一節可驗收 | Chrome 看 GitHub 實際渲染 |
| S7 | 驗證 | `pytest`、`ruff`、`pre-commit` 全綠；全新 clone 用 Docker 起得來 | 貼實際輸出 |
| S8 | git 收尾 | 本分支 ff 進 `main`、push；本機 `main` ＝ `origin/main` | `git status -sb` 顯示無 ahead/behind |

**不做**：新功能、訓練、批次評估、超參實驗（含 RL handover §0.3 沒跑完的那些——改列 future work）。
**對外動作不代做**：把模型權重上傳到 Hugging Face／GitHub Release 需要本人帳號，只寫提案與指令。

---

## 2. 已查明的事（S0 結果，2026-09-19 06:20 左右）

### 2.1 git

| worktree | 分支 | HEAD | 未 commit | 已併進 `main`？ |
|---|---|---|---|---|
| `ml-workshop`（主） | `main` | `ab6634b` | 無 | —（**比 `origin/main` 超前 6**，未 push） |
| `zip-vlm`（本 session） | `docs/publish-weights-future-work` | `ab6634b`（已快轉） | 無 | ✅ |
| `zip-rl` | `feat/rl-exit` | `ab6634b` | 無 | ✅ |
| `zip-solvers` | `feat/expose-heuristic-solvers` | `b72d371` | **有**：PSO「包 5 秒預算」量測結果（4 檔改＋`pso-served.json` 新檔，02:57 寫入） | ✅（commit 部分） |
| `zip-infra` | `feat/infra-slim-image` | `babc1cb` | **有**：`deployment-guide.md` 補「2026-09-12 本人定案不補測／不再瘦身」三段 | ✅（commit 部分） |

其餘本機／remote 分支（`dev-hero`、`feat/rl-masked-ppo`、`feat/vlm-parser`、`feat/rl-a2-training`、`feat/rl-ppo-finetune`、
`origin/chore/ai-collab-linkedin-zip`、`origin/fix/session-brief-status-parse`、`origin/research/board-game-rl`）**全部已是 `main` 的祖先**。
沒有訓練在跑；`.agent-heavy-job` 是 `free`；只有一個 Claude session。
Docker：`zip-app-zip-infra` 在 7440 跑舊程式（`babc1cb`）；`zip_ollama_server` 已停 6 天。

### 2.2 基線

`zip-vlm` @ `ab6634b`：`uv sync` 乾淨；`uv run pytest` → **330 passed, 1 skipped, 8 xfailed in 27.47s**。

### 2.3 陌生人 clone 會撞到的問題（待 S3 逐一確認）

1. **沒有 NVIDIA GPU 的機器 `start.py` 會整個中止**：`docker-compose.ollama.yml` 要求 nvidia device，
   而 `start.py` 起 ollama 用 `check=True` ⇒ 失敗就 `sys.exit`，app 根本不會起（待實測／查證錯誤訊息）。
2. **VLM 權重無法取得**：`zip-qwen35-4b-p4c:f16` 只存在本機 Ollama volume；GGUF 文字塔 8.42 GB ＋ mmproj 672 MB。
3. **RL 權重「一個指令重生」對陌生人不成立**：`goal3_multi` 讀 `datasets/rl_datasets_v2/seed20300000_n20000_456`（34 MB，不進版控），
   而產生器是「wall-clock 逾時 ⇒ 只在負載不變時決定性」，重生出來是**另一包題目**。服務用的 checkpoint 只有 **14 MB**。
4. README 結果表還是 10 epochs 的 `bc_multi_456`（6×6 bo32 0.8535），服務中的是 `bc_multi_456_e6`（0.9465）；
   「What's next」寫的 PPO 微調已做完；測試數寫 276。
5. `roadmap.md` 標頭「最後更新 2026-09-05」、§現況表多處過時；`deployment-guide.md` §2 寫「4 種 solver」。
6. **已發表的 RL 評分腳本在不進版控的 `hi-collab/scratch/`**（`probe_cross_size.py` 等）⇒ 二次開發者無法從 repo 重現報告數字。

### 2.4 RL 沒做完的實驗（改列 future work，不跑）

`handover-rl-solver.md` §0.3：嚴格尺 BC vs ExIt 對照（只量完 e4）、6×6 seed 雜訊、贏家補 4×4／5×5、ExIt 第二輪。

---

## 3. 進度

- [x] S0 盤點（06:20）
- [x] S1 本檔
- [x] S2 收攏散落改動（`git apply --3way`；五個檔的 hunk 與原 worktree 逐行相同、`pso-served.json` `cmp` 相同；原 worktree 未動）
- [ ] S3 修 issue
- [ ] S4 Survey
- [ ] S5 文件
- [ ] S6 最終報告
- [ ] S7 驗證
- [ ] S8 git 收尾

## 4. 各階段筆記（邊做邊記）

（空）
