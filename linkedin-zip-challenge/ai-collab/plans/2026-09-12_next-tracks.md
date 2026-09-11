# 任務計畫 — 可平行的三條 track（2026-09-12）

> **給接手的 agent：只讀屬於你那一節就夠了。**
> 平行開發的規則（worktree、資源號誌、共用檔案協定）在 [`../../../AGENTS.md` §10](../../../AGENTS.md)。
>
> **現況**：`main` = `origin/main` = `5d43511`。三條 track 的**檔案零交集**，可以同時進行。

---

## 0. 一覽

| Track | 分支 | worktree | 做什麼 | GPU |
|---|---|---|---|---|
| **A** | `feat/rl-ppo-finetune` | `zip-rl` | BC → PPO 微調：回答「RL 在這題到底有沒有加值」 | ✅ 要搶號誌 |
| **B** | `feat/infra-slim-image` | `zip-infra` | app image 22.9 GB 瘦身 ＋ 實跑驗證 `start.py --dev` | ❌ |
| **C** | `feat/expose-heuristic-solvers` | `zip-solvers` | 把六種啟發式 solver 掛上 API（4 → 10 種） | ❌ |
| ~~D~~ | — | — | 視覺評估集變難 —— **未排程，沒有明確指派就不要開工** | — |

**檔案所有權（切分的依據，不要越界）**

| Track | 擁有 | **不准動** |
|---|---|---|
| A | `src/core/rl/`（**`solver_service.py` 除外**）、`logs/rl_*`、`models/rl_a2/` | `src/app/`、`src/core/solvers/`、Docker 檔 |
| B | `.devcontainer/`、`docker-compose*.yml`、`.dockerignore`、`start.py`、`ai-collab/deployment-guide.md` | `src/` 底下任何東西、`README*.md` |
| C | `src/core/solvers/`、`src/app/`、`src/core/tests/solvers/`、`README.md` 與 `README_zh-TW.md` 的 **solver 表格那一段** | `src/core/rl/`、Docker 檔 |

三條都會碰 `ai-collab/dev_log.md` 與 `roadmap.md` ⇒ **各加各的 `###`、只改自己那一項**。

**共同的開工動作**（每條都一樣）：

```powershell
cd linkedin-zip-challenge
uv run pytest        # 2026-09-12 基準：276 passed, 8 xfailed
uv run ruff check .  # All checks passed!
```

---

## Track A — BC → PPO 微調（RL 主線）

**為什麼做**：本人 2026-09-11 定案「**就是要用 RL 做**」。目前最好的模型是**監督式**訓練出來的
（BC 用 1/9 成本贏過 PPO），所以「這題不該用 RL」目前只是**假設**。
**BC → PPO 微調是唯一能推翻它的實驗**——兩個方向都是交付物。

**先讀**：[`../handover-rl-solver.md`](../handover-rl-solver.md) §1–§3 ＋
[`../notes/01-rl-methods-explained.md`](../notes/01-rl-methods-explained.md) §4。

**要做的事**

1. 實作 `train_maskable_ppo.py` 的 `--init-from <run-id>`：用 BC 的 checkpoint 當 PPO 起點。
2. 微調要跑**全長、不用 curriculum**（`CurriculumState(current_k=None)`，`_maybe_promote` 對 `None` 是 no-op）。
3. 從 `bc_multi_456` 出發，**先在 4×4 跑分鐘級的**確認不會崩，再談 6×6。
4. 對照組是 **BC 自己**（同一個 checkpoint 未微調），deterministic 與 best-of-32 兩個數字都要報。

**done 條件**

- [ ] `--init-from` 有測試：載入後的 policy 權重與來源 checkpoint 相同
- [ ] 至少一組「BC vs BC+PPO」的對照數字落盤在 `logs/rl_probes/`
- [ ] 明確回答 **PPO 微調買到了什麼**；買不到就寫清楚「這題不該用 RL」現在有證據了

**資源**：需要 GPU，**要搶 `../ml-workshop/.agent-heavy-job` 號誌**。
4×4 是分鐘級可自行跑；**6×6 全長是小時級，開跑前要本人授權**。

**已知陷阱**

- ⚠ PPO 用 `V(s)` 算 advantage。value head 已在 BC 訓好（`--value-coef 0.5` vs `0` 差 **−0.0083**，
  在 ±0.02–0.04 雜訊內）⇒ **直接用帶 critic 的 checkpoint，不要再從零訓 critic**。
- ⚠ **6×6 的 seed 雜訊從沒量過**（用 BC 量約 16 分鐘：3 seed × 228s ＋ 評估）。
  沒量之前，6×6 的 0.0x 差異一律標「**有動但未確證**」。
- ⚠ **不要外推**。這條 track 外推錯過三次，最近一次差 0.037。
- ⚠ env 有**三個建構點**（訓練、評估、BC）。改觀測形狀要三個都改。

---

## Track B — app image 瘦身 ＋ 驗證開發環境

**為什麼做**：服務已經能一鍵起，但 **image 有 22.9 GB**——基底是
`pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`（約 10 GB，含 CUDA toolkit），
`uv sync` 之後又裝一份 `torch 2.4.1+cu121`。而 **app 容器根本不用 GPU**（GPU 在 ollama 那邊，
RL 推論是 `device="cpu"`）。

**先讀**：[`../deployment-guide.md`](../deployment-guide.md)。

**要做的事**

1. 換掉 `.devcontainer/Dockerfile` 與 `Dockerfile.dev` 的基底（例如 `python:3.11-slim`），
   補上必要的系統函式庫後重建，量前後大小。
   ⚠ **未驗證**：gradio／matplotlib／opencv 可能需要 `libgl1`、`libglib2.0-0` 之類。
   **時間盒：兩次建置失敗就退回可動版本，把失敗原因寫進 `deployment-guide.md`。**
2. **實跑驗證 `python start.py --dev`**——目前**只驗過 `--status` 這條路徑**，完整啟動沒跑過。
3. 兩份 compose 都要能起，`/api/echo/health` 要 200。

**done 條件**

- [ ] `docker images` 的前後大小對照（貼實際輸出），或「試過但退回」的理由
- [ ] `python start.py --dev` 完整啟動 ＋ health 200 的實際輸出
- [ ] `python start.py`（正式）仍然能起，四種 solver 都回 200
- [ ] `deployment-guide.md` 的數字同步更新

**資源**：不吃 GPU，**不用搶號誌**。Docker build 吃網路與 IO。

**已知陷阱**

- ⚠ **正式 image 把原始碼烤進去**，改完程式要 `--build`，`restart` 不會更新。
- ⚠ **新 worktree 沒有 `models/`**（不進版控）⇒ RL solver 會回 **503**，這是正確行為。
  要完整測 RL solver 的話，從 `zip-rl` 複製 `models/rl_a2/bc_multi_456`（14 MB）過來。
- ⚠ `.env` 不進版控、每個 worktree 一份，**會各自過期**。

---

## Track C — 把六種啟發式 solver 掛上 API

**為什麼做**：專案實作了 **10 種 solver，但 API 只上線 4 種**。
六種啟發式（蟻群／基因／粒子群／模擬退火／禁忌搜尋／蒙地卡羅）都已實作且有測試，
只差沒有進 registry——掛上去之後才能做同尺度的比較。

**先讀**：`src/core/solvers/registry.py`（唯一正本）＋ `AGENTS.md §5` 的任務地圖。

**要做的事**

1. 在 `src/core/solvers/registry.py` 為六種啟發式各加一個 `SolverEntry`。
2. ⚠ **它們的簽名和精確解法不同**，而且彼此也不同：
   `solve_puzzle_monte_carlo(puzzle, attempts=1000)`、
   `solve_puzzle_tabu_search(puzzle, num_iterations=100, tabu_list_size=15, ...)`。
   `registry.SolverFn` 目前是 `Callable[..., ...]`，而 router 只用 `solver_func(puzzle_data)` 呼叫
   ⇒ **靠預設值可以先跑起來**；要不要把「預算」開成 API 參數（`SolverRequest` 加欄位）
   **是這條 track 要決定並寫下理由的設計問題**。
3. `kind` 需要第三類（目前只有 `EXACT` / `LEARNED`）——啟發式**不精確也不保證**，
   `EXACT_SOLVERS` 的語意不能被破壞（`test_registry.py` 有釘）。
4. 更新兩份 README 的 solver 表格（「Served today」那一欄）。

**done 條件**

- [ ] 十種 solver 都能從 `POST /api/solver/solve` 叫到，**貼出每一種的實際回應**
- [ ] `test_registry.py` 擴充並通過；`uv run pytest` 全綠
- [ ] 若加了 API 參數，`src/app/tests/test_solver_api.py` 要涵蓋
- [ ] 兩份 README 的 solver 表格更新

**資源**：不吃 GPU，**不用搶號誌**。

**已知陷阱**

- ⚠ 啟發式在小盤面上**會輸給 CP-SAT**，掛上去是為了比較不是為了效能——
  `SolverEntry.note` 要誠實寫清楚。
- ⚠ 有些啟發式**跑很久**（`num_iterations` 預設 100）。API 有 timeout，
  Gradio 那邊 `timeout=120`——預設值要挑得讓一題在那之內跑完。
- ⚠ **不要動 `src/core/utils.py`**（`calculate_fitness_score` 牽動全部啟發式）。

---

## ~~Track D — 把視覺評估集變難~~（未排程）

> **VLM track 已完成並合併**（讀圖 2026-08-29 上線；`feat/vlm-parser` 有 **0 個 commit** 沒進 main）。
> 這是從 README *What's next* 抄下來的**選項**，**沒有本人明確指派就不要開工**。
> 內容：合成 held-out 四層指標全部飽和在 1.000，需要加視覺雜訊／多渲染風格／失真／大盤面
> 才能恢復鑑別力。要做的話先讀 [`../handover-vlm-parser.md`](../handover-vlm-parser.md)。
