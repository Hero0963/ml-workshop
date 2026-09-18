# Handover — solvers track（把 solver 掛上 API：九種上線，PSO 保留實作不開放）

> **接手這條線只要讀這一份。** Track C 的程式碼於 **2026-09-12** 完成，**2026-09-19** 收尾（文件校正、rebase 到 `main`）；
> **2026-09-19 已 `--ff-only` 合併進 `main` 並 push**（`a8c3626..78c36c6`，含 Svelte 編輯器，見 §5.1）。
> 完整的量測與設計理由在 [`reports/2026-09-12_heuristic-solvers-on-the-api.md`](reports/2026-09-12_heuristic-solvers-on-the-api.md)。
> Last Updated: 2026-09-19

---

## 1. 現在的狀態（一句話）

**九種 solver 從 `POST /api/solver/solve`、截圖端點、Gradio、Svelte 都叫得到**，
而且**凡是畫出圖的回應都通過 `is_solution` 驗證**（2026-09-12 十種上線時實測兩題各 10/10 回 200）。
**2026-09-19 本人定案：PSO 不開放**，但實作、測試與量測都保留（見 §4.6）。

| 類別 | 誰 | 保證 |
|---|---|---|
| `EXACT` | DFS、A\* (heapq)、CP-SAT | 一定給答案或證明無解 |
| `LEARNED` | RL (behaviour cloning) | 不保證；分布外（多牆盤面）會放棄 |
| `HEURISTIC` | ACO、GA、SA、Tabu、Monte Carlo | 不保證；放棄 ≠ 盤面無解 |
| （不上線） | PSO | 有實作、有測試、有量測；不在 registry 裡 |

## 2. 架構：三個東西，各自只做一件事

```
src/core/solvers/registry.py     ← 唯一正本。SOLVER_ENTRIES 一改，三個入口一起變
  ├─ SolverEntry(name, solve, kind, note)
  ├─ _until_verified(solve)      ← 只包 HEURISTIC：重跑到通過驗證，或用完 5 秒
  ├─ SOLVERS                     ← name -> fn，所有呼叫端用的形狀
  └─ EXACT_SOLVERS               ← 只有 kind == EXACT，給預設值與「不能顯示錯誤」的地方用

src/core/solvers/verify.py       ← is_solution(puzzle, path)：唯一的裁判
```

三個入口（**不要再各抄一份清單**，2026-09-12 前正是因為抄了三份才會加一處另外兩處不動）：
`src/app/routers/solver.py`、`src/app/routers/vision.py`、`src/ui/gradio_app.py`。
第四個前端 Svelte 編輯器不 import Python，所以它**透過 `GET /api/solver/list` 問**（2026-09-19 起）。

### 為什麼裁判要獨立一個模組

**不能用 `calculate_fitness_score()` 當裁判——啟發式正在最佳化它。**
一條靠著分數漏洞拿高分的路，會被同一個漏洞認證為解。實際存在的兩個漏洞：
第一步之後不再檢查 blocked cell；負座標會從網格另一側繞回去索引而不是報錯。

`verify.py` 用的是精確 solver 的停止條件（`dfs.py` 的 base case，`rl_env_v2._is_solved` 同源）：
每個開放格恰好走一次、每步相鄰且不穿牆、從 1 出發、數字依序收集、**結束在最後一個數字上**。
⚠ 最後這條是 **2026-09-19 本人定案後才加的**（RL track 的 `feat/rl-exit`）：之前全專案 7 個判定器（env、`dfs.py`、A* 兩版、`cp.py`、
fitness jackpot、本檔、VLM 的 `path_is_legal`）都接受「收完最大數字還繼續走」的路；現在一律要求停在最大數字，
由 `src/core/tests/test_end_on_last_number.py` 一次釘住。細節見 `dev_log.md` 2026-09-19 RL 小節。

## 3. 已驗證的事實（都是一手實測，不是估計）

| 事實 | 數字 | 出處 |
|---|---|---|
| 不包驗證，單次呼叫回的是真解的比例 | **8.9%（16/180）** | 報告 §2 |
| 不包驗證，會被畫成答案的假路徑 | **164/180 ＝ 91.1%** | 報告 §2 |
| 包了之後，六題的解出率 | **58.3%（21/36）** | 報告 §3 |
| 最慢的單次啟發式呼叫 | 0.128s | 報告 §2 |
| 最慢的整個請求 | **5.04s**（Gradio timeout 是 120s） | 報告 §3 |
| API 往返 | 兩題各 **10/10 回 200**，畫出圖的 100% 通過驗證 | 報告 §9 |
| PSO 單次呼叫（RL test split，決定性） | 4×4 **0.590**（1,139／1,931）、6×6 **0.000**（0／2,000） | [PSO 報告](reports/2026-09-19_pso-not-served.md) §1 |
| PSO 包 5 秒預算（各抽 200 題） | 4×4 **1.000**（200／200）、6×6 **0.030**（6／200）——不開放的理由是 6×6 | [PSO 報告](reports/2026-09-19_pso-not-served.md) §1 |

**難度的指標不是盤面大小，是「數字密度高 ＋ 牆少」**：`puzzle_04`（7×7、14 道牆）四種解得出來，
`puzzle_06`（7×7、21 個數字、**0 道牆**）六種全滅。牆砍掉合法後繼步，反而讓隨機走法更好走。

## 4. 已經定案、不要再重開的決策

1. **預算不開成 API 參數。** 六種的預算單位不可共量（隨機走法／迭代／世代×族群／溫度排程），
   秒是唯一共同單位。要開的話開 `budget_seconds`（並設上限），**不是 `attempts`**——
   舊 roadmap 的那條 done 條件已作廢。**2026-09-19 本人定案：`budget_seconds` 也不做**
   （沒人需要，開了就多一個要設上限的 DoS 面；想看「多給時間能多解幾題」就離線改 `measure_budget.py` 的預算跑）。
2. **同步就夠，不要背景任務。** 最壞 5.04s ≪ 120s。⚠ 這個結論綁在 5 秒這個值上。
3. **`solvable` 是三態，不是布林。** 只有精確 solver 的「無解」能證明圖讀錯了；
   啟發式或 RL 放棄是 `None`（未判定）。schema、router、Gradio 三邊都已對齊。
4. **`_until_verified` 只包 `HEURISTIC`。** RL 不包——它自己有 `DEFAULT_ATTEMPTS` 的取樣邏輯，
   包了會變成雙層重試，預算難以推理。
5. **不要加 API 參數 `budget_seconds`**（2026-09-19 本人定案，理由見 4.1）。
6. **PSO 不開放、但保留**（2026-09-19 本人定案）。它的移動是「交換路徑上兩格」，會把相鄰的路徑拆成跳躍的步，
   所以 6×6 幾乎解不出來；留在程式裡當「移動方式不守約束會怎樣」的教材。
   `test_registry.py::test_pso_is_implemented_but_not_served` 釘住這個決定，要加回去先讀 [`reports/2026-09-19_pso-not-served.md`](reports/2026-09-19_pso-not-served.md)。

## 5. 還沒做的事（給下一個人）

### 5.1 ~~Svelte 下拉還寫死三種~~ ✅ 2026-09-19 已解（計畫書 C2）

Svelte 編輯器現在開啟時打 `GET /api/solver/list`（直接由 `SOLVER_ENTRIES` 產生），下拉依 `kind` 分組、
顯示所選 solver 的 `note`；拿不到清單就顯示錯誤並停用 Solve——**刻意不退回寫死的清單**，那就是第四份。
放棄的回應（API 沒附 GIF）改成獨立的「No solution found」框，不再放在「Solution」底下配兩張破圖。

- **守門**：`test_registry.py::test_the_svelte_editor_keeps_no_copy_of_the_list` 讀 `Index.svelte`，
  裡面出現任何一個 registry 裡的名稱就失敗（改之前實測會抓到 `DFS`／`A* (heapq)`／`CP-SAT`）。
- **端到端**：[`reports/artifacts/svelte-solver-list/`](reports/artifacts/svelte-solver-list/) 用 headless Chrome 操作建置版：
  4×4 可解盤面上**十種全部畫出答案、10/10 通過 `is_solution`**；把一格用牆隔開之後，精確／學習／啟發式各一種
  都顯示「No solution found」（啟發式 5.07s，用滿預算）。
- ⚠ 改 `Index.svelte` 後要 `npm ci`（第一次）＋ `npm run build` 才會反映到 `/svelte-ui`；`dist/` 不進版控。

### 5.2 ~~PSO「包 5 秒預算」的分數待補~~ ✅ 2026-09-19 02:53 量完

4×4 **1.000**、6×6 **0.030**（各抽 200 題），數字與解讀在 PSO 報告 §1、§4。要重量的話照下面的步驟
（這個預算是牆鐘，別人佔著 CPU 時量出來會被壓低）——**號誌是 `free` 時**：改成 `busy solvers <時間>` →
`PYTHONPATH=. uv run python ai-collab/reports/artifacts/pso-score/measure_pso.py served <dataset 目錄>`（8 個 worker）→ 改回 `free`，
結果寫進 `pso-served.json`。
資料集只在跑過 RL 的 worktree 裡（例如 `zip-rl` 的 `datasets/rl_datasets_v2/seed20300000_n20000_456`）。
PSO 已經不上線，所以這個數字**只影響紀錄，不影響服務**。

### 5.3 可以做但沒有人要求的

- **同尺度的 solver 比較表進 README**：現在上線的九種在同一個 registry 下可比了，
  但 README 只說「CP-SAT 每一項都贏」，沒有把表放上去。報告 §2/§3 的數字可以直接用。
- ~~`budget_seconds` 請求欄位~~、~~PSO 要不要留著~~：**2026-09-19 都已定案**，見 §4.5、§4.6。

## 6. 陷阱

- ⚠ **`_until_verified` 會全域關掉該 solver module 的 loguru**（`logger.disable(module)`），
  跑完再打開。這是為了不讓幾百次重跑的 DEBUG log 淹掉伺服器記錄，
  但**兩個併發請求打同一種 solver 時，先跑完的那個會提早把 log 打開**。
  影響只有「log 多印一些」，不影響正確性；真的要修就改成 per-call 的 sink filter。
- ⚠ **預算是「跑完一次才檢查」**，所以最壞耗時是 `5s + 一次完整跑的時間`。
  實測 5.04s，因為單次最慢 0.128s。**若哪天有人調大啟發式的預設參數，這個上界會跟著變。**
- ⚠ **測試裡一定要 monkeypatch `HEURISTIC_TIME_BUDGET_SECONDS`**，否則上線的啟發式的
  參數化測試會把測試套件拖長 30 秒。現成範例在 `src/app/tests/test_solver_api.py`。
- ⚠ **`models/` 不進版控**：新開的 worktree 沒有它 ⇒ RL solver 回 503（正確行為），
  `src/core/tests/rl/test_solver_service.py` 的端到端測試會 skip。要完整測九種，從 `zip-rl` 複製
  `models/rl_a2/bc_multi_456_e6`（14 MB）過來——`zip-solvers` 在 2026-09-12 做 API 往返（報告 §9）前已經複製了。

## 7. 怎麼驗證你沒弄壞它

```powershell
cd linkedin-zip-challenge
uv run pytest                    # 2026-09-19 基線（C2 之後、有 models/）：312 passed, 8 xfailed
                                 # 沒有 models/ 時 RL 端到端測試會變 skip（見 §6）
uv run ruff check .

# 重跑預算量測（約 4 分鐘，seed 已固定在腳本裡）。⚠ 它從 registry 取啟發式，
# 所以 2026-09-19 之後重跑不含 PSO；PSO 的舊數字留在已提交的 budget-measurements.json
PYTHONPATH=. uv run python ai-collab/reports/artifacts/heuristic-api-budget/measure_budget.py

# Svelte 編輯器端到端（約 15 秒）：先 build、起服務，再用 headless Chrome 操作建置版
cd src/custom_components/puzzle_editor/frontend; npm ci; npm run build; cd ../../../..
uv run uvicorn src.app.main:app --port 7452          # 另一個終端機
PYTHONPATH=. uv run python ai-collab/reports/artifacts/svelte-solver-list/drive_editor.py <repo 外的暫存目錄>

# PSO 在 RL held-out 4×4／6×6 上的分數（資料集只在跑過 RL 的 worktree 裡，例如 zip-rl）
PYTHONPATH=. uv run python ai-collab/reports/artifacts/pso-score/measure_pso.py raw <dataset 目錄>     # 約 2 分鐘，決定性
PYTHONPATH=. uv run python ai-collab/reports/artifacts/pso-score/measure_pso.py served <dataset 目錄>  # 8 個 worker，先看號誌
```

`src/core/tests/solvers/test_registry.py` 是這條線的守門員：它釘住「三個入口共用一份 registry」、
「`EXACT_SOLVERS` 只有那三種」、「每個啟發式都被包過」。**這三條斷言不要放寬。**
