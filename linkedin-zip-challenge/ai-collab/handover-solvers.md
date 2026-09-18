# Handover — solvers track（把十種 solver 掛上 API）

> **接手這條線只要讀這一份。** Track C 的程式碼於 **2026-09-12** 完成，**2026-09-19** 收尾（文件校正、rebase 到 `main`）；
> 合併進 `main` 由本人照 [`AGENTS.md` §10.5](../../AGENTS.md) Step 5 執行（`git merge --ff-only feat/expose-heuristic-solvers`）。
> 完整的量測與設計理由在 [`reports/2026-09-12_heuristic-solvers-on-the-api.md`](reports/2026-09-12_heuristic-solvers-on-the-api.md)。
> Last Updated: 2026-09-19

---

## 1. 現在的狀態（一句話）

**十種 solver 全部從 `POST /api/solver/solve` 可以叫到**，實測兩題各 10/10 回 200，
而且**凡是畫出圖的回應都通過 `is_solution` 驗證**。

| 類別 | 誰 | 保證 |
|---|---|---|
| `EXACT` | DFS、A\* (heapq)、CP-SAT | 一定給答案或證明無解 |
| `LEARNED` | RL (behaviour cloning) | 不保證；分布外（多牆盤面）會放棄 |
| `HEURISTIC` | ACO、GA、PSO、SA、Tabu、Monte Carlo | 不保證；放棄 ≠ 盤面無解 |

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

### 為什麼裁判要獨立一個模組

**不能用 `calculate_fitness_score()` 當裁判——啟發式正在最佳化它。**
一條靠著分數漏洞拿高分的路，會被同一個漏洞認證為解。實際存在的兩個漏洞：
第一步之後不再檢查 blocked cell；負座標會從網格另一側繞回去索引而不是報錯。

`verify.py` 用的是精確 solver 的停止條件（`dfs.py` 的 base case，`rl_env_v2._is_solved` 同源）：
每個開放格恰好走一次、每步相鄰且不穿牆、從 1 出發、數字依序收集，**不要求結束在最後一個數字上**。

## 3. 已驗證的事實（都是一手實測，不是估計）

| 事實 | 數字 | 出處 |
|---|---|---|
| 不包驗證，單次呼叫回的是真解的比例 | **8.9%（16/180）** | 報告 §2 |
| 不包驗證，會被畫成答案的假路徑 | **164/180 ＝ 91.1%** | 報告 §2 |
| 包了之後，六題的解出率 | **58.3%（21/36）** | 報告 §3 |
| 最慢的單次啟發式呼叫 | 0.128s | 報告 §2 |
| 最慢的整個請求 | **5.04s**（Gradio timeout 是 120s） | 報告 §3 |
| API 往返 | 兩題各 **10/10 回 200**，畫出圖的 100% 通過驗證 | 報告 §9 |

**難度的指標不是盤面大小，是「數字密度高 ＋ 牆少」**：`puzzle_04`（7×7、14 道牆）四種解得出來，
`puzzle_06`（7×7、21 個數字、**0 道牆**）六種全滅。牆砍掉合法後繼步，反而讓隨機走法更好走。

## 4. 已經定案、不要再重開的決策

1. **預算不開成 API 參數。** 六種的預算單位不可共量（隨機走法／迭代／世代×族群／溫度排程），
   秒是唯一共同單位。要開的話開 `budget_seconds`（並設上限），**不是 `attempts`**——
   舊 roadmap 的那條 done 條件已作廢。
2. **同步就夠，不要背景任務。** 最壞 5.04s ≪ 120s。⚠ 這個結論綁在 5 秒這個值上。
3. **`solvable` 是三態，不是布林。** 只有精確 solver 的「無解」能證明圖讀錯了；
   啟發式或 RL 放棄是 `None`（未判定）。schema、router、Gradio 三邊都已對齊。
4. **`_until_verified` 只包 `HEURISTIC`。** RL 不包——它自己有 `DEFAULT_ATTEMPTS` 的取樣邏輯，
   包了會變成雙層重試，預算難以推理。

## 5. 還沒做的事（給下一個人）

### 5.1 Svelte 下拉還寫死三種 ★ 唯一的功能缺口

`src/custom_components/puzzle_editor/frontend/Index.svelte` 裡的選項是寫死的
`DFS`／`A* (heapq)`／`CP-SAT`——**連 RL 都沒有**，更不用說六種啟發式。
Gradio 那邊是從 registry 動態拿的，所以兩個前端現在不一致。

- 正確做法是**讓前端去問後端**，不要再寫死第四份清單。
  後端已經有現成的資料：`registry.SOLVER_ENTRIES` 帶 `name`／`kind`／`note`，
  缺的只是一個回傳它的端點（目前沒有）。
- ⚠ 改完要 `npm run build` 才會反映到 `/svelte-ui`。
- ⚠ 這個檔不在 Track C 的檔案範圍內，所以刻意沒動。

### 5.2 可以做但沒有人要求的

- **`budget_seconds` 請求欄位**（見 §4.1）。做之前先想清楚上限，否則是 DoS 面。
- **同尺度的 solver 比較表進 README**：現在十種在同一個 registry 下可比了，
  但 README 只說「CP-SAT 每一項都贏」，沒有把表放上去。報告 §2/§3 的數字可以直接用。
- **PSO 要不要留著**：它 0/180、0/6，是唯一一個從沒解出過任何一題的。
  留著的理由是「它示範了一個 move 設計錯誤會怎麼樣」——交換兩格會把相鄰路徑拆散，
  它探索的空間裡合法路徑幾乎是零測度集。**這是教材價值，不是功能價值**，要砍要留是取捨不是 bug。

## 6. 陷阱

- ⚠ **`_until_verified` 會全域關掉該 solver module 的 loguru**（`logger.disable(module)`），
  跑完再打開。這是為了不讓幾百次重跑的 DEBUG log 淹掉伺服器記錄，
  但**兩個併發請求打同一種 solver 時，先跑完的那個會提早把 log 打開**。
  影響只有「log 多印一些」，不影響正確性；真的要修就改成 per-call 的 sink filter。
- ⚠ **預算是「跑完一次才檢查」**，所以最壞耗時是 `5s + 一次完整跑的時間`。
  實測 5.04s，因為單次最慢 0.128s。**若哪天有人調大啟發式的預設參數，這個上界會跟著變。**
- ⚠ **測試裡一定要 monkeypatch `HEURISTIC_TIME_BUDGET_SECONDS`**，否則六種啟發式的
  參數化測試會把測試套件拖長 30 秒。現成範例在 `src/app/tests/test_solver_api.py`。
- ⚠ **`models/` 不進版控**：新開的 worktree 沒有它 ⇒ RL solver 回 503（正確行為），
  `src/core/tests/rl/test_solver_service.py` 的端到端測試會 skip。要完整測十種，從 `zip-rl` 複製
  `models/rl_a2/bc_multi_456_e6`（14 MB）過來——`zip-solvers` 在 2026-09-12 做 API 往返（報告 §9）前已經複製了。

## 7. 怎麼驗證你沒弄壞它

```powershell
cd linkedin-zip-challenge
uv run pytest                    # 2026-09-12 基線：304 passed, 1 skipped, 8 xfailed
uv run ruff check .

# 重跑預算量測（約 4 分鐘，seed 已固定在腳本裡）
PYTHONPATH=. uv run python ai-collab/reports/artifacts/heuristic-api-budget/measure_budget.py
```

`src/core/tests/solvers/test_registry.py` 是這條線的守門員：它釘住「三個入口共用一份 registry」、
「`EXACT_SOLVERS` 只有那三種」、「每個啟發式都被包過」。**這三條斷言不要放寬。**
