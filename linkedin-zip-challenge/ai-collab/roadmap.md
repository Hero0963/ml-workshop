# 現況與下一步 — linkedin-zip-challenge

> **新 session 的第一站。** 每次工作告一段落就更新這裡（現況一句話、下一步順序、進度日誌加一列）。
> 架構與啟動方式看 [project_guide.md](project_guide.md)、完整開發歷程看 [dev_log.md](dev_log.md)、規範看 [../AGENTS.md](../AGENTS.md)。
> 最後更新：2026-08-08

## 現況（2026-08-08）

核心功能**已完成且可跑**：9 種 solver、FastAPI 後端、Gradio 主控台、Svelte WYSIWYG 編輯器、程序化出題、GIF／PNG 視覺化、Docker 雙環境。

> **本專案自 2025-10-30 之後停了約 9 個月**，2026-08-08 重建 AI 協作骨架（本檔＋`AGENTS.md`＋`project_guide.md`＋`commands.txt`）並**完成端到端復原驗證**：
> `uv sync` 乾淨（resolved 190／audited 172）、Python **3.11.13**、`uv run pytest` **46 passed in 8.10s**、`ruff` 全綠；
> 服務實際起來打過 API（三種 solver 對 `puzzle_01` 都回傳與 ground truth **逐格相同的 36 步路徑**），
> Gradio `/ui`、Svelte `/svelte-ui`、Swagger `/docs` 三個頁面都用 Chrome headless 驗過渲染。
> **基線是好的，可以直接接著開發。**

| 項目 | 狀態 |
|------|------|
| 精確解 solver（DFS／A\*(heapq)／A\*(SortedList)／CP-SAT） | ✅ 完成，有 ground-truth 單元測試 |
| 啟發式 solver（Monte Carlo／SA／GA／Tabu／PSO／ACO） | ✅ 完成，各有 smoke test |
| Fitness function（`calculate_fitness_score`） | ✅ 完成，含非連續路徑懲罰 |
| FastAPI 後端（`/api/solver/solve`、`/api/echo`） | ✅ 完成 |
| Gradio 主控台（`/ui`，4 分頁） | ✅ 完成：出題／naive 解題／互動解題／Echo |
| Svelte 互動編輯器（`/svelte-ui`，Canvas WYSIWYG） | ✅ 完成，已整合進 FastAPI 靜態掛載 |
| 程序化出題（隨機回溯 Hamiltonian path ＋ 多核） | ✅ 完成 |
| Docker 雙環境（dev hot-reload／prod multi-stage） | ✅ 完成，2025-10-30 驗證過 |
| **API 只掛了 3/9 種 solver** | ⚠ 已知落差（`src/app/routers/solver.py` 的 `SOLVERS` 只有 DFS／A\*(heapq)／CP-SAT）；**Gradio 與 Svelte 兩邊的下拉選單也同樣只有 3 種**（2026-08-08 實測確認） |
| Swagger `/docs` 的 Echo 端點重複兩份 | 🐞 小 bug：`main.py` 掛 router 時給 `tags=["Echo"]`，而 `echo.py` 的 router 自帶 `tags=["echo"]`，FastAPI 合併後產生兩個群組 |
| Svelte UI 的 Instructions 顯示原始 Markdown | 🐞 小 bug：`**middle**`／`**border**` 直接印出星號，該處沒有走 Markdown 渲染 |
| RL solver（`src/core/rl/`） | ⏸ **刻意暫停**（2025-10-15，見下方決策） |
| VL 圖片解析（`src/core/vl_models/`） | 🧪 **實驗性 scratchpad**，技術路線已驗證可行但未整合進主流程 |
| 環境復原驗證（9 個月未動） | ✅ **2026-08-08 完成**：46 tests passed、ruff 全綠 |

## 下一步

1. **把 9 種 solver 全部掛進 API**（最高優先——功能已經寫好且測過，只差沒接出來）
   - 為什麼：`src/core/solvers/` 有 9 種實作、每種都有測試，但 `src/app/routers/solver.py` 的 `SOLVERS` dict 只暴露 3 種。
     2026-08-08 實測佐證：打 `POST /api/solver/solve` 指定 `Simulated Annealing` → **404 `Solver 'Simulated Annealing' not found.`**；
     Svelte 前端 bundle 內的下拉選項字串也只有 `DFS`／`A* (heapq)`／`CP-SAT`。**六種啟發式解法目前完全無法從介面觸及。**
   - 注意：啟發式 solver 需要 `attempts` 參數，API schema（`src/app/schemas/solver.py`）要一併擴充，且要考慮逾時（同步阻塞可能拖很久）。
     改完 Svelte 的下拉要重新 `npm run build` 才會反映。
   - Done 條件：`SOLVERS` 含全部 9 種、schema 支援 `attempts`、Gradio 與 Svelte 兩邊下拉都可選、新增對應 API 測試且全綠。

2. **VL 圖片解析整合進主流程**
   - 現況：`src/core/vl_models/` 是實驗腳本堆，技術路線已驗證（見下方決策），但沒有正式的 parser 進 API／UI。
   - Done 條件：一個穩定的 `image → Puzzle dict` 函式 ＋ 單元測試 ＋ Gradio 上傳分頁；實驗腳本移進 `ai-collab/` 或標明為 scratchpad。

3. **RL 重啟前置研究**（不急）
   - 先讀 `../more_simple_reinforcement_learning/` 的 DQN 與 PPO 章節，再看 AlphaGo／AlphaZero 架構。
   - 重啟時的簡化方向（2025-10-15 已想好）：縮小 `map_size`、放寬環境限制（允許重走，從 Hamiltonian path 降級為一般尋路）。
   - Done 條件：寫一份「RL 重啟方案」到 `ai-collab/reports/`，說明改哪些設計、為什麼這次不會再掉進 policy loop。

4. **測試報告目錄整理**（雜務）
   - `run_tests.bat` 會把報告寫進 `src/core/tests/reports/`，堆久了會亂。決定要不要納入 `.gitignore`。

## 已定案，不要再重開的決策

| 決策 | 理由 |
|------|------|
| **RL 暫停，不是放棄** | 2025-10-15。根因是 **deterministic policy loop**：距離型 reward shaping 造成「獎勵陷阱」，即使把權重從 0.1 降到 0.01 仍會讓策略卡在小迴圈。訓練期看起來好只是 ε-greedy 的隨機性意外把 agent 撞出迴圈。**不要再靠調 reward 權重硬解**，要先補理論 |
| **VL 採「混合策略」，不用 tool-calling** | 2025-10-24 實測：`bsahane/Qwen2.5-VL-7B-Instruct` 支援 tool-calling 但**視覺模組壞掉**（回傳結構正確但空的物件）；`openbmb/minicpm-o2.6` 視覺正常但**不支援 tools**（400 Bad Request）。結論：用 `pydantic-ai` 但把 `output_type` 設成 `str`，靠 prompt engineering 要模型吐 JSON 字串再自己 parse——已驗證完全成功 |
| **Docker 雙環境（dev／prod 分開）** | 2025-10-28。`docker-compose.dev.yml` 兩容器＋volume mount 換 hot-reload；`docker-compose.yml` 單一 multi-stage 映像檔換 production 乾淨。不要合併成一個 |
| **牆一律叫 `walls`** | 曾與 `blocked_cells` 混用造成前後端資料格式不符。`walls` ＝牆，`blocked_cells`／障礙是另一回事，不可互換 |
| **lint 只用 `ruff`** | 2025-10-01。`black`＋`isort`＋`ruff` 三者互相打架造成格式化無限迴圈，已移除前兩者，`ruff` 一手包辦 lint／import 排序／格式化 |
| **`pytest` 路徑設定放 `pytest.ini`** | 用 `pythonpath = .`，不要回頭在 `conftest.py` 動 `sys.path`（那是舊做法，會在測試收集階段炸 `ModuleNotFoundError`） |
| **不進 root uv workspace** | 本專案鎖 Python 3.11 ＋ `torch==2.4.1`／cu121 自訂 index，與 repo 根的 3.9 衝突。維持獨立 `.venv`＋`uv.lock`（見 `../AGENTS.md §5`） |
| **不做「兵器庫式」solver 比較報告** | 目前無此需求；要比較請先確認目的是效能數據還是教學說明 |

## 開放問題（想到就補，不急著答）

- 啟發式 solver 掛進 API 後，逾時要怎麼處理？（同步阻塞 vs 背景任務）
- `puzzle_dataset/`／`zip_puzzles/` 已累積多批資料集且不進版控，要不要定個保留策略？
- Svelte 前端目前沒有自己的測試，值得補嗎？

## 進度日誌（摘要；完整版見 [dev_log.md](dev_log.md)）

| 日期 | 事件 |
|------|------|
| 2025-09-20 | 專案初始化：DFS 回溯解題器 ＋ pytest ＋ loguru 測試報告流程 |
| 2025-09-21 ~ 09-23 | 輸入系統重構（文字版面 → parser）、6 題 ground truth 進 `conftest.py`、GIF 動畫視覺化 |
| 2025-09-25 | A\* ＋ CP-SAT（AddCircuit ＋ dummy node）完成；solver 移進 `src/core/solvers/` |
| 2025-10-01 | 啟發式框架：fitness function ＋ Monte Carlo baseline；工具鏈統一為 `ruff` |
| 2025-10-04 | 啟發式擴充：SA（truncate-and-regrow 修好路徑不連續）、GA（no-crossover）、Tabu、PSO |
| 2025-10-08 | 程序化出題：隨機回溯 Hamiltonian path ＋ retry/decrement ＋ 內部逾時；多核資料集腳本 |
| 2025-10-12 ~ 10-13 | RL 深度除錯：確診 deterministic policy loop，兩次調 reward shaping 皆失敗 |
| 2025-10-15 | **RL 暫停**；專案轉向服務化（FastAPI ＋ Gradio） |
| 2025-10-16 | 服務化第一階段：FastAPI ＋ `pydantic-settings` ＋ routers/schemas 分層 ＋ Gradio 多分頁 |
| 2025-10-20 | Gradio 互動編輯器（WYSIWYG）、即時預覽、大量 bug 修正 |
| 2025-10-21 | `run_docker_dev.py` 一鍵啟動開發環境 |
| 2025-10-22 ~ 10-24 | VL 模型驗證：兩個模型各缺一半能力 → **混合策略**（prompt engineering 產 JSON）驗證成功 |
| 2025-10-28 | Production-ready 重構：設定集中化、DRY（`prepare_solver_input`）、Svelte 整合、Docker 雙環境 |
| 2025-10-30 | 文件精修 ＋ 本機／Docker dev／Docker prod 三種環境全部實測驗證 |
| 2026-08-08 | **建立 AI 協作骨架**：`AGENTS.md`／`CLAUDE.md`／`ai-collab/`（roadmap・project_guide・dev_log・commands），`dev_log.md` 移入 `ai-collab/`，補 `.python-version` |
