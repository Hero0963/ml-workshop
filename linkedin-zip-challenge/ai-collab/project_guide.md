# 專案指南 (Project Guide) — linkedin-zip-challenge

> 架構、模組職責、資料流、啟動方式。**現況與下一步看 [roadmap.md](roadmap.md)**，規範看 [../AGENTS.md](../AGENTS.md)。
> Last Updated: 2026-08-08

## 這個專案在解什麼

LinkedIn 的 **Zip** 解謎遊戲：在格盤上畫**一條連續路徑**，經過每個可走格子恰好一次，
並依序串起所有編號格（1 → 2 → 3 …），且不能穿牆。
本專案做的是「用各種演算法解它、程序化出題、並包成可互動的網頁服務」。

數學上這是**帶順序約束的 Hamiltonian path 問題**——所以既有精確解（DFS／A\*／CP-SAT），也有啟發式解（SA／GA／Tabu／PSO／ACO），
兩類的取捨正是本專案想比較的東西。

## 核心設計理念：三層解耦

```
UI 層 ──→ API 層 ──→ Core 層
(展示)     (協定)     (演算法，不知道有 Web 這回事)
```

1. **Core (`src/core/`)**：純演算法與資料結構，**不依賴 FastAPI／Gradio**。可以單獨在 REPL 裡跑。
2. **App (`src/app/`)**：FastAPI，只負責收 request → 呼叫 core → 包成 response。routers／schemas 分離。
3. **UI (`src/ui/`、`src/custom_components/`)**：Gradio 與 Svelte，扮演 **Adapter**——把使用者的直覺操作翻譯成 API 要的精確格式。

> 這層分工是刻意的：改演算法不該碰到 UI，改 UI 不該碰到演算法。新增 solver 只要動 `src/core/solvers/` ＋ API 的 `SOLVERS` 表。

## 專案結構

```text
linkedin-zip-challenge/
├── AGENTS.md                    # 操作規範正本（給 AI agent 與本人）
├── CLAUDE.md                    # 一行 @AGENTS.md
├── ai-collab/                   # AI 協作文件
│   ├── roadmap.md               # ★ 現況與下一步（新 session 第一站）
│   ├── project_guide.md         # 本檔：架構與啟動方式
│   ├── dev_log.md               # 開發日誌（逆時序，最新在上）
│   ├── commands.txt             # 常用咒語
│   └── reports/                 # 任務報告 YYYY-MM-DD_<主題>.md
├── src/
│   ├── app/                     # FastAPI 後端
│   │   ├── main.py              # app 定義、CORS、掛載 Svelte 靜態檔與 Gradio
│   │   ├── routers/             # echo.py、solver.py（端點）
│   │   ├── schemas/             # echo.py、solver.py（Pydantic 請求/回應）
│   │   └── tests/               # test_solver_api.py（TestClient）
│   ├── core/                    # 核心邏輯（不依賴 Web 框架）
│   │   ├── utils.py             # ★ 共用中樞：Puzzle 型別、parser、fitness、視覺化
│   │   ├── solvers/             # 9 種解題演算法
│   │   ├── puzzle_generation/   # 程序化出題與資料集腳本
│   │   ├── rl/                  # ⏸ RL 實驗（暫停，見 roadmap 決策表）
│   │   ├── vl_models/           # 🧪 圖片解析實驗（scratchpad）
│   │   └── tests/               # conftest.py（6 題 ground truth）＋ solvers/ 測試
│   ├── custom_components/
│   │   └── puzzle_editor/frontend/   # Svelte + Vite（Canvas WYSIWYG 編輯器）
│   ├── ui/
│   │   ├── gradio_app.py        # Gradio 多分頁介面
│   │   └── tests/               # test_gradio_app.py（unittest.mock）
│   └── settings.py              # pydantic-settings 集中設定
├── .devcontainer/
│   ├── Dockerfile               # PRODUCTION：多階段（node build → python runtime）
│   └── Dockerfile.dev           # DEVELOPMENT
├── docker-compose.yml           # production（單一自足服務）
├── docker-compose.dev.yml       # development（後端 ＋ Svelte dev server 兩容器）
├── run_docker_dev.py            # 一鍵啟動開發環境
├── .env / .env.example          # 環境變數（.env 絕不進版控）
├── .python-version              # 3.11
├── pyproject.toml / uv.lock     # 相依（獨立於 repo 根，不進 uv workspace）
└── pytest.ini                   # pythonpath = .
```

## 關鍵模組

### `src/core/utils.py` — 共用中樞（動它會牽動全部 solver）

| 函式／型別 | 職責 |
|---|---|
| `Puzzle`（TypedDict） | 謎題的**資料契約**：所有 solver 收到的都是這個型別 |
| `SolverInput`（NamedTuple） | solver 共用的輸入束 |
| `parse_puzzle_layout()` | 文字版面 → `Puzzle`。**唯一的 parser**，不要在別處重寫 |
| `prepare_solver_input()` | 抽出 solver 共用的參數擷取與驗證（DRY 重構的成果） |
| `calculate_fitness_score()` | 啟發式 solver 的評分核心，含**非連續路徑（跳格）懲罰** |
| `generate_random_path()` / `generate_neighbor_path()` | 啟發式 solver 共用的路徑產生與鄰域擾動 |
| `save_animation_as_gif()` / `save_detailed_animation_as_gif()` / `save_solution_as_image()` | 視覺化輸出 |

### `src/core/solvers/` — 9 種演算法

| 類型 | 檔案 | 特性 |
|---|---|---|
| 精確解 | `dfs.py` | 回溯 DFS，基準實作 |
| 精確解 | `a_star.py` | A\*（Manhattan 啟發式）；含 `heapq` 與 `SortedList` 兩個變體 |
| 精確解 | `cp.py` | OR-Tools CP-SAT，用 **dummy node ＋ `AddCircuit`** 表達 Hamiltonian path |
| 啟發式 | `monte_carlo.py` | 隨機取樣 baseline |
| 啟發式 | `simulated_annealing.py` | 鄰域用 **truncate-and-regrow**（不是 2-opt，2-opt 在格盤上會產生跳格） |
| 啟發式 | `genetic_algorithm.py` | 刻意 **no-crossover**（傳統交叉會破壞路徑連續性），靠菁英保留＋突變 |
| 啟發式 | `tabu_search.py` | `deque(maxlen=…)` 當短期記憶，存路徑 hash 省記憶體；含 aspiration 準則 |
| 啟發式 | `particle_swarm_optimization.py` | 離散化 PSO：位置＝路徑、速度＝交換操作序列 |
| 啟發式 | `ant_colony_optimization.py` | 蟻群 |

> ⚠ **API 目前只暴露 3 種**（`src/app/routers/solver.py` 的 `SOLVERS`：DFS／A\*(heapq)／CP-SAT）。補齊是 roadmap 下一步 #2。

### `src/app/` — API 契約

- `POST /api/solver/solve` — 收 `SolverRequest`（`puzzle_layout_str`、`walls_str`、`solver_name`），回 `SolverResponse`（`solution_path` ＋ Base64 GIF ＋ Base64 PNG）
- `POST /api/echo` — 連通性測試
- `GET /docs` — Swagger

### `src/settings.py` — 集中設定（`pydantic-settings`，讀 `.env`）

| 設定 | 預設 |
|---|---|
| `app_port` | `7440` |
| `app_host` | `127.0.0.1` |
| `svelte_port` | `5173` |
| `ollama_model_name` / `ollama_provider_url` | `""`（VL 實驗用） |

`model_config` 設了 `extra="ignore"`——`.env` 多幾個變數不會炸。**不要在別處硬寫埠號**，一律從這裡讀。

## 啟動方式

### 方式 1：本機（開發演算法時最快）

```powershell
cd linkedin-zip-challenge
uv sync
uv run pytest                      # 先確認基線
uv run python -m src.app.main      # http://localhost:7440/ui
```

Svelte UI 需先建置一次（沒建置只是 `/svelte-ui` 不可用，不影響其他功能）：

```powershell
cd src/custom_components/puzzle_editor/frontend
npm install
npm run build
```

### 方式 2：Docker 開發環境（含 hot-reload）

```powershell
python run_docker_dev.py
```

兩個容器（後端 ＋ Svelte dev server）＋ volume mount。
- 統一入口：`http://localhost:7440/ui`、`http://localhost:7440/svelte-ui`
- 前端 hot-reload 要直接開 `http://localhost:5173`

### 方式 3：Docker 生產模擬

```powershell
docker compose -f docker-compose.yml up --build -d
```

多階段建置（`node:lts-alpine` 建前端 → Python runtime），單一自足映像檔，全部從 `7440` 出。

## 測試

```powershell
cd linkedin-zip-challenge
uv run pytest                      # 全部
uv run pytest src/core/tests -v    # 只跑核心演算法
.\run_tests.bat                    # 跑完把報告寫進 src/core/tests/reports/
```

- **Ground truth 在 `src/core/tests/conftest.py`**：6 題（`puzzle_01`～`puzzle_06`）連同人工驗證過的解答，用 `pytest.parametrize` 餵給精確解 solver。
- 精確解 solver **比對完整路徑**；啟發式 solver 只做 smoke test（驗格式與路徑合法性，不驗最佳性）。
- ⚠ 曾有過教訓：測試失敗不一定是 solver 錯——2025-09-23 查了半天才發現是 `conftest.py` 裡的參考答案打錯。**懷疑演算法前先驗 ground truth**。
