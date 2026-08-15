# AGENTS.md — linkedin-zip-challenge 操作指南（子專案正本）

> **本檔是這個子專案的操作規範正本。** `CLAUDE.md` 只用 `@AGENTS.md` 載入本檔。
> repo 級規範（monorepo 地圖、venv 分工、紅線）在 [`../AGENTS.md`](../AGENTS.md)；Python 程式碼風格在 [`../rules.md`](../rules.md)。
> **衝突時以本檔為準**（較具體者優先）。
> Last Updated: 2026-08-08

---

## 0. 這個專案是什麼

**LinkedIn Zip 解謎挑戰**：用多種演算法解 Zip 遊戲、程序化出題、並包成可互動的網頁服務。
本質是**帶順序約束的 Hamiltonian path 問題**——精確解與啟發式解的取捨正是專案的主題。

含：9 種 solver、FastAPI 後端、Gradio 主控台、Svelte Canvas 編輯器、出題器、GIF／PNG 視覺化、Docker 雙環境。
另有兩塊**未進主流程**的實驗：`src/core/rl/`（⏸ 暫停）與 `src/core/vl_models/`（🧪 scratchpad）。

## 1. 每次上線的標準動作

1. 看 SessionStart 簡報（分支／最近 commit／工作區狀態／下一步）。
2. 讀 [`ai-collab/roadmap.md`](ai-collab/roadmap.md)——**現況、下一步、已定案不要再重開的決策**。
3. 要動程式再讀 [`ai-collab/project_guide.md`](ai-collab/project_guide.md)（架構、模組職責、關鍵函式）。
4. 需要歷史脈絡（某個設計為什麼長這樣）才翻 [`ai-collab/dev_log.md`](ai-collab/dev_log.md)——它是 600+ 行的完整檔案，不要整份讀，用關鍵字搜。
5. 開工前給計畫（見 §3），確認後才動手。

> ⚠ **本專案自 2025-10-30 起休眠約 9 個月**。任何開發前**先跑 `uv sync` ＋ `uv run pytest` 建立基線**，
> 不要假設環境還是好的，也不要把「環境漂移造成的失敗」誤判成程式壞掉。

## 2. 文件分工（改東西前先確認要動哪一份）

| 檔案 | 角色 |
|------|------|
| `AGENTS.md`（本檔） | 操作規範正本：流程、環境、驗證、任務地圖、紅線 |
| `CLAUDE.md` | 只有一行 `@AGENTS.md` |
| `ai-collab/roadmap.md` | **現況與下一步**。新 session 第一站；每次做完事要更新 |
| `ai-collab/project_guide.md` | 架構、模組職責、資料流、啟動方式 |
| `ai-collab/dev_log.md` | 開發日誌（逆時序，最新在上）。做完一段就加一則 |
| `ai-collab/commands.txt` | 常用咒語 |
| `ai-collab/plans/` | **任務計畫書** `YYYY-MM-DD_track-<名稱>.md`：給接手某條 track 的 agent 用，含 worktree 環境建置、分階段 done 條件、track 間的協作約定 |
| `ai-collab/reports/` | 任務報告 `YYYY-MM-DD_<主題>.md`（較大的任務才出；分析與推理放這裡，計畫書只放要做什麼） |
| `README.md` / `README_zh-TW.md` | **對外門面**（英文／中文）。功能、遊戲規則、安裝與啟動 |
| `gemini_readme_raw.md` | 歷史檔案：2025 年的原始協作提示語，**已被本檔取代**，保留當考古用 |
| `.env.example` | 環境變數樣板（`.env` 本身**絕不進版控**） |

私人脈絡（工作節奏、失敗實驗、暫存腳本）寫 `../hi-collab/`，不要寫進本目錄。

## 3. 交辦任務的執行流程（標準五步）★

1. **先獲取相關知識**：動手前先讀 `roadmap.md`／`project_guide.md`／相關程式碼，不憑記憶開工。
2. **給計畫（Plan）**：拆步驟、每步怎麼算完成、風險；需確認的先確認再往下。有多種解讀就列出來。
3. **需要時派 subagent 並行**：多檔搜尋、跨模組查證等獨立重活才分派；緊耦合的事自己做。
4. **自己驗證、自己定義 done**：**明列 done 條件逐項確認**再回報。測試沒跑就說沒跑。
5. **做完就落地更新文件**：`ai-collab/roadmap.md`（現況＋下一步）＋ `ai-collab/dev_log.md`（加一則）；
   較大任務出 `ai-collab/reports/`；私人細節 → `../hi-collab/worklog/`。**不只留在對話**。

## 4. 環境與驗證（硬性要求）

**環境**（詳見 `../AGENTS.md §5`）
- Python **3.11**（`.python-version` 已指定，`requires-python = ">=3.11,<3.12"`）
- **一律 `cd linkedin-zip-challenge` 之後才 `uv run`**；不要用 repo 根的 `.venv`（那是 py3.9 devtools）
- **本專案刻意不進 root uv workspace**（鎖 `torch==2.4.1` ＋ cu121 index，與根的 3.9 衝突）——不要為了統一把它加進去
- 新增套件**通知本人手動 `uv add`**，Agent 不自己裝

**驗證指令**

```powershell
cd linkedin-zip-challenge
uv sync                            # 建立/同步環境
uv run pytest                      # 全部測試
uv run pytest src/core/tests -v    # 只跑核心演算法
uv run ruff check .                # 快速 lint
uv run python -m src.app.main      # 起服務：http://localhost:7440/ui
python run_docker_dev.py           # Docker 開發環境（hot-reload）
```

- **沒跑就說沒跑**：不要推測輸出、不要拿記憶中的數字當實測結果。回報要貼**實際輸出關鍵行**。
- 測試失敗**不要順手改測試讓它變綠**——先判斷是環境漂移、ground truth 錯、還是真的壞了。
- 動到 solver 一定要跑 `src/core/tests/solvers/` 的對應測試；動到 API 要跑 `src/app/tests/`；動到 Gradio 要跑 `src/ui/tests/`。

**實作後三件事**
1. **驗證底線**：測試全過、既有功能沒被破壞、任務定義的功能有實際驗過。
2. **清理**：移除 debug log、沒用到的變數與 import、臨時註解；實驗探針放 `../hi-collab/scratch/`。
3. **反思**：這個解法優雅嗎？特殊案例能不能消掉？

## 5. 常見任務地圖

| 任務 | 動哪裡 |
|------|--------|
| 新增一種 solver | `src/core/solvers/<name>.py` ＋ `src/core/tests/solvers/test_<name>.py`；要上 API 再加進 `src/app/routers/solver.py` 的 `SOLVERS` |
| 把既有 solver 掛上 API | `src/app/routers/solver.py` 的 `SOLVERS` dict ＋ `src/app/schemas/solver.py`（啟發式需要 `attempts` 參數） |
| 改謎題資料格式／parser | `src/core/utils.py` 的 `Puzzle` 與 `parse_puzzle_layout()`——**唯一 parser，不要在別處重寫**；改了要全體 solver 回歸 |
| 改評分邏輯 | `src/core/utils.py` 的 `calculate_fitness_score()`；牽動所有啟發式 solver |
| 改視覺化（GIF／PNG） | `src/core/utils.py` 的 `save_*` 系列 |
| 改 API 端點或回應格式 | `src/app/routers/` ＋ `src/app/schemas/`（成對改）；跑 `src/app/tests/` |
| 改 Gradio 介面 | `src/ui/gradio_app.py`；它是 **Adapter**，負責把 UI 操作翻成 API 格式，邏輯不要塞進來 |
| 改 Svelte 編輯器 | `src/custom_components/puzzle_editor/frontend/`（`Index.svelte`／`vite.config.ts`）；改完要 `npm run build` 才會反映到 `/svelte-ui` |
| 改埠號／設定 | `src/settings.py`（唯一來源）＋ `.env.example`；**不要在程式裡硬寫** |
| 改出題器 | `src/core/puzzle_generation/puzzle_generator.py`（回溯 ＋ retry/decrement ＋ 內部逾時） |
| 碰 RL | `src/core/rl/`——**先讀 `roadmap.md` 的決策表**，不要重蹈「調 reward 權重硬解 policy loop」 |
| 碰 VL 圖片解析 | `src/core/vl_models/`——目前是 scratchpad；混合策略（`output_type=str` ＋ prompt engineering）是已驗證路線 |

## 6. 程式慣例（承 `../rules.md`，此處只列本專案特有）

- `.py` 開頭加路徑註解，例如 `# src/core/solvers/dfs.py`（**用正斜線**，跨平台一致）
- 絕對匯入：`from src.core.utils import ...`，不用相對匯入
- 路徑操作一律 `pathlib`，不用 `os.path`
- 現代 type hint：`|` 取代 `Optional`、小寫 `list`/`dict`
- **日誌用 `loguru`**：`from loguru import logger`；生產路徑禁止 `print()`
  （例外：`src/app/main.py` 啟動階段的掛載訊息目前用 `print`，屬既有行為，不順手改）
- 消除魔術數字，模組頂端定義具名常數
- 例外處理用 `logger.exception()` 而非 `logger.error(f"...")`——才會留下完整 stack trace
- **牆一律叫 `walls`**，不要用 `blocked_cells` 指涉牆（那是障礙格，不同概念）
- lint／格式化只用 `ruff`（`black`／`isort` 已刻意移除，不要加回來）

## 7. 紅線

- **git 需當次授權**：commit／push／PR／merge 都要本次明確授權；單獨要求 commit **不含** push。不在 `main` 直接開發，**禁 force push**。
- **`.env` 絕不 commit**（含 `ollama_provider_url` 等本機設定）。要改樣板改 `.env.example`。
- **不永久刪除任何檔案**：不用 `rm`／`git rm`／`Remove-Item`／`shutil.rmtree()`。
  要移除的移進 `../soft-delete/<時間戳>/<原相對路徑>` 並回報怎麼還原；檔案搬家用 `git mv`。
- **不動大型產物**：`models/`、`logs/`、`datasets/`、`puzzle_dataset/`、`zip_puzzles/` 都不進版控，也不要擅自清除。
- **這是公開 repo**：進版控的內容不得出現真實姓名、公司名、公司專案路徑、個人聯絡方式。
- **重跑長時間訓練或刪資料集先確認**（RL 訓練與資料集生成都是小時級）。

## 8. 回報格式

完成任務時要包含：**改了哪些檔**、實際跑過的指令與**輸出關鍵行**、有沒有更新 `roadmap.md` 與 `dev_log.md`、**逐項確認的 done 條件**。
沒跑就說沒跑；有跳過的部分明講原因與未驗範圍。
