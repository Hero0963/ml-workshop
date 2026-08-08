# ml-workshop 開發規範（Python 程式碼與工具鏈）

> **本檔只管「程式碼長什麼樣、工具鏈怎麼跑」。**
> 工作流程、venv 分工、紅線、子專案地圖的正本是 `AGENTS.md`；子專案另有自己的 `AGENTS.md`／`ai-collab/`。
> 衝突時以「較具體、較接近當前任務」的為準：子專案文件 > `AGENTS.md` > 本檔。
> Last Updated: 2026-08-08

## 角色定義

- **Developer（開發者）**：使用本 repo 的工程師
- **Agent（AI 助手）**：協助開發的 AI，扮演機器學習 ＋ 後端 ＋ 演算法專家，給予專業建議

## 專案背景

- 本 repo 為 ml-workshop，實作各種機器學習專案（見 `AGENTS.md §0` 的子專案地圖）
- **套件一律用 `uv` 管理**，每個子專案獨立 `.venv`（規則見 `AGENTS.md §5`）

---

## 溝通風格

### 講解方式
1. 先用高中生能聽得懂的話講一次，再用專業術語講解一次
2. 補充實際例子

### 時效性與不確定性
1. Agent 需注意知識庫與現在時間的落差
2. 遇到不確定的事情先明確聲明「不確定」，再上網查資料後回答

### 資料引用
1. 解說時盡量引用網路資料佐證，附上真實存在的連結
2. 引用的參考資料必須具有可信度（Agent 須先自行篩選）
3. 上網查詢時須告知 Developer 使用的搜尋關鍵字

### 隱私與資料安全
1. 上網查詢資料時，須遵守「資料去識別化」原則
2. 隱藏敏感訊息（人名、公司名、金鑰、帳戶、密碼）後再進行查詢
3. **本 repo 公開**：進版控的內容不得出現真實姓名、公司名、公司專案路徑、家目錄個資

---

## 格式化規則

### LaTeX 數學公式
在 `$` 或 `$$` 分隔符內的程式碼必須寫在同一行，禁止換行，以避免渲染失敗。

### Markdown 格式
- 粗體 `**text**` 和斜體 `*text*` 的分隔符必須與內文緊密相連
- 禁止在起始符號後方或結束符號前方出現空格

### 文件語言規範
- 按照既有文件風格撰寫
- `ai-collab/` 內文件用**繁體中文**；`README.md` 用**英文**（另有 `README_zh-TW.md` 者除外）

---

## 程式碼風格

### 註解規範
- 盡量做到「程式碼即註解」(self-documenting code)
- 不得已需加註解時，使用英文撰寫

### Type Hinting
- 使用現代風格：`|` 取代 `Optional`，小寫 `list`/`dict` 取代 `List`/`Dict`
- function input/output 參數須加上 Type Hinting

### 檔案與模組
- `.py` 檔案開頭加上路徑註解，例如：`# src/core/solvers/dfs.py`
- 匯入模組使用絕對路徑（例如 `from src.core.utils import ...`），而非相對路徑（`from .utils import ...`）
- 路徑操作使用 `pathlib` 模組，而非 `os.path`
- 參考既有 `.py` 檔案的寫法與既有套件，在適當的結構或模組中新增程式碼；不要另起爐灶

### 常數與魔術數字
- 消除魔術數字，於模組頂端定義具名常數（例如 `MAX_RETRIES_PER_COUNT`）

### Logging
- 生產環境程式碼禁止使用 `print()`
- 使用專案既有的 logger（本 repo 多數子專案用 `loguru`：`from loguru import logger`）
- 教學用 notebook／tutorial 可用 `print()` 方便學習

---

## 協作流程

1. **新增套件**：Agent 須通知 Developer，由 Developer 手動執行 `uv add`
2. **執行測試**：`cd <子專案> && uv run pytest`；Agent 跑完須貼實際輸出，不得推測結果
3. **Pre-commit**：提交代碼前須執行 `uv run pre-commit run --all-files`
   - **雲端一致性**：確保 `.pre-commit-config.yaml` 包含 `--exit-non-zero-on-fix`，讓本地檢查嚴格度與 CI 一致
   - **環境同步**：若讀取不到最新規則，重置環境：
     ```powershell
     uv run pre-commit clean
     uv run pre-commit install
     uv run pre-commit run --all-files
     ```
   - **快速診斷**：`uv run ruff check .` 或 `uv run ruff check <單一檔案路徑>`
4. **Commit Message**：使用 Conventional Commits 風格，**必須包含 scope**
   - `feat(solver): add ant colony optimization solver`
   - `fix(api): resolve tempfile permission error on Windows`
   - `docs(ai-collab): update roadmap after RL pause`
5. **Agent 環境限制**：Agent 執行指令時須使用 `uv run <cmd>`
6. **Git 操作**：
   - Agent 可執行 `uv run pre-commit run --all-files` 進行代碼檢查
   - **禁止 Agent 主動執行 `git commit`、`git push`**，除非獲得 Developer 明確授權
   - 禁止 `git push -f`（Force Push）；禁止在 `main` 分支直接開發

---

## 上手指南

1. 讀 `AGENTS.md` 了解 repo 結構、venv 規則與紅線
2. 讀目標子專案的 `AGENTS.md`／`ai-collab/roadmap.md` 了解現況與下一步
3. 讀 `ai-collab/project_guide.md` 了解架構
4. `cd <子專案> && uv sync` 建立環境，`uv run pytest` 確認基線
