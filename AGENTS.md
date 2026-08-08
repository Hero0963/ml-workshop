# AGENTS.md — ml-workshop 操作指南（repo 級正本）

> **本檔是 repo 級的唯一正本**：monorepo 地圖、venv 規則、工作流程、紅線都在這裡。
> `CLAUDE.md` 只用 `@AGENTS.md` 載入本檔，不另外維護內容。
> **子專案有自己的正本**：進某個子專案工作，以該子專案的 `AGENTS.md` 為準（見 §8）。
> 適用對象：AI coding agent（Claude Code／Codex／Gemini CLI…）與本人。
> Last Updated: 2026-08-08

---

## 0. 這個 repo 是什麼

**ml-workshop ＝ 機器學習實作練功房**，一個 monorepo，底下每個子專案各自獨立（各自的 `pyproject.toml`、`uv.lock`、`.venv`）。
公開 repo：<https://github.com/Hero0963/ml-workshop>

| 子專案 | 一句話 | 協作文件 |
|--------|--------|----------|
| `linkedin-zip-challenge/` | ★ **最大最活躍**。LinkedIn Zip 解謎：9 種 solver ＋ FastAPI ＋ Gradio ＋ Svelte 編輯器＋ RL／VL 實驗 | `AGENTS.md` ＋ `ai-collab/` |
| `board-game-rl/` | 井字遊戲 RL：Q-Learning／Alpha-Beta／DQN ＋ FastAPI ＋ Gradio，DDD 分層 | `ai-collab/` |
| `deep-learning-karpathy/` | Karpathy 教材重現：minBPE tokenizer ＋ nanoGPT | `ai-collab/` |
| `lingua-tutor/` | 語言學習助理：STT 轉錄與評分 | `dev_log.md` |
| `more_simple_reinforcement_learning/` | RL 演算法 notebook 教材（8 章：Q-Learning → SAC／TD3） | `readme.md` |
| `notes/` | ML 主題筆記（faiss、hnsw、jieba、TrueSkill…） | `README.md` |

repo 根的 `scripts/`、`main.py` 是零星工具，不屬於任何子專案。

## 1. 每次上線的標準動作

1. 看 SessionStart 印出的簡報（`.claude/session-brief.py`：分支／最近 commit／工作區狀態／各子專案下一步）。
2. **確認要動哪個子專案**，讀它的 `AGENTS.md`（若有）與 `ai-collab/roadmap.md`（現況、下一步、已定案不要再重開的決策）。
3. 動程式前讀 `ai-collab/project_guide.md`（架構與啟動方式）。
4. 開工前給計畫（見 §3），確認後才動手。

> **不要憑記憶開工**：架構、埠號、指令、目前進度一律以子專案文件為準，不要用上一輪對話或訓練資料裡的印象。

## 2. 文件分工（改東西前先確認要動哪一份）

| 檔案 | 角色 | 版控 |
|------|------|------|
| `AGENTS.md`（本檔） | **repo 級操作規範正本**。monorepo 地圖、venv 規則、流程、紅線 | ✅ 公開 |
| `CLAUDE.md` | 只有一行 `@AGENTS.md` ＋ Claude Code 備註 | ✅ 公開 |
| `rules.md` | **Python 程式碼與工具鏈規範**（type hint、pathlib、logging、pre-commit、commit 訊息） | ✅ 公開 |
| `<子專案>/AGENTS.md` | 該子專案的操作規範正本 | ✅ 公開 |
| `<子專案>/ai-collab/roadmap.md` | **現況與下一步**。新 session 的第一站 | ✅ 公開 |
| `<子專案>/ai-collab/project_guide.md` | 架構、模組職責、啟動方式 | ✅ 公開 |
| `<子專案>/ai-collab/dev_log.md` | 開發日誌（逆時序，最新在上） | ✅ 公開 |
| `<子專案>/ai-collab/commands.txt` | 常用咒語（貼給助理的提示語模板） | ✅ 公開 |
| `<子專案>/ai-collab/reports/` | 任務報告 `YYYY-MM-DD_<主題>.md` | ✅ 公開 |
| `hi-collab/` | **私人工作區**：工作日誌情境、失敗實驗、暫存腳本 | ❌ 本機 |
| `.claude/settings.json` | SessionStart hook ＋ 權限 allowlist（共用設定） | ✅ 公開 |
| `.claude/settings.local.json` | 個人覆寫 | ❌ 本機 |

**公開／私人的判準**：對陌生讀者或未來的自己有用的技術內容 → 進版控；只對本人有用的（工作節奏、個人脈絡、當下的心情與取捨、未成熟的想法、失敗實驗）→ `hi-collab/`。
**真實姓名、公司名、家目錄個資、帳號、金鑰兩邊都不寫**（要記就寫進助理 memory）。

## 3. 交辦任務的執行流程（標準五步）★

1. **先獲取相關知識**：動手前先讀相關檔案／查證，不憑記憶開工；不確定就先查、先讀。
2. **給計畫（Plan）**：拆步驟、每步怎麼算完成、風險；需確認的先確認再往下。
   有多種解讀就列出來，不要默默挑一個；**先對齊範圍再動工**。
3. **需要時派 subagent 並行**：可拆解又獨立的重活（多檔搜尋、跨模組查證、彼此不依賴的子任務）才分派；**緊耦合、需高一致性**的事自己做，不為並行而並行。
4. **自己驗證、自己定義 done**：產出後自檢，**明列 done 條件逐項確認**再回報。
5. **做完就落地更新文件**：`ai-collab/roadmap.md`（現況＋下一步）、`ai-collab/dev_log.md`（做了什麼）；私人細節 → `hi-collab/worklog/`；必要時本檔與助理 memory。**不只留在對話**。

## 4. 核心工作守則

**溝通**
- 預設**繁體中文**；相對日期一律換絕對日期（`YYYY-MM-DD`，Asia/Taipei）。
- **雙層講解**：先用高中生聽得懂的白話建立直觀，再補術語與正式定義；補實際例子。
- 給多個方案**直接給推薦**，不長篇羅列不採用的。
- 不確定的事**先明講「不確定」**再查證；上網查詢要說搜尋關鍵字、引用附真實連結、先篩可信度。
- 不捏造已讀／已測／已跑的結果；不知道就寫「待補」。

**對話判讀（容易誤判的訊號）**
- 「這樣寫夠清楚嗎／這樣規劃可以嗎」是**評估請求**，先回答評估，不要直接動工。
- 「commit」單獨出現時**不隱含 push**；授權一任務一次，不沿用上一個任務。
- 提出的風險被本人否決或說「先這樣」，就標注風險後照做，不反覆推銷。

**做事**
- **先 Plan、找根因、簡單優先**：最小可行先、只動必要的、找通用解不貼藥膏。
- **外科式改動**：只動必要的地方，不順手「改善」鄰近程式碼；風格跟隨既有程式碼。
- 「目前沒出事」不等於沒風險——看到潛在問題即使還沒爆也要提。
- 數值／數學宣稱能算的**先用程式實跑驗證**，不只採信轉述。

## 5. venv 與 uv 規則 ★（本 repo 最容易踩雷的地方）

**每個子專案一個獨立 `.venv`，不共用。**

| 位置 | Python | 用途 |
|------|--------|------|
| `ml-workshop/.venv` | 3.9 | 只裝 devtools（`pre-commit`、`pytest`），跑 repo 級 lint 用 |
| `linkedin-zip-challenge/.venv` | 3.11 | 該子專案（鎖 `torch==2.4.1` ＋ cu121 index） |
| `board-game-rl/.venv` | 3.13 | 該子專案 |
| `lingua-tutor/`、`more_simple_reinforcement_learning/`、`notes/` | 各自 | 各自 |

**三條硬規則**

1. **一律 `cd <子專案>` 之後才 `uv run <cmd>`。** 絕不在 repo 根的 `.venv`（py3.9 devtools）跑子專案程式——會 import 不到、或裝錯版本。
2. **`linkedin-zip-challenge` 刻意不進 root uv workspace。** 根 `pyproject.toml` 的 `[tool.uv.workspace] members` 只有 `deep-learning-karpathy`。原因：它鎖 Python 3.11 ＋ `torch==2.4.1`／`cu121` 自訂 index，與根的 3.9 衝突；**不要為了「統一」把它加進 workspace**。
3. **新增套件由本人手動執行 `uv add`**，Agent 只負責通知要裝什麼、為什麼。

**環境確定性**：子專案應有 `.python-version` 讓 `uv sync` 決定性挑版本。目前 `linkedin-zip-challenge`（3.11）與 `board-game-rl`（3.13）有；其餘待補。

## 6. 驗證與交付

- 測試一律 `cd <子專案> && uv run pytest`；**沒跑就說沒跑**，不要推測輸出、不要引用記憶中的數字當實測結果。
- repo 級 lint：repo 根 `uv run pre-commit run --all-files`（`ruff` 統一負責 lint／import 排序／格式化）。
- 快速診斷：`uv run ruff check .`。
- 提交前確認測試通過；有失敗就**如實回報**，不順手改測試讓它變綠。

**實作後三件事**
1. **驗證底線**：測試全過、既有功能沒被破壞、任務定義的功能有實際驗過。
2. **清理**：移除 debug log、沒用到的變數與 import、臨時註解、實驗探針（要留就放 `hi-collab/scratch/`）。
3. **反思**：這個解法優雅嗎？特殊案例能不能消掉？不優雅就重構。

## 7. 紅線

- **git 需當次授權**：commit／push／建立或更新 PR／merge 都要本次明確授權，上次的不沿用；單獨要求 commit **不含** push。
  **不在 `main` 直接開發**——開功能分支；**禁 force push**。
- **不永久刪除任何檔案**：不用 `rm`／`rmdir`／`git rm`／`git clean`／`Remove-Item`／`shutil.rmtree()`。
  要移除的移進 `soft-delete/<時間戳>/<原相對路徑>`（已在 `.gitignore`），保留原始相對路徑與可復原副本，並回報移到哪、怎麼還原。
  `soft-delete/` 內容不得清空或再刪；永久刪除只由本人親自執行。檔案**搬家**用 `git mv`（是移動不是刪除）。
- **這是公開 repo**：進版控的檔案、commit 訊息、issue 都不得出現真實姓名、公司名、公司專案路徑、家目錄個資、帳號或個人聯絡方式。上網查詢一律去識別化。
- **絕不 commit 機敏資料**：`.env`、金鑰、token。`.env.example` 才是進版控的那份。
- **大檔不進版控**：`models/`、`logs/`、`datasets/`、`puzzle_dataset/`、`node_modules/`、`*.pth` 都已在 `.gitignore`。
- **不可逆或對外的動作先確認**：發佈、對外投稿、刪資料集、重跑長時間訓練，先問再做。

## 8. 子專案地圖（規範正本在哪）

| 要動哪裡 | 先讀 |
|---------|------|
| Zip 解謎（solver／API／Gradio／Svelte／RL／VL） | `linkedin-zip-challenge/AGENTS.md` → `ai-collab/roadmap.md` |
| 井字遊戲 RL（Q-Learning／DQN／Alpha-Beta／UI） | `board-game-rl/ai-collab/rules.md` ＋ `project_guide.md` |
| Karpathy 教材（tokenizer／nanoGPT） | `deep-learning-karpathy/ai-collab/rules.md` ＋ `handover.md` |
| 語言學習助理 | `lingua-tutor/README.md` ＋ `dev_log.md` |
| RL notebook 教材 | `more_simple_reinforcement_learning/readme.md` |
| ML 主題筆記 | `notes/README.md` |
| Python 程式碼風格、commit 規範 | `rules.md`（repo 根） |

## 9. 回報格式

完成任務時要包含：**改了哪些檔**、實際跑過的指令與**輸出關鍵行**、有沒有更新對應文件（roadmap／dev_log）、**逐項確認的 done 條件**。
沒跑就說沒跑；有跳過的部分明講原因。
