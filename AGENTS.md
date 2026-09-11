# AGENTS.md — ml-workshop 操作指南（repo 級正本）

> **本檔是 repo 級的唯一正本**：monorepo 地圖、venv 規則、工作流程、紅線都在這裡。
> `CLAUDE.md` 只用 `@AGENTS.md` 載入本檔，不另外維護內容。
> **子專案有自己的正本**：進某個子專案工作，以該子專案的 `AGENTS.md` 為準（見 §8）。
> 適用對象：AI coding agent（Claude Code／Codex／Gemini CLI…）與本人。
> Last Updated: 2026-08-29

---

## 0. 這個 repo 是什麼

**ml-workshop ＝ 機器學習實作練功房**，一個 monorepo，底下每個子專案各自獨立（各自的 `pyproject.toml`、`uv.lock`、`.venv`）。
公開 repo：<https://github.com/Hero0963/ml-workshop>

| 子專案 | 一句話 | 協作文件 |
|--------|--------|----------|
| `linkedin-zip-challenge/` | ★ **最大最活躍**。LinkedIn Zip 解謎：9 種 solver ＋ FastAPI ＋ Gradio ＋ Svelte 編輯器 ＋ **微調 VLM 讀圖解題**（已上線）＋ RL 實驗（⏸ 暫停） | `AGENTS.md` ＋ `ai-collab/` |
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
| `<子專案>/ai-collab/plans/` | **任務計畫書**（給接手某條 track 的 agent：worktree 環境建置、分階段 done 條件、協作約定） | ✅ 公開 |
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

**技術 survey 的保鮮紀律 ★**（2026-08-15 因實際犯錯新增）
- **列表型文章只能用來「列舉有哪些家族」，不能用來決定「該家族最新是第幾版」。**
  搜尋引擎大量回傳標題掛當年年份、內容卻是前一年的 SEO 文；照抄它的名單就會寫出過期報告。
- **逐家族向一手來源確認**：官方 repo／HF org 頁面（依日期排序）／vendor 部落格／套件 registry。
  每個家族補一次「有沒有更新版」的**反向探測**（搜 `<家族> N+1`），一次查詢遠比報告出錯便宜。
- **工具鏈索引是保鮮度神諭**：Unsloth docs／notebook 清單、Ollama library、llama.cpp 支援列表反映「現在真的能訓、能跑什麼」。
  它與二手名單打架時**以工具鏈為準**。
- **驗證要對稱**：不能 A 家族開模型卡查證、B 家族靠部落格一句話帶過。同一份 survey 兩套標準＝必出錯。
- **版本號是知識庫裡最先過期的資訊**；距今半年內的版本宣稱一律重查，不吃記憶（訓練資料與過期文章會互相「確認」，形成確認偏誤）。
- 報告中標註**查證日期**與**一手／二手**，讓讀者知道哪些段落會腐化。

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

---

## 10. 多 agent 平行開發 ★（2026-09-12 新增）

> 這一節有兩個讀者：**接手的 agent**（§10.1–10.4、10.6）與**本人**（§10.5 的操作卡）。

### 10.1 先選對工具——一句話決策表

| 你要做的事 | 用什麼 | 為什麼 |
|---|---|---|
| **兩個以上的功能要同時開發（會改檔）** | **git worktree ＋ 每個 worktree 一個 session** | 唯一能做到檔案系統層級隔離的做法 |
| 同一條 track 裡要查很多檔、做互不相依的**唯讀**調查 | **subagent** | 省你的 context，而且不會改到檔 |
| 只是要看程式碼／問問題，不改東西 | 隨便哪個 session 都行 | 沒有衝突風險 |
| **同一個 worktree 開兩個 session 一起改檔** | **❌ 不要做** | 見 §10.2 |

**預設規則：一條 track ＝ 一個 worktree ＝ 一個 branch ＝ 一個 session。**

### 10.2 三種方式的實際差別

| 方式 | 隔離程度 | 適合 | **不適合** |
|---|---|---|---|
| **worktree ＋ 各自 session** | 完全隔離：各自的工作目錄、分支、`.venv`、測試產物 | **會改檔的平行開發** | 要各自 `uv sync` ＋ 複製 `.env`；吃磁碟（每個 `.venv` 好幾 GB）|
| 同 worktree 多 session | **幾乎沒有** | 唯讀查詢 | **會改檔的工作** |
| subagent（同 session 內） | 共用同一個工作目錄 | **唯讀 fan-out**：多檔搜尋、跨模組查證 | 平行改檔；不要為了並行而並行 |

**為什麼「同 worktree 多 session」會壞**，三件事同時發生：

1. **互相覆蓋**：A 讀了檔、B 改了同一個檔、A 再寫回去 ⇒ B 的改動消失，而且沒有人會發現。
2. **git index 打架**：兩邊同時 `git add`／`commit`，`.git/index.lock` 會噴錯，或把對方的檔一起 commit 進去。
3. **測試互相污染**：`pytest` 產生的檔、訓練寫進 `logs/`／`models/` 的東西，兩邊分不清是誰的。

**subagent 不是拿來平行開發的**：它和你共用同一個工作目錄，所以第 1、2 點一樣會發生。
它的價值是「**替你讀東西、不佔你的 context**」。

### 10.3 本機資源是互斥的（最容易被忽略的約束）

一台機器一張 GPU、24 核，**訓練／資料生成／批次評估都會吃滿預算**（§4 的 75% 上限）。
兩個 agent 同時開跑不只變慢，**量出來的時間數字會全部作廢**。

**共用號誌檔**（worktree 們都是 `ml-workshop` 的兄弟目錄，路徑互通）：

```
ml-workshop/.agent-heavy-job     # 內容一行：free  或  busy <track> <YYYY-MM-DD HH:MM>
```

| 動作 | 規則 |
|---|---|
| 開跑重工作前 | 先讀它。是 `free` 才能跑，然後覆寫成 `busy <track> <時間>` |
| 跑完 | **覆寫回 `free`**（忘記的話別人會一直等） |
| 被佔用時 | **不要等**——先做不吃資源的部分（寫程式、寫測試、寫文件），資源工作往後排 |
| 不算重工作 | `pytest`、`ruff`、Docker build、讀檔寫檔——不用搶 |

⚠ 這是**君子協定不是互斥鎖**。它的價值在於「跑完的數字可信」，不在於防止競爭。

### 10.4 共用檔案協定（不遵守就會 merge 衝突）

| 檔案 | 規則 |
|---|---|
| `<子專案>/ai-collab/dev_log.md` | 每條 track **各加自己的 `###` 小節**，不要動別人的 |
| `<子專案>/ai-collab/roadmap.md` | **只改自己那一項**；衝突時兩邊都保留 |
| `pyproject.toml` / `uv.lock` | **序列化處理**：一次只有一條 track 動相依；新套件由本人手動 `uv add` |
| `AGENTS.md` / `rules.md` | 動之前先說一聲——所有 track 的共同地板 |
| `src/core/utils.py`、`src/core/puzzle_generation/` | **只讀不改**，要改先提出 |
| `src/core/solvers/registry.py` | 新增 solver 的唯一入口；兩條 track 同時加會衝突，先講好 |

**★ 分工要按「檔案所有權」切，不是按「功能聽起來像不像」切。**
開新 track 前先問一句：**它會動到的檔案，和進行中的 track 有沒有交集？** 有就先談。

---

### 10.5 【給本人】怎麼把任務分配下去——五步操作卡

**Step 1｜先把 `main` 弄成最新**（所有 track 都從它長出來）

```powershell
cd D:\it_project\github_sync\ml-workshop
git checkout main
git pull
```

**Step 2｜把工作切成「檔案不重疊」的幾份**

寫一份任務計畫書放 `<子專案>/ai-collab/plans/YYYY-MM-DD_<主題>.md`，每條 track 一節，
每節都要有：**目標／擁有哪些檔／不准動哪些檔／done 條件／要不要 GPU／已知陷阱**。
（範例：[`linkedin-zip-challenge/ai-collab/plans/2026-09-12_next-tracks.md`](linkedin-zip-challenge/ai-collab/plans/2026-09-12_next-tracks.md)）

⚠ **檢查交集**：把每條 track 會動的資料夾列出來，**有重疊就不要平行**，改成序列做。

**Step 3｜每條 track 開一個 worktree**

```powershell
cd D:\it_project\github_sync\ml-workshop
git worktree add ..\zip-<track> -b feat/<track>

# .env 不進版控，一定要手動複製，否則服務起不來
Copy-Item .\linkedin-zip-challenge\.env ..\zip-<track>\linkedin-zip-challenge\.env

cd ..\zip-<track>\linkedin-zip-challenge
uv sync
```

⚠ **一律從 `main` 長**，不要從別的 feature 分支長——會讓兩條 track 的歷史糾纏，合併時很痛。
⚠ 已經存在的 worktree 就直接用（例如 `zip-vlm`、`zip-rl`），不用重開。

**Step 4｜在每個 worktree 目錄各開一個 Claude Code session，貼 §10.6 的起手式**

**Step 5｜收工：一次合併一條**

```powershell
cd D:\it_project\github_sync\ml-workshop
git merge --ff-only feat/<track>      # 不能 ff 就代表 main 動過了，先 rebase 那條 track
git push origin main
```

合完一條再合下一條。**同時合多條時 `uv.lock` 與 `dev_log.md` 最容易衝突。**

**分配任務時的三個判準**

1. **能不能獨立驗證**：這條 track 自己跑 `pytest` 就能確認做完了嗎？不行就切得不對。
2. **要不要搶 GPU**：兩條都要 GPU ⇒ 不要同時開，或讓其中一條先做不吃資源的部分。
3. **會不會動到共用地板**（`AGENTS.md`／`rules.md`／`utils.py`／`registry.py`）：會的話只給一條 track。

---

### 10.6 【給 agent】新 session 的第一則訊息（照貼）

```
你負責 <track 名稱>，worktree 是 D:\it_project\github_sync\zip-<track>。
所有指令都在這個目錄跑，不要 cd 回 ml-workshop。

1. 讀 linkedin-zip-challenge/ai-collab/plans/<任務計畫書>.md 裡「<track 名稱>」那一節
2. 讀該 track 的 handover（若有）：linkedin-zip-challenge/ai-collab/handover-<track>.md
3. 讀 AGENTS.md §10（平行開發規則）
4. 建立基線：cd linkedin-zip-challenge && uv run pytest && uv run ruff check .
5. 回報：目標是什麼、你要動哪些檔、done 條件、風險
6. 先不要動手，等我確認

規則：
- 只動計畫書說你擁有的檔案，其他 track 的檔案不要碰
- 重工作（訓練／資料生成／批次評估）開跑前先看 ../ml-workshop/.agent-heavy-job，
  是 free 才能跑，跑前改成 busy <track> <時間>，跑完改回 free
- commit 需要我當次授權；單獨說 commit 不含 push
- 告一段落就更新 dev_log.md（加自己的 ###）與 roadmap.md（只改自己那一項）
```

### 10.7 收工與合併

- 每條 track **自己 commit 自己的分支**（commit 需當次授權，單獨說 commit **不含** push）。
- 合併前該 track 要自己跑過 `uv run pytest` ＋ repo 根 `uv run pre-commit run --all-files`。
- track 結束時更新自己的 `handover-<track>.md`，**下一個接手的人只讀那一份就要能開工**。
- **一次合併一條**進 `main`，合完再合下一條。
