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

### S3 修 issue（進行中）

**已確認並修掉**（commit 見 git log）：

| # | 問題 | 證據 | 修法 | 已驗 |
|---|---|---|---|---|
| A | `start.py` 在 Python 3.9 直接崩潰（macOS 內建 `python3` 是 3.9） | `TypeError: unsupported operand type(s) for \|` @ `start.py:123` | `from __future__ import annotations` | py3.9 `--help`／`--status` ✅、py3.11 ✅、ruff ✅ |
| B | 無 NVIDIA GPU 的機器，`start.py` 起 ollama 失敗就 `sys.exit`，app 根本不起 | Docker 錯誤 `could not select device driver "nvidia" with capabilities: [[gpu]]`（ragflow#9573） | `ensure_ollama()` 改非致命、回傳 bool；失敗就跳過等 ollama 的 60 秒 | 待故障注入實測 |
| C | Ollama 起來 ≠ 視覺模型在；陌生人只會在上傳時才看到 503 | `report_ollama` 只列模型不比對 | 比對 `.env` 的 `OLLAMA_MODEL_NAME`，缺就明說並指向 README「Model weights」 | 待實測 |
| D | torch 只有 `linux_x86_64`／`win_amd64` wheel ⇒ ARM 主機（Apple Silicon）原生建置 `uv sync` 失敗 | `uv.lock` 只列兩個 wheel | 兩份 app compose 加 `platform: linux/amd64`（x86 主機無差別） | compose config ✅；**ARM 無法實測** |
| E | Dockerfile `uv sync` 沒鎖 lockfile | uv 官方 Docker 指南建議 `uv sync --locked` | 兩份 Dockerfile 改 `--locked` | 待冷建置實測 |

| F | 浮動標籤：`node:lts-alpine`（Node 下一個 LTS 約 2026-10 接手）、`ollama/ollama:latest` | 實測版本 Node v24.21.0／npm 11.19.0、Ollama 0.32.13（Hub 上 `0.32.13` 與本機 `latest` 同日 2026-08-14） | 釘 `node:24-alpine`（兩份 Dockerfile）、`ollama/ollama:0.32.13`；Hub API 兩個標籤都 200 | 待重建實測 |

**冷建置實測（修 A–E 後，`--no-cache --pull`）**：88 秒，`uv sync --locked` 73.8 秒（容器內 uv 0.12.17，lock 沒被判過期）、
`npm install` 8.8 秒、`vite build` 成功（`dist/` 在），image 5.86 GB。npm 11 警告 4 個套件的 install script 未放行（esbuild 等），**目前無害**。
log：scratchpad `fresh/cold-build.log`。

**實測計畫**：本機 `git clone` 本分支到 scratchpad（ASCII 路徑）→ `docker compose build --no-cache --pull` →
停掉佔 7440 的 `zip-app-zip-infra`（`docker stop`，可 `docker start` 還原）→ `python start.py` →
API 各 solver、RL 503、放入 14 MB checkpoint 後 RL 200、視覺模型缺席訊息、無 GPU 故障注入。

**⚠ 實測撞到本機環境問題（不是 repo 的錯）**：`python start.py`（py3.9）建置與起容器都成功、容器內 health 一直 200，
但主機端 `curl 127.0.0.1:7440` 000、`netstat` 沒人聽 7440／11435。**對照組**（無關的 `python -m http.server` 容器 `-p 18081:8000`）也 000；
重啟 Docker Desktop 4.32 後仍 000；只綁 `127.0.0.1:` 也 000；app 容器內打 `host.docker.internal:11435` 被拒。
**根因**：`%USERPROFILE%\.wslconfig` 於 **2026-09-18 15:49** 改成 `networkingMode=mirrored`（晚於 09-12 的 Docker 驗收），
WSL mirrored 模式與 Docker Desktop 埠轉發衝突是已知問題（microsoft/WSL#10494、#41284）。
**處置**：不改本人的全域設定；改做容器內驗收（見下）；需要主機埠的項目（`start.py` 輪詢、Svelte 瀏覽器 e2e）最後問本人要不要暫切 NAT。
順帶：`zip-app-zip-infra` 已 `docker stop`（還原：`docker start zip-app-zip-infra-zip-challenge-app-1`）；Docker Desktop 重啟過一次。
**README 要加一條 troubleshooting**：Windows ＋ WSL mirrored networking 會讓 published port 失效。

另發現文件錯字：`vlm-operating-guide.md` L221 路徑 `datasetsl\main_6x6` 少了 `\v`。

**容器內驗收結果（fresh clone 的 image，2026-09-19 06:30–06:45）**——腳本與原始 JSON 在
`ai-collab/reports/artifacts/wrap-up-acceptance/`：
- `acceptance.py`（只走 HTTP，答案用 `verify.is_solution` 裁判）：5 個頁面 200；`/api/solver/list` 9 種、無 PSO。
  **無 RL 權重**：3 種精確解 5／5 盤全解（含 7×7）；RL 在 4／5／6×6 回 **503**（訊息指出缺哪個 checkpoint）、7×7 回 **400**；
  啟發式 7×7 有 ACO、Monte Carlo 放棄（5 秒用完，回 200＋「could not find」）、其餘全解。
  **放入 14 MB checkpoint（sha256 `653efaa5…` 與 `zip-rl` 原檔相同）後不重啟**：RL 解出生成的 4／5／6×6；
  真實題 `puzzle_01`（10 道牆，訓練分布外）這次放棄——與部署指南「同題 20 次解 10 次」一致。
- `vision_check.py`（版面逐格、牆集合、路徑能否解**標準答案那張盤**）：**全新合成 6 張（seed 919000000，牆 2–12、亮／暗）6/6 全對**、
  **held-out 4 張 4/4 全對**；第一張 58.2 秒（載模型），之後 3.2–7.8 秒／張。
- 真實截圖 6 張：輸出與 `vlm-operating-guide.md` §3.5 的預期表**逐列相同**（含刻意保留的 `puzzle_03` False）。
- 模型缺席（`OLLAMA_MODEL_NAME=zip-model-not-imported:f16`）：**503**，detail 寫 `model ... not found`。
- 讀圖測試走的是**同 image 另起的探針容器、直連 `zip_ollama_server:11434`**（主機埠在本機壞了）；GPU 號誌用完已改回 `free`。
- 容器內 `uv run pytest`：**331 passed, 8 xfailed**（Python 3.11.16、torch 2.4.1+cu121、cuda_available False）。
- `start.py` 新路徑：假 Ollama（主機 loopback）測 `report_ollama` 四種 model 值、假 compose 測 `ensure_ollama` 失敗／成功 ⇒ **ALL OK**（py3.9，腳本 scratchpad `fresh/test_start_report.py`）。
  **真實故障注入**（scratch clone 把 driver 改成不存在的名字）：印出 `could not select device driver ... [[gpu]]` 與新訊息、**沒有中止**、app 照常 up。
- 開發版：`docker compose -f docker-compose.dev.yml up -d --build` 9 秒（與正式版共用 `uv sync --locked` 層）、app healthy、`uvicorn --reload` 是 PID 1、
  vite 5.4.20 在 `0.0.0.0:5173`，容器內 `/svelte-ui/` 200、node v24.21.0（與先前 `lts` 同 digest）。
- **待本人決定**：主機端輪詢（`start.py` 的 health 等待）與 Svelte 瀏覽器 e2e 需要主機埠，本機 mirrored 模式下測不到。

**⚠ 我造成的副作用，已復原**：重啟 Docker Desktop 讓 4 個別專案容器（`ai_translator_app`、`pdf2zh_server`、`omni_parser`、`speech_motion_aligner`，
全是 `restart=always`，原本 Exited）被守護程序拉起來，`omni_parser` 吃約 50% CPU、`ai_translator_app` 重啟迴圈（exit 127）。
06:50 已 `docker stop` 四個，回到原本的停止狀態。**教訓：重啟 Docker 前先列出 `restart=always` 的容器。** 要回報本人。

S3 其餘待辦：README 加 WSL mirrored troubleshooting、修 `vlm-operating-guide.md` L221 錯字（併入 S5）。

### S4 Survey 筆記（進行中）

**GPT-6 computer use（2026-09-19 查證）**
- 一手：OpenAI〈GPT-6 Astra〉https://openai.com/index/gpt-6-astra/ （WebFetch 403，改用 Chrome headless dump；zh-TW 版頁面）。
  2026-09-03 發布。電腦操作：OSWorld 2.0（v2026.08.08 offline, partial）**72.6%**（GPT-5.6 Sol 65.7%、Claude Opus 5 70.2%）、
  ScreenSpot-Pro（no tools）92.7%、Agents' Last Exam 59.3%；OSWorld 延遲模擬每題約 40 分鐘。示範：KiCad 佈 PCB、Excel、遊戲開發、1040 表單、前端 QA。
  Codex harness 更新，Mind2Web 1.9× 快。API $10／$50 per M tokens。
- **官方頁沒有「小畫家」「2048」**。小畫家畫肖像是**社群示範**（happycapy.ai、magiccreator.ai 整理）；
  「2048」搜到的是**用 Astra 做出來的 2048 類遊戲 CityMaker**，不是它去玩 2048 ⇒ 本人說的「會玩 2048」**查無一手出處**，報告要照實寫。
  另有 Tom's Hardware：Astra 自主玩通 Portal（24 小時、token 成本 $571）——待讀原文確認。
- 搜尋關鍵字：`GPT-6 computer use OpenAI`、`OpenAI GPT-6 release announcement 2026`、`GPT-6 Astra Microsoft Paint drawing 2048 game computer use demo`、`"Astra" OpenAI "2048" game computer use`
