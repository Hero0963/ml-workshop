# 收尾第二輪：UI 整併、Svelte 補齊、授權、改名、權重發佈（2026-09-19）

> 接手的 session 先讀這份。本人授權的範圍、已定案的決策、每一階段的狀態都在這裡。
> 第一輪收尾的總帳是 [`../reports/2026-09-19_project-wrap-up.md`](../reports/2026-09-19_project-wrap-up.md)。

## 已定案（本人 2026-09-19）

| 項目 | 決定 |
|---|---|
| WSL 網路 | 永久改回 NAT（`.wslconfig` 的 mirrored 註解掉，已完成、埠轉發已驗證）|
| 別專案的 `restart=always` 容器 | `docker update --restart unless-stopped`（已完成）；3 組測試容器本人已 `compose down`（已驗證）|
| Gradio | 拿掉 Echo 分頁（API 保留）；Generate 改成選 4×4／5×5／6×6、不要障礙格；三個解題分頁合成一頁兩種模式：上傳截圖／編輯器 |
| 牆清單 bug | Gradio 6.15.1 唯讀 Dataframe「欄數不變、列數增加」不重畫（最小重現；6.28.0 已修）。**換元件，不升級套件** |
| Svelte | 補齊出題與上傳截圖（新增出題 API）。**本人驗收後定案**：左側只剩「1. Start from」（一組大小 4×4／5×5／6×6，共用給 **Generate** 與 **Blank grid**，加讀截圖）與「2. Solve」；Rows／Cols 與 Reset 拿掉（Blank grid 就是重來；手動建 7×7／長方形空白盤因此不再支援，讀截圖讀到 7×7 仍可顯示）|
| 授權 | Apache-2.0，只放子專案；LinkedIn 截圖不在授權範圍；著作權人寫 GitHub 帳號 |
| CLAUDE.md／AGENTS.md | **維持現在的做法，不動** |
| 改名 | `linkedin-zip-challenge` → **`thread-the-grid`**，內部前綴 **`threadgrid`**；對外名稱（Ollama 標籤、HF repo、release、Docker 容器）跟著改 |
| 改名不動的 | VLM 訓練 prompt 裡的「Zip puzzle」（改了模型失準）、Ollama volume 名稱（Docker 不能改名）、歷史報告內文（只修連結）、本機 worktree／分支名稱、GGUF 檔內 metadata |
| README 對來源的說法 | 一句：「Inspired by the Zip puzzle games found online」，**不提 LinkedIn**；第三方截圖寫「屬於各自的權利人、不在授權範圍」，不點名。model card 同口徑 |
| 權重公開 | 先非公開（draft release、private HF repo）→ 下載驗證 → 本人看過 model card 才公開；HF 帶 LoRA adapter |

## 階段與狀態

| # | 階段 | done 條件 | 狀態 |
|---|---|---|---|
| 1 | Gradio 改版＋牆清單 | pytest 綠；瀏覽器實測兩種模式、出題帶入編輯器、牆可增刪 | ✅ 瀏覽器 19/19（`scratchpad/ui/e2e_gradio.py`）；**需要 Gradio 6.17.3**（見下）|
| 2 | 出題 API＋Svelte 補齊 | API 測試；`npm run build`；瀏覽器實測出題、上傳讀圖、帶入畫布、解題 | ✅ API 7 測試；瀏覽器 10/10；既有 9 solver e2e 9/9＋不可解 3/3 |
| 3 | 授權 | LICENSE、README 授權一節、`package.json` license 欄位一致 | ✅ 官方 Apache-2.0 全文（sha256 `cfc7749b…3d30`）；README 中英授權一節；排除 puzzle_01–06、cat、bird |
| 4 | 改名 | 無舊名殘留（允許清單除外）；`.gitignore` 仍排除 `models/` 等大檔；pytest、ruff、`start.py` 全過 | ✅ 資料夾已搬（259 個追蹤檔全到、11 個新檔）；新 `.venv` 344 passed；容器 `threadgrid-app-zip-vlm-*`、`threadgrid_ollama_server`；舊 `.gitignore` 規則刻意保留（主 checkout 還沒搬）；舊資料夾只剩 `.venv` |
| 5 | 截圖＋端到端 | 新 UI 截圖逐張目視；Svelte e2e 9 種 solver 過裁判 | ✅ 對 Docker（7440）：Gradio 20/20、Svelte 10/10、9 solver 9/9＋不可解 3/3；5 張截圖逐張看過（途中抓到並修掉「上傳模式露出編輯器」）；紀錄在 `reports/artifacts/wrap-up-round-2/` |
| 6 | 權重發佈 | draft／private 上傳 → 下載核對 SHA-256 → 匯入 → `vision_check.py` 過 → 本人同意後公開 | 🔶 RL：draft release `thread-the-grid-models-v1` 已建、下載 SHA 相符、匿名 404；VLM：暫存資料夾 `models/hf-release/threadgrid-qwen35-4b-p4c-gguf/` 已備妥，**待本人 `hf auth login`** |
| 7 | 文件 | README、model-weights、收尾報告、部署指南、roadmap、dev_log、memory | ✅（權重網址待公開後補）；第二輪報告 `reports/2026-09-19_wrap-up-round-2.md` |
| 8 | commit | 當次授權 | 本人授權「commit ＋ push，main 本機與 remote 都最新」；HF 放最後 |

## 途中定案：Gradio 6.15.1 → 6.17.3（本人 2026-09-19 授權並親自 `uv lock`）

- 6.15.1 的前端在「同一個事件切換分頁＋改區塊顯示」時卡死或無限迴圈（最小重現＋逐版二分：6.15.1／6.15.2／6.16.0 壞，**6.17.3 好**）。「帶入編輯器」正是這種事件。
- 6.18 起要求 `huggingface-hub>=1.x`，與 `transformers<5` 衝突 ⇒ 6.17.3 是可用的最高版。
- 連帶：fastapi 0.119.1→0.141.1、starlette 0.48.0→1.6.0、新增 annotated-doc 0.0.5。升級後全測 344 passed／8 xfailed。
- 牆清單表格的 bug 要到 6.26 才修（需 transformers 5）⇒ 維持「多選標籤」取代表格。

## 搬資料夾的步驟（改名最後一步）

1. 停掉：本機 uvicorn（7452）、`zip-app-zip-vlm` 與 `zip_ollama_server` 容器（它們 bind mount 了 `models/`）。
2. `.gitignore` 先加新規則（`thread-the-grid/...`），舊規則保留到確認沒有大檔進 git 為止。
3. 建 `zip-vlm/thread-the-grid/`，把舊資料夾的內容**逐項搬過去**；`.venv` 不搬（VS Code 的 Black LSP 正在用它的 python），在新資料夾 `uv sync` 重建。
4. `git status` 應只看到改名；確認 `models/`、`datasets/`、`logs/` 沒有被追蹤。
5. 新資料夾跑 pytest、ruff、`python start.py`（新容器名 `threadgrid_*`）；舊的 `zip_*` 容器只停不刪，交給本人。
6. 舊資料夾只剩 `.venv`，由本人事後移除。
7. **主 checkout `ml-workshop` 合併後也要搬**：`models/`、`datasets/`、`logs/`、`.env` 是 ignored 檔，git 不會幫忙搬。

## 陷阱（做之前先讀）

- **Svelte 畫布**：換盤面大小時要「DOM 更新後再畫」（`tick().then(draw_puzzle)`），否則改 canvas 尺寸會清空剛畫好的圖（本人驗收抓到：Generate 後要再點一下才出現）。
  而 `draw_puzzle` 裡**不能對元件變數 `ctx` 做成員賦值**（`ctx.fillStyle = …` 會被 Svelte 4 當成 `ctx` 改變 ⇒ 非同步重畫無限循環、頁面凍住），改用函式內的區域 context。
- **Gradio 6.17.3**：`visible="hidden"` 在 Column 的初始狀態藏不住（上傳模式露出編輯器）⇒ 用一般 `visible=False`。

- **本 session 的工作目錄就是子專案資料夾**，Windows 不准改名正被當工作目錄的資料夾。改名採「逐項搬進新資料夾」，舊資料夾留空，由本人事後移除。
- `git mv` 只搬被追蹤的檔；`models/`（9.7 GB）、`datasets/`、`logs/`、`.env`、`.venv` 要在檔案系統層一起搬。
- `.gitignore` 過渡期新舊規則並存，確認 `git status` 沒有大檔才收掉舊規則。
- 搬之前先停掉 app 與 ollama 容器（它們 bind mount 了 `models/`）。
- 本機 `.env` 的 `OLLAMA_MODEL_NAME` 要跟著改（不進版控）；Ollama 用 `ollama cp` 複製新標籤，舊標籤保留。
