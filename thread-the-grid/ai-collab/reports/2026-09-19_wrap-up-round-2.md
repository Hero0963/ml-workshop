# 收尾第二輪報告 — 改名 thread-the-grid、UI 整併、授權、權重發佈準備（2026-09-19）

> 第一輪總帳：[`2026-09-19_project-wrap-up.md`](2026-09-19_project-wrap-up.md)。本輪的逐步紀錄與已定案決策：
> [`../plans/2026-09-19_wrap-up-round-2.md`](../plans/2026-09-19_wrap-up-round-2.md)。所有數字都是本輪實測。

---

## 0. 一頁結論

| 項目 | 結果 |
|---|---|
| **主機連不到 Docker 埠** | ✅ 根因是 `.wslconfig` 的 `networkingMode=mirrored`（2026-09-18 由另一個 agent 當成「可選加分項」加上）搭配 Docker Desktop 4.32。本人定案改回 NAT；對照組（無關的最小容器）從 Windows 回 200、`com.docker.backend` 開始監聽 |
| **第一輪沒驗到的兩項** | ✅ `start.py` 從主機輪詢到 healthy；Svelte 在 Windows Chrome 裡按解題 9 種 solver 過裁判 |
| **Gradio 整併** | ✅ 兩個分頁：Generate（4×4／5×5／6×6、Open in editor）、Solve（Upload screenshot／Editor 兩種模式、Edit in editor）；Echo 分頁拿掉、API 保留 |
| **Svelte 補齊** | ✅ 出題、空白盤、讀截圖；一組大小共用給 Generate 與 Blank grid（本人驗收後定案）|
| **新 API** | ✅ `POST /api/puzzle/generate`（4–6，出題器放棄時重試 3 次）；Gradio 與 Svelte 都走它 |
| **Gradio 6.15.1 → 6.17.3** | ✅ 本人授權並親自 `uv lock`；理由見 §2 |
| **授權** | ✅ 子專案層 Apache-2.0（官方全文）；README 中英授權一節 |
| **改名** | ✅ `linkedin-zip-challenge` → `thread-the-grid`，內部前綴 `threadgrid`；README 不提 LinkedIn |
| **驗證** | ✅ `pytest` **344 passed, 8 xfailed**；瀏覽器對 Docker（7440）：Gradio **20/20**、Svelte **12/12**、9 solver **9/9**＋不可解 3/3 |
| **權重** | 🔶 RL：GitHub **draft** release 已建、下載 SHA-256 相符、未公開；VLM：上傳資料夾與 model card 已備妥，**等本人登入 Hugging Face** |

---

## 1. 做了什麼

### 1.1 介面

- **Gradio**（`src/ui/gradio_app.py`）：出題改走 API（Adapter 原則，以前是同程序直接呼叫出題器）；
  三個解題分頁合成 `Solve` 一頁兩種模式；讀圖結果可一鍵帶進編輯器修正（以前要手動把 Python literal 貼到 Naive 分頁）；
  牆清單改成多選標籤（按 × 刪除），取代「選一列再按刪除」。
- **Svelte**（`src/custom_components/puzzle_editor/frontend/Index.svelte`）：新增 Generate、Blank grid、Read screenshot；
  讀到的盤面直接放上畫布；左側只剩「1. Start from」「2. Solve」。
- **API**：`src/app/routers/puzzle.py`＋`schemas/puzzle.py`＋`tests/test_puzzle_api.py`（7 個測試：三種尺寸都用 CP-SAT 解開並過裁判、超出範圍 422、重試、一直失敗回 500）。

### 1.2 改名

| 改了 | 刻意不改 |
|---|---|
| 資料夾、`pyproject` 名稱、三份 compose、`start.py`、`.env.example`、session-brief hook、pre-commit 註解、舊 RL 腳本裡的路徑、Swagger／Gradio／Svelte 標題、現行文件 | VLM 訓練 prompt 裡的「Zip puzzle」（改了模型失準）、Ollama volume `linkedin-zip-challenge_ollama_data`（Docker 不能改名）、有日期的歷史紀錄（報告、dev_log、舊計畫書、驗收輸出）、本機 worktree／分支名 |

資料夾用「逐項搬」而不是改名：本 session 的工作目錄就是舊資料夾，Windows 不准改名。259 個追蹤檔全數到位、新增 11 個檔；
`.venv` 被 VS Code 的 Black 語言伺服器佔用沒搬，在新資料夾重建。根目錄 `.gitignore` 刻意保留舊路徑規則（其他 checkout 還沒搬）。

---

## 2. 途中發現並修掉的問題（每一個都有最小重現）

| 問題 | 根因 | 修法 | 證據 |
|---|---|---|---|
| Gradio 牆清單只顯示第 1 道 | Gradio 6.15.1 唯讀 Dataframe「欄數不變、列數增加」不重畫（最小重現 4 種寫法全錯）；6.26.0 才修 | 改用多選標籤 | 臨時隔離環境逐版測 6.20／6.24／6.26／6.27 |
| 「帶入編輯器」後整頁卡在 processing 或凍住 | Gradio 6.15.1 前端：同一事件切分頁＋改區塊顯示 | **升級 6.17.3** | 逐版二分：6.15.1／6.15.2／6.16.0 壞、6.17.3 好；6.18+ 要 `huggingface-hub>=1.x`，與 `transformers<5` 衝突 ⇒ 6.17.3 是可用的最高版。連帶 fastapi 0.119.1→0.141.1、starlette 0.48.0→1.6.0、新增 annotated-doc 0.0.5；升級後全測 344 passed |
| 上傳模式底下露出編輯器 | 為避開上一個 bug 用的 `visible="hidden"`，在 6.17.3 的 Column 初始狀態藏不住 | 升級後不再需要，改回 `visible=False` | 端到端補上「剛打開頁面」的檢查 |
| Svelte：Generate 後畫布空白，要再點一下 | 先畫圖、後改 canvas 尺寸，而改尺寸會清空畫布 | DOM 更新後再畫（`tick()`）| 讀畫布像素：換尺寸後透明像素 = 寬×高 → 修後 0 |
| 上一個修法讓 Svelte 整頁凍住 | Svelte 4 把 `ctx.fillStyle = …` 當成元件變數 `ctx` 改變 ⇒ 非同步重畫無限循環 | 畫圖改用函式內的區域 context | 主執行緒逾時探測 |
| 改名後 app 的 compose 專案仍叫 `zip-app-…` | `start.py` 的 `stack_name()` 只改了說明文字 | 補改程式；重跑前掃過全部舊識別字 | `docker ps` 名稱 |

**教訓**：前兩輪瀏覽器測試都只驗「結果對不對」，沒驗「使用者看得到」——初始狀態與畫布內容是本人實際操作才抓到的。現在兩項都寫進了端到端檢查。

---

## 3. 名稱與授權的查證（2026-09-19，一手資料）

- **LinkedIn 已申請「ZIP」商標**：USPTO 申請號 [99060408](https://tsdr.uspto.gov/statusview/sn99060408)，2025-02-27 申請，第 9／41 類「邏輯與益智遊戲的軟體與服務」，**2026-08-11 公告進入異議期**。⇒ 名稱連 zip 都拿掉。
- 候選名查重：`number-trail` 已有同玩法商業 App（Num Trail／NumTrail）而排除；`thread-the-grid` 查無同名 App 與 repo。
- [LinkedIn 品牌規範](https://brand.linkedin.com/policies)：不得以造成來源／贊助混淆的方式使用其名稱，並要求不模仿平台外觀。
- 玩法本身不受著作權保護、具體視覺呈現受保護：[Tetris Holding v. Xio（2012）](https://en.wikipedia.org/wiki/Tetris_Holding,_LLC_v._Xio_Interactive,_Inc.)。本專案的渲染器刻意仿照真實截圖（`render_puzzle.py:2`）——它是訓練資料產生器、不是可玩的遊戲，風險判斷為低，未處理。
- 授權：Qwen3.5-4B 為 Apache-2.0（HF API 一手）；`deep-learning-karpathy/references/` 保留了 Karpathy 的 MIT 聲明，合規。

---

## 4. 權重發佈狀態

| 模型 | 狀態 | 下一步 |
|---|---|---|
| RL `bc_multi_456_e6`（14 MB）| GitHub draft release `thread-the-grid-models-v1`，附件 `threadgrid-rl-bc_multi_456_e6.zip`，下載 SHA-256 `653efaa5…0a28` 與原檔相同，匿名存取 404 | 本人同意後 `gh release edit --draft=false` |
| VLM（9.1 GB）| `models/hf-release/threadgrid-qwen35-4b-p4c-gguf/`：兩個 GGUF（hardlink，SHA-256 與既有紀錄相同）、Modelfile、model card、LoRA adapter、Qwen 的 LICENSE | 本人 `hf auth login` → private repo 上傳 → 下載核對、匯入、`vision_check.py` → 本人同意後公開 |

完整步驟：[`../model-weights.md`](../model-weights.md)。

---

## 5. 本機環境變動（給作者）

| 動作 | 還原方式 |
|---|---|
| `%USERPROFILE%\.wslconfig`：`networkingMode=mirrored` 註解掉並寫明原因（本人定案永久改回 NAT）；原檔備份在當次 scratchpad | 拿掉那行的 `#`，再 `wsl --shutdown` |
| 別專案的 4 個容器 `docker update --restart unless-stopped`（`ai_translator_app`、`pdf2zh_server`、`omni_parser`、`speech_motion_aligner`）| `docker update --restart always <名稱>`；那些專案下次 `compose up` 也會蓋回 |
| 第一輪留下的 3 組測試容器：本人自己 `compose down`（已驗證）| — |
| Ollama 多一個標籤 `threadgrid-qwen35-4b-p4c:f16`（`ollama cp`，同一份資料）；舊標籤保留 | — |
| `zip-vlm/thread-the-grid/.env` 的 `OLLAMA_MODEL_NAME` 改成新標籤（只改那一行）| 改回舊標籤（兩個都在）|
| `zip_ollama_server` 停止（未刪）；本 session 建的 `zip-app-zip-vlm` 兩個容器已 `compose down` | — |
| 3 張過時截圖移到 `soft-delete/20260919-191902/` | 從那裡搬回 |
| 舊資料夾 `zip-vlm/linkedin-zip-challenge/` 只剩 4.8 GB 的 `.venv` | 本人關掉 VS Code 與 Claude 後自行刪除 |
| Gradio 升級（本人 `uv lock`）| `git revert` 該 commit 的 `uv.lock` |

---

## 6. 出處

本專案：[`../plans/2026-09-19_wrap-up-round-2.md`](../plans/2026-09-19_wrap-up-round-2.md)、[`artifacts/wrap-up-round-2/`](artifacts/wrap-up-round-2/)（端到端檢查紀錄）。
外部（查證 2026-09-19）：[USPTO TSDR 99060408](https://tsdr.uspto.gov/statusview/sn99060408)、[LinkedIn Brand Policies](https://brand.linkedin.com/policies)、
[Tetris Holding v. Xio](https://en.wikipedia.org/wiki/Tetris_Holding,_LLC_v._Xio_Interactive,_Inc.)、[Nominative use](https://en.wikipedia.org/wiki/Nominative_use)、
[microsoft/WSL#41284](https://github.com/Microsoft/wsl/issues/41284)（新版 Docker Desktop 在 mirrored 模式下 localhost 可用）、
[Claude Code memory 文件](https://code.claude.com/docs/en/memory)（AGENTS.md 讀取條件）、PyPI（gradio 各版的 `huggingface-hub` 需求）。
