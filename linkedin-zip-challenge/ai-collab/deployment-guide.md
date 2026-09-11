# 部署指南 — 用 Docker 起整個服務

> 2026-09-12 實測寫成（分支 `feat/rl-a2-training`），同日 image 瘦身後整份重新驗收（分支 `feat/infra-slim-image`）。
> 每一條指令與輸出都跑過，沒跑過的會標「未驗」。
> 想知道模型怎麼推論 → [`notes/03-inference-and-serving.md`](notes/03-inference-and-serving.md)；
> 架構與模組職責 → [`project_guide.md`](project_guide.md)。

---

## 1. 一句話

```powershell
cd linkedin-zip-challenge
docker compose up -d --build
```

兩個容器起來，然後開 <http://127.0.0.1:7440/ui>。或者用 `python start.py`：它會等到 API 真的回 200 才說好，
並告訴你缺了什麼（ollama、RL 權重）。

**要多久**（2026-09-12 本機實測）：建 image **110 秒**，其中 `uv sync` 從網路下載全部 wheel 佔 89 秒；
當時 Svelte 前端那一段吃了快取，**全新的機器還要再加上 `npm install` ＋ `npm run build`（未量）**。
image 建好之後，`start.py` 從 `up` 到 health 200 約 **10 秒**。

---

## 2. 這個 stack 裡有什麼

| 容器 | 是什麼 | 埠 | 要 GPU 嗎 |
|---|---|---|---|
| `zip_challenge_prod_app` | FastAPI ＋ Gradio ＋ 已建好的 Svelte 編輯器 ＋ **4 種 solver（含 RL）** | `7440` | ❌ 不用 |
| `zip_ollama_server` | 服務微調過的視覺語言模型，`/api/vision/solve` 靠它讀截圖 | `11435`（對外）→ `11434`（容器內）| ✅ 要 |

**app 容器不要 GPU 是刻意的**：RL 推論走 CPU 就夠（一題 4×4 約 40 毫秒），GPU 留給 ollama。
所以 app image 的基底是純 Python（`python:3.11-slim-trixie`），不是 CUDA image——見 §7 的 image 大小。
（2026-09-12 以前基底是 CUDA image，app 的 log 開頭會印 `WARNING: The NVIDIA Driver was not detected`；
換基底後這行已經不見了。）

### 三個入口

| 網址 | 是什麼 |
|---|---|
| <http://127.0.0.1:7440/ui> | Gradio 主控台（解題、出題、讀截圖）|
| <http://127.0.0.1:7440/svelte-ui/> | Svelte 互動式編輯器 |
| <http://127.0.0.1:7440/docs> | OpenAPI 文件（可直接試打）|

---

## 3. 兩份 compose 的差別

| | `docker-compose.yml`（正式）| `docker-compose.dev.yml`（開發）|
|---|---|---|
| 程式碼 | **烤進 image**，改了要重建 | `./src` 掛進去，`--reload` 熱更新 |
| Svelte | 建置階段編好，走 `7440/svelte-ui/` | 另開一個容器跑 vite：<http://127.0.0.1:5173/svelte-ui/>。**`7440/svelte-ui/` 在 dev 是 404**（`./src` 蓋掉 image 裡的程式碼，而 host 沒有編好的 `dist/`）|
| 啟動 | `docker compose up -d --build` | `docker compose -f docker-compose.dev.yml up -d --build`（或 `python start.py --dev`）|
| image | `zip-challenge-app:prod` | `zip-challenge-app:dev`（和 prod 共用 5.858 GB 的層，並存不多佔空間）|

dev 的編輯器頁面在 5173，但它打 API 的位址**寫死**在 `Index.svelte` 的 `API_BASE_URL = "http://127.0.0.1:7440"`，
也就是瀏覽器直接跨來源打 app；後端 CORS 是 `allow_origins=["*"]`，從 5173 送 preflight 實測 200。
所以 compose 裡的 `VITE_API_URL` **其實沒有被讀**。⚠ 未驗：在瀏覽器裡實際按下解題按鈕。

⚠ **兩份是二選一，不是疊加**：ollama 容器同名（`zip_ollama_server`）、app 用同一個埠（`7440`）、
共用 `ollama_data` volume。換一份前先 `python start.py --down`（或 `--down --dev`）。
app 容器名稱其實不同（`zip_challenge_prod_app`／`zip_challenge_dev_app`），image 也各自命名
（`zip-challenge-app:prod`／`zip-challenge-app:dev`）——2026-09-12 以前兩份都叫
`linkedin-zip-challenge-zip-challenge-app`，**建一份就蓋掉另一份**，`--no-build` 可能起到錯的那個。

### ⚠ 整台機器只有一組 stack——多個 worktree 會互相接管

compose 的專案名稱預設是**目錄名**，而每個 worktree 的子目錄都叫 `linkedin-zip-challenge`；
再加上固定的容器名稱與埠號，**不管從哪個 worktree 執行 `up`，操作的都是同一組容器**。
2026-09-12 實測：`zip-rl` 起的 stack，從 `zip-infra` 一跑 `up` 就被換成 `zip-infra` 的程式碼與 image。

這是刻意維持的：ollama 佔 GPU，一台機器跑兩份沒有意義。所以規則是——
**要動 stack 之前先看是誰起的**，接管前先講一聲：

```powershell
docker inspect zip_challenge_prod_app --format '{{json .Config.Labels}}'   # 看 com.docker.compose.project.working_dir
```

（PowerShell 5.1 會吃掉 `--format` 裡的雙引號，所以用 `json` 整包印，不要寫 `index .Config.Labels "..."`。）

---

## 4. RL solver 要的東西：`models/` 掛載

RL solver 需要訓練好的 checkpoint，而 **`models/` 有 6.6 GB 且不進版控**。
所以它是**唯讀掛載**進容器的（`./models:/app/models:ro`），不是烤進 image：

- image 因此不會膨脹，`.dockerignore` 也把 `models/`、`datasets/`、`logs/` 全排除；
- 沒有 checkpoint 的機器**照樣起得來**——RL solver 回 **503**，其他三種 solver 正常。

目前服務的 checkpoint（`src/core/rl/solver_service.py` 的 `RUN_ID_BY_SIZE`）：

| 盤面 | run id | 沒有它會怎樣 |
|---|---|---|
| 4×4／5×5／6×6 | **`bc_multi_456`**（一個模型全包）| 503 Service Unavailable |
| 其他尺寸 | 沒有 | **400 Bad Request**，訊息會說支援哪些尺寸 |

一個模型服務三個尺寸是量出來的結果，不是省事：它在三個盤面**都**贏過同資料訓練的單尺寸專用模型
（deterministic 0.9404／0.7496／0.5205 vs 0.9042／0.7131／0.4890），訓練成本還略低。
詳見 [`reports/2026-09-12_rl-wrap-up.md`](reports/2026-09-12_rl-wrap-up.md) §2。

---

## 5. 驗收：這些都實際跑過（2026-09-12，瘦身後的 image）

題目用 `src/core/tests/conftest.py` 的 `puzzle_01`（6×6）、`puzzle_04`（7×7），外加一題手寫的 4×4。
「解出」＝回傳路徑的格數等於盤面格數。

**正式版 `python start.py`**

```
health       200 {"status":"ok"}
gradio /ui   200      svelte /svelte-ui/  200      docs  200      openapi  200

                         4x4 解出   秒      6x6 解出   秒
DFS                      16/16    0.100    36/36    0.367
A* (heapq)               16/16    0.103    36/36    0.672
CP-SAT                   16/16    0.096    36/36    0.514
RL (behaviour cloning)   16/16    4.267    沒解出   0.317     <- 全部都是 HTTP 200
7x7 via RL   400 {"detail":"No RL policy for a 7x7 board; trained sizes are [4, 5, 6]."}
```

⚠ **200 不等於解出來**：solver 放棄時也回 200，`solution_path` 是
`Solver 'RL (behaviour cloning)' could not find a solution.`——看狀態碼驗收會漏掉這件事。
RL 是抽樣的：同一題 6×6 連打 20 次**解出 10/20**、4×4 **20/20**（熱的時候中位數 0.106 秒；
上表的 4.267 秒是第一次呼叫在載模型）。換基底前的舊 image 在同一題 6×6 也是 2 次中 1 次，
⇒ 這是這題對 RL 的難度，不是瘦身造成的：同一個 torch wheel、同一個 checkpoint，容器內測試全過。

**容器內跑完整測試**（「系統函式庫夠不夠」的真正答案，health 200 回答不了這個問題）

```
$ docker run --rm --env-file .env -v <models>:/app/models:ro zip-challenge-app:prod sh -c '... uv run pytest -q'
Python 3.11.16
torch 2.4.1+cu121 cuda_available False py 3.11.16
276 passed, 8 xfailed in 19.02s
```

**app → ollama**（從 app 容器裡面打 compose 網路的 `ollama:11434`）與容器狀態

```
app -> ollama 200 ['zip-qwen35-4b-p4c:f16', 'gemma4:e4b-it-q8_0', 'gemma4:e4b'] ...
zip_challenge_prod_app  Up (healthy)  zip-challenge-app:prod   513.2MiB
zip_ollama_server       Up (healthy)  ollama/ollama:latest     300MiB
```

**開發版 `python start.py --dev`**（11 秒到 health 200）

```
health 200   /ui 200   /docs 200   vite http://127.0.0.1:5173/svelte-ui/ 200
四種 solver 在 4x4 全部解出、7x7 via RL 400（同正式版）
http://127.0.0.1:7440/svelte-ui/  404      <- 預期行為，見 §3
uvicorn: Will watch for changes in these directories: ['/app/src']
         Started reloader process [11] using WatchFiles
```

⚠ **未驗**：hot reload 實際被觸發（要改 `src/` 才測得到）。上面只證明 reloader 有起來、盯的是對的目錄。

---

## 6. 修掉的缺陷（都是「看起來有起來、其實沒有」那種）

| 缺陷 | 症狀 | 修法 |
|---|---|---|
| **兩份 compose 建出同名 image**（2026-09-12 瘦身時修） | 建 dev 就蓋掉 prod 的 image，`start.py --no-build` 可能起到錯的那個 | 兩份各加 `image:`（`zip-challenge-app:prod`／`:dev`）|
| **`start.py --dev` 印出打不開的網址**（同上） | dev 把 `./src` 掛進容器、host 沒有編好的 `dist/` ⇒ `7440/svelte-ui/` 是 404，腳本卻叫你去開它；vite 的網址也少了 `/svelte-ui/` | dev 模式改印 `http://127.0.0.1:5173/svelte-ui/` |
| **正式 image 沒有 `CMD`** | `docker compose up` 建好 image、容器 Up，**但裡面沒有 server**（base image 的預設指令是 shell）| `.devcontainer/Dockerfile` 加 `CMD` 跑 uvicorn |
| **開發 image 停在 `tail -f /dev/null`** | 靠 `start.py` 把 server exec 進去，但那行**沒有 `-d`**、`subprocess.run` 會一直等 ⇒ 後面的 healthcheck 永遠跑不到 | `Dockerfile.dev` 自己起 uvicorn `--reload`；腳本刪掉那一步 |
| **`models/` 進了 build context** | 每次建置把 **6.6 GB** 送給 daemon | `.dockerignore` 排除，改成唯讀 volume |
| **沒有 healthcheck** | `docker ps` 的 Up 不代表服務活著 | 兩個服務都補 healthcheck（app 打 `/api/echo/health`）|

---

## 7. 常見問題

**Q：`/api/vision/solve` 回 503。**
ollama 還在載模型，或容器沒起來。`docker compose logs -f ollama`；第一次呼叫會付模型載入時間（4B 模型約 1–2 分鐘）。

**Q：視覺辨識結果很差。**
⚠ **先檢查 `.env` 的模型與 prompt 是否配對。** 2026-09-12 在 `zip-rl` worktree 實測到
`.env` 停在 `OLLAMA_MODEL_NAME=openbmb/minicpm-o2.6`（2026-08-15 的舊設定），
而 `.env.example` 現在的正確值是：

```
OLLAMA_MODEL_NAME=zip-qwen35-4b-p4c:f16
VISION_PROMPT_VARIANT=finetune
```

**模型與 prompt 必須配對**：`finetune` 的 prompt 配未微調的模型，等於問它一個沒訓練過的問題。
`.env` 不進版控，每個 worktree 各一份，**所以它會各自過期**。

**Q：RL solver 回 503。**
`models/rl_a2/bc_multi_456/checkpoints/model_final.zip` 不存在。訓練它，或確認 volume 有掛上
（`docker exec zip_challenge_prod_app ls /app/models/rl_a2`）。

**Q：image 為什麼從 22.9 GB 變成 5.86 GB？還能更小嗎？**
2026-09-12 把基底從 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` 換成 `python:3.11-slim-trixie`：

```
zip-challenge-app:prod                           5.86GB     # 換基底後
linkedin-zip-challenge-zip-challenge-app:latest  22.9GB     # 換基底前
```

舊基底裡有兩樣東西 app 從來沒用過：conda（附一份 torch 2.3.0，**7.59 GB**）和 CUDA toolkit／cuDNN
（**4.79 ＋ 2.45 ＋ 2.01 GB**）。app 容器不用 GPU，這些全是死重。
**系統函式庫一個都不用補**：`src/` 沒有 import `cv2`，`matplotlib` 雖然被間接帶進來，但它的 Linux wheel
自帶需要的函式庫。在新 image 裡跑完整測試 `276 passed, 8 xfailed`，和 host 一樣。
順帶一提，uv 現在直接用 image 內建的 CPython 3.11.16，不再另外下載一份（舊 image 多了 96 MB）。

**剩下的 5.66 GB 幾乎全是 `uv sync`**，而它的大宗是 `torch 2.4.1+cu121` 帶進來的 12 個 `nvidia-*` wheel。
換成 CPU 版 torch 才能再砍一大截，但那要改 `pyproject.toml`／`uv.lock`（整個專案共用的相依設定，
訓練一定要 CUDA 版）⇒ **不是 Docker 層能單獨決定的事，沒有做**。要做的話得另外設計
「服務用 CPU torch、訓練用 CUDA torch」的相依分組。

**Q：`Found orphan containers ([svelte_frontend_dev])`。**
上一次用 dev compose 留下的。`docker compose -f docker-compose.dev.yml down` 或加 `--remove-orphans`。

---

## 8. 不用 Docker 的起法（開發最快）

```powershell
cd linkedin-zip-challenge
uv sync
uv run python -m src.app.main       # http://127.0.0.1:7440/ui
```

這條路不會有 ollama，`/api/vision/solve` 需要另外自己跑一個（`.env` 的 `OLLAMA_PROVIDER_URL`
預設指向 `http://127.0.0.1:11435/v1`，也就是 compose 對外開的那個埠）。
