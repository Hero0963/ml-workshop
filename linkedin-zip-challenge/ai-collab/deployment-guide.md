# 部署指南 — 用 Docker 起整個服務

> 2026-09-12 實測寫成（分支 `feat/rl-a2-training`），同日在分支 `feat/infra-slim-image` 做了兩件事並整份重新驗收：
> **app image 瘦身**（22.9 GB → 5.86 GB）與**根治 worktree 撞名**（app 每個 checkout 一組、ollama 整台機器一個）。
> 每一條指令與輸出都跑過，沒跑過的會標「未驗」。
> 想知道模型怎麼推論 → [`notes/03-inference-and-serving.md`](notes/03-inference-and-serving.md)；
> 架構與模組職責 → [`project_guide.md`](project_guide.md)。

---

## 1. 一句話

```powershell
cd linkedin-zip-challenge
python start.py           # 正式版；開發版加 --dev
```

然後開 <http://127.0.0.1:7440/ui>。`start.py` 做四件事：從範本建 `.env` → 確認整台機器的 ollama 在跑
（沒跑才起）→ 建並起**這個 checkout 的** app → 等到 API 真的回 200，並告訴你缺了什麼（ollama、RL 權重）。

不用 `start.py` 的話是兩行（ollama 從 app 的 compose 拆出來了，見 §3）：

```powershell
docker compose -f docker-compose.ollama.yml up -d    # 整台機器一次
docker compose up -d --build                         # 這個 checkout 的 app
```

**要多久**（2026-09-12 本機實測）：建 image **110 秒**，其中 `uv sync` 從網路下載全部 wheel 佔 89 秒；
當時 Svelte 前端那一段吃了快取，**全新的機器還要再加上 `npm install` ＋ `npm run build`（未量）**。
image 建好之後，`start.py` 從 `up` 到 health 200 約 **10 秒**（dev 約 20 秒）。

---

## 2. 這個 stack 裡有什麼

| 容器 | compose 專案 | 是什麼 | 埠 | 要 GPU 嗎 |
|---|---|---|---|---|
| `zip-app-<checkout>-zip-challenge-app-1` | `zip-app-<checkout>`（**每個 checkout 一個**）| FastAPI ＋ Gradio ＋ 已建好的 Svelte 編輯器 ＋ **4 種 solver（含 RL）** | `APP_PORT`（預設 `7440`）| ❌ 不用 |
| `zip_ollama_server` | `zip-ollama`（**整台機器一個**）| 服務微調過的視覺語言模型，`/api/vision/solve` 靠它讀截圖 | `11435`（對外）→ `11434`（容器內）| ✅ 要 |

`<checkout>` 是 `linkedin-zip-challenge` 的**上一層目錄名**：主 checkout 是 `ml-workshop`，worktree 是 `zip-infra`、`zip-rl`……
所以在 `zip-infra` 裡，app 容器叫 `zip-app-zip-infra-zip-challenge-app-1`。

**app 容器不要 GPU 是刻意的**：RL 推論走 CPU 就夠（一題 4×4 約 40 毫秒），GPU 留給 ollama。
所以 app image 的基底是純 Python（`python:3.11-slim-trixie`），不是 CUDA image——見 §8 的 image 大小。
（2026-09-12 以前基底是 CUDA image，app 的 log 開頭會印 `WARNING: The NVIDIA Driver was not detected`；
換基底後這行已經不見了。）

### 三個入口

| 網址 | 是什麼 |
|---|---|
| <http://127.0.0.1:7440/ui> | Gradio 主控台（解題、出題、讀截圖）|
| <http://127.0.0.1:7440/svelte-ui/> | Svelte 互動式編輯器（開發版改走 vite，見 §4）|
| <http://127.0.0.1:7440/docs> | OpenAPI 文件（可直接試打）|

埠號是 `.env` 的 `APP_PORT`；換了埠，網址跟著換（`start.py` 會印出正確的那個）。

---

## 3. 身分：app 每個 checkout 一組，ollama 整台機器一個

**白話**：以前每個 worktree 的服務都掛同一張名牌。compose 認名牌不認人，所以在 A 資料夾按「啟動」，
它會把 B 資料夾正在跑的那一組當成自己的、拆掉重蓋——而且一聲不吭。2026-09-12 就這樣把 `zip-rl` 起的 stack 換掉了。

**撞在哪**（四個地方，前三個造成「安靜地接管」）：

| 撞名的東西 | 從哪來 | 為什麼每個 worktree 都一樣 |
|---|---|---|
| compose 專案名 | 預設＝compose 檔所在的**目錄名** | 每個 worktree 的子目錄都叫 `linkedin-zip-challenge` |
| 容器名 | 寫死的 `container_name:` | 寫在版控裡的檔案 |
| host 埠 | `.env` 的 `APP_PORT` | `.env` 是從 main 複製過來的 |
| image 標籤 | 預設＝`<專案名>-<服務名>` | 跟著專案名一起撞 |

**根本原因**是兩種身分被綁在同一個 compose 專案裡：app 服務的是「**這個 checkout 的程式碼**」，應該每個 checkout 一份；
ollama 佔的是 **GPU**，應該整台機器一份。所以拆開，各自拿符合它本質的名字：

| | 專案名 | 誰決定 | 行為 |
|---|---|---|---|
| ollama | `zip-ollama` | 寫死在 `docker-compose.ollama.yml` 的 `name:` | **刻意的單例**。`start.py` 只在它沒跑時才起，**在跑就完全不碰** |
| 正式 app | `zip-app-<checkout>` | `start.py` 的 `stack_name()`，用 `-p` 傳給 compose | 每個 checkout 各自的容器、網路、image |
| 開發 app | `zip-dev-<checkout>` | 同上 | 同一個 checkout 的 prod／dev 用同一個埠，`start.py` 起一個時會先停另一個 |

幾個細節，都是實測出來的：

- **為什麼在跑的 ollama 一定不能碰**：ollama 掛了 `./models`，那是「當初起它的那個 checkout」的路徑。
  從別的 checkout 對它 `up`，compose 會認為設定變了而**重建**它，把別人正在用的模型卸掉。實測：
  `docker compose --dry-run -f docker-compose.ollama.yml up -d` → `Container zip_ollama_server  Recreate`。
- **app 怎麼找 ollama**：不同專案不在同一個網路，所以 app 走 ollama 對外開的埠，
  `OLLAMA_PROVIDER_URL=http://host.docker.internal:11435/v1`——和不用 Docker 時的 `127.0.0.1:11435` 是同一個概念。
  Docker Desktop 自己認得 `host.docker.internal`；compose 另外加了 `extra_hosts: host-gateway` 讓 Linux 也認得。
- **剩下唯一會撞的是 host 埠，而它會大聲失敗**（`port is already allocated`），不會安靜接管。
  兩個 worktree 要同時起 app，就在其中一個的 `.env` 改 `APP_PORT`（開發版再改 `SVELTE_PORT`）。
  `python start.py --status` 會列出整台機器上在跑的 compose 專案、各自從哪個目錄起的。
- **Svelte 編輯器原本寫死 `127.0.0.1:7440`，2026-09-12 一併修掉**：不在 7440 的 checkout，它的編輯器會打到
  7440 那一個 app（別的 checkout 的程式）。現在建置版走**同源相對路徑**（`API_BASE_URL` 是空字串），
  所以 app 在哪個埠它就打哪個埠；dev 因為頁面來自 vite 的 5173，由 compose 的 `VITE_API_URL` 提供位址。
  Gradio 與 API 本身從來沒有這個問題——它們讀的是 `.env` 的 `APP_PORT`。
- **不用 `start.py` 的時候**：專案名退回預設，正式版是目錄名 `linkedin-zip-challenge`、開發版是檔案裡寫的
  `linkedin-zip-challenge-dev`（兩者分開，image 才不會互相覆蓋）。單一 checkout 沒問題；**多個 worktree 請用 `start.py`**
  （或自己給 `docker compose -p <名字>`）。

**驗證**（2026-09-12）：在 scratchpad 複製一份假的第二個 checkout（`APP_PORT=7441`，沒有 `models/`），和 `zip-infra` 同時起：

```
NAME                    STATUS        CONFIG FILES
zip-app-fake-worktree   running(1)    ...\scratchpad\fake-worktree\linkedin-zip-challenge\docker-compose.yml
zip-app-zip-infra       running(1)    D:\...\zip-infra\linkedin-zip-challenge\docker-compose.yml
zip-ollama              running(1)    D:\...\zip-infra\linkedin-zip-challenge\docker-compose.ollama.yml

  :7441 DFS                      200 solved
  :7441 RL (behaviour cloning)   503  No checkpoint at /app/models/rl_a2/bc_multi_456/...   <- 它真的是另一份
  :7440 DFS                      200 solved
  :7440 RL (behaviour cloning)   200 solved

zip-infra 的 app 與 ollama：對方起來前、起來後、關掉後，容器 ID 與啟動時間三次完全相同
```

### 從舊布局遷移（每台機器一次）

2026-09-12 以前的 stack 屬於專案 `linkedin-zip-challenge`，而它的 `zip_ollama_server` 會和新的 `zip-ollama` 撞名：

```powershell
docker compose -p linkedin-zip-challenge down --remove-orphans
```

- 還沒合併新 compose 的 worktree（舊檔案）若再 `up`，會因為 `zip_ollama_server` 已存在而**失敗**——大聲的，不會蓋掉別人。先合併 `main` 再起。
- 新的 ollama 起來時 compose 會印一行 warning：volume `linkedin-zip-challenge_ollama_data` 是舊專案建的。
  **無害**，模型照樣都在。刻意不改成 `external: true`——那會讓沒有這個 volume 的新機器直接起不來。

---

## 4. 兩份 app compose 的差別

| | `docker-compose.yml`（正式）| `docker-compose.dev.yml`（開發）|
|---|---|---|
| 程式碼 | **烤進 image**，改了要重建 | `./src` 掛進去，`--reload` 熱更新 |
| Svelte | 建置階段編好，走 `7440/svelte-ui/` | 另開一個容器跑 vite：<http://127.0.0.1:5173/svelte-ui/>。**`7440/svelte-ui/` 在 dev 是 404**（`./src` 蓋掉 image 裡的程式碼，而 host 沒有編好的 `dist/`）|
| 啟動 | `python start.py` | `python start.py --dev` |
| 專案／image | `zip-app-<checkout>`／`zip-app-<checkout>-zip-challenge-app` | `zip-dev-<checkout>`／`zip-dev-<checkout>-zip-challenge-app`（和正式版共用 5.858 GB 的層，並存不多佔空間）|

dev 的編輯器頁面在 5173，瀏覽器直接跨來源打 app（後端 CORS 是 `allow_origins=["*"]`，從 5173 送 preflight 實測 200），
位址由 compose 的 `VITE_API_URL=http://127.0.0.1:${APP_PORT}` 提供——**建置版不吃這個變數**，走同源相對路徑。
不透過 Docker 直接 `npm run dev` 的話要自己設 `VITE_API_URL`，否則編輯器會把 API 打到 vite 自己的埠。

已驗到的是：正式版 image 的 `dist/` 裡搜不到 `127.0.0.1:7440`、`/svelte-ui/` 200；dev 的 svelte 容器裡
`VITE_API_URL=http://127.0.0.1:7440`（`docker exec` 實測）。⚠ **未驗**：在瀏覽器裡實際按下解題按鈕。
Vite 在 dev 是**執行期**注入 `import.meta.env`，不開瀏覽器量不到最終值——這一段要靠人工點一次才算數。

---

## 5. RL solver 要的東西：`models/` 掛載

RL solver 需要訓練好的 checkpoint，而 **`models/` 有 6.6 GB 且不進版控**。
所以它是**唯讀掛載**進容器的（`./models:/app/models:ro`），不是烤進 image：

- image 因此不會膨脹，`.dockerignore` 也把 `models/`、`datasets/`、`logs/` 全排除；
- 沒有 checkpoint 的機器**照樣起得來**——RL solver 回 **503**，其他三種 solver 正常。
- 掛的是**這個 checkout 的** `models/`。新 worktree 沒有它（不進版控），要測 RL 就從別的 worktree 複製
  `models/rl_a2/bc_multi_456`（14 MB）過來。

目前服務的 checkpoint（`src/core/rl/solver_service.py` 的 `RUN_ID_BY_SIZE`）：

| 盤面 | run id | 沒有它會怎樣 |
|---|---|---|
| 4×4／5×5／6×6 | **`bc_multi_456`**（一個模型全包）| 503 Service Unavailable |
| 其他尺寸 | 沒有 | **400 Bad Request**，訊息會說支援哪些尺寸 |

一個模型服務三個尺寸是量出來的結果，不是省事：它在三個盤面**都**贏過同資料訓練的單尺寸專用模型
（deterministic 0.9404／0.7496／0.5205 vs 0.9042／0.7131／0.4890），訓練成本還略低。
詳見 [`reports/2026-09-12_rl-wrap-up.md`](reports/2026-09-12_rl-wrap-up.md) §2。

---

## 6. 驗收：這些都實際跑過（2026-09-12，瘦身後的 image ＋ 新布局）

題目用 `src/core/tests/conftest.py` 的 `puzzle_01`（6×6）、`puzzle_04`（7×7），外加一題手寫的 4×4。
「解出」＝回傳路徑的格數等於盤面格數。

**正式版 `python start.py`**（新布局，專案 `zip-app-zip-infra`）

```
health       200 {"status":"ok"}
gradio /ui   200      svelte /svelte-ui/  200      docs  200      openapi  200

                         4x4 解出   秒      6x6 解出   秒
DFS                      16/16    0.109    36/36    0.323
A* (heapq)               16/16    0.146    36/36    0.575
CP-SAT                   16/16    0.094    36/36    0.490
RL (behaviour cloning)   16/16    3.641    沒解出   0.362     <- 全部都是 HTTP 200
7x7 via RL   400 {"detail":"No RL policy for a 7x7 board; trained sizes are [4, 5, 6]."}
```

⚠ **200 不等於解出來**：solver 放棄時也回 200，`solution_path` 是
`Solver 'RL (behaviour cloning)' could not find a solution.`——看狀態碼驗收會漏掉這件事。
RL 是抽樣的：同一題 6×6 連打 20 次**解出 10/20**、4×4 **20/20**（熱的時候中位數 0.106 秒；
上表的 3.641 秒是第一次呼叫在載模型）。換基底前的舊 image 在同一題 6×6 也是 2 次中 1 次，
⇒ 這是這題對 RL 的難度，不是瘦身造成的：同一個 torch wheel、同一個 checkpoint，容器內測試全過。

**容器內跑完整測試**（「系統函式庫夠不夠」的真正答案，health 200 回答不了這個問題）

```
$ docker run --rm --env-file .env -v <models>:/app/models:ro <app image> sh -c '... uv run pytest -q'
Python 3.11.16
torch 2.4.1+cu121 cuda_available False py 3.11.16
276 passed, 8 xfailed in 19.02s
```

**app → ollama**（從 app 容器裡，用 app 實際讀到的 `OLLAMA_PROVIDER_URL`）與容器狀態

```
OLLAMA_PROVIDER_URL = http://host.docker.internal:11435/v1
GET http://host.docker.internal:11435/api/tags -> 200 ['zip-qwen35-4b-p4c:f16', 'gemma4:e4b-it-q8_0'] ...
zip-app-zip-infra-zip-challenge-app-1  Up (healthy)  508.8MiB
zip_ollama_server                      Up (healthy)   61.96MiB
```

**開發版 `python start.py --dev`**（新布局；會先自己停掉 `zip-app-zip-infra`，約 20 秒到 health 200）

```
health 200   /ui 200   /docs 200   vite http://127.0.0.1:5173/svelte-ui/ 200
4x4：DFS／CP-SAT／RL 解出；6x6：DFS／CP-SAT 解出；7x7 via RL 400（同正式版）
http://127.0.0.1:7440/svelte-ui/  404      <- 預期行為，見 §4
uvicorn: Will watch for changes in these directories: ['/app/src']
         Started reloader process [11] using WatchFiles
ollama：切到 dev、再切回正式版，容器 ID 與啟動時間都沒變
```

⚠ **未驗**：hot reload 實際被觸發（要改 `src/` 才測得到）。上面只證明 reloader 有起來、盯的是對的目錄。

---

## 7. 修掉的缺陷（都是「看起來有起來、其實沒有」那種）

| 缺陷 | 症狀 | 修法 |
|---|---|---|
| **worktree 共用同一組 stack**（2026-09-12） | 在一個 worktree `up`，**安靜地**拆掉另一個 worktree 正在跑的服務 | ollama 拆成單例專案 `zip-ollama`；app 專案依 checkout 命名，拿掉寫死的容器名與 image 名（§3）|
| **兩份 compose 建出同名 image**（同上） | 建 dev 就蓋掉 prod 的 image，`start.py --no-build` 可能起到錯的那個 | prod／dev 各是一個專案，image 名稱跟著分開 |
| **`start.py --dev` 印出打不開的網址**（同上） | dev 把 `./src` 掛進容器、host 沒有編好的 `dist/` ⇒ `7440/svelte-ui/` 是 404，腳本卻叫你去開它；vite 的網址也少了 `/svelte-ui/` | dev 模式改印 `http://127.0.0.1:5173/svelte-ui/` |
| **正式 image 沒有 `CMD`** | `docker compose up` 建好 image、容器 Up，**但裡面沒有 server**（base image 的預設指令是 shell）| `.devcontainer/Dockerfile` 加 `CMD` 跑 uvicorn |
| **開發 image 停在 `tail -f /dev/null`** | 靠 `start.py` 把 server exec 進去，但那行**沒有 `-d`**、`subprocess.run` 會一直等 ⇒ 後面的 healthcheck 永遠跑不到 | `Dockerfile.dev` 自己起 uvicorn `--reload`；腳本刪掉那一步 |
| **`models/` 進了 build context** | 每次建置把 **6.6 GB** 送給 daemon | `.dockerignore` 排除，改成唯讀 volume |
| **沒有 healthcheck** | `docker ps` 的 Up 不代表服務活著 | 兩個服務都補 healthcheck（app 打 `/api/echo/health`）|

---

## 8. 常見問題

**Q：`/api/vision/solve` 回 503。**
ollama 還在載模型，或容器沒起來。`docker logs -f zip_ollama_server`；第一次呼叫會付模型載入時間（4B 模型約 1–2 分鐘）。
沒起來就 `docker compose -f docker-compose.ollama.yml up -d`（或直接跑 `python start.py`）。

**Q：`up` 失敗，`port is already allocated`。**
別的 checkout（或 `uv run` 起的 server）佔著這個埠。`python start.py --status` 看是誰；要兩個同時跑，
就在這個 checkout 的 `.env` 改 `APP_PORT`（開發版再改 `SVELTE_PORT`）。

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
這個 checkout 的 `models/rl_a2/bc_multi_456/checkpoints/model_final.zip` 不存在。訓練它、從別的 worktree 複製過來，
或確認 volume 有掛上（`docker compose -p zip-app-<checkout> exec zip-challenge-app ls /app/models/rl_a2`）。

**Q：image 為什麼從 22.9 GB 變成 5.86 GB？還能更小嗎？**
2026-09-12 把基底從 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` 換成 `python:3.11-slim-trixie`：

```
zip-app-zip-infra-zip-challenge-app:latest       5.86GB     # 換基底後
linkedin-zip-challenge-zip-challenge-app:latest  22.9GB     # 換基底前
```

舊基底裡有兩樣東西 app 從來沒用過：conda（附一份 torch 2.3.0，**7.59 GB**）和 CUDA toolkit／cuDNN
（**4.79 ＋ 2.45 ＋ 2.01 GB**）。app 容器不用 GPU，這些全是死重。
**系統函式庫一個都不用補**：`src/` 沒有 import `cv2`，`matplotlib` 雖然被間接帶進來，但它的 Linux wheel
自帶需要的函式庫。在新 image 裡跑完整測試 `276 passed, 8 xfailed`，和 host 一樣。
順帶一提，uv 現在直接用 image 內建的 CPython 3.11.16，不再另外下載一份（舊 image 多了 96 MB）。

**剩下的 5.66 GB 幾乎全是 `uv sync`**：`.venv` 5.5 GB 裡 `nvidia/` 佔 **2.7 GB**、`torch/` **1.6 GB**、`triton/` **0.55 GB**，
都是 `torch 2.4.1+cu121` 帶進來的。換成 CPU 版 torch 才能再砍一大截，但那要改 `pyproject.toml`／`uv.lock`
（整個專案共用的相依設定，訓練一定要 CUDA 版）⇒ **不是 Docker 層能單獨決定的事，沒有做**。要做的話得另外設計
「服務用 CPU torch、訓練用 CUDA torch」的相依分組。

**Q：`Found orphan containers`。**
同一個專案裡有現在的 compose 檔沒定義的容器（例如舊布局留下的）。`python start.py --down` 會帶 `--remove-orphans` 清掉。

---

## 9. 不用 Docker 的起法（開發最快）

```powershell
cd linkedin-zip-challenge
uv sync
uv run python -m src.app.main       # http://127.0.0.1:7440/ui
```

這條路不會有 ollama，`/api/vision/solve` 需要另外自己跑一個：`docker compose -f docker-compose.ollama.yml up -d`
（`.env` 的 `OLLAMA_PROVIDER_URL` 預設指向 `http://127.0.0.1:11435/v1`，也就是它對外開的那個埠）。
