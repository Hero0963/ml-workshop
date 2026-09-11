# 部署指南 — 用 Docker 起整個服務

> 2026-09-12 實測寫成（分支 `feat/rl-a2-training`）。每一條指令與輸出都跑過，沒跑過的會標「未驗」。
> 想知道模型怎麼推論 → [`notes/03-inference-and-serving.md`](notes/03-inference-and-serving.md)；
> 架構與模組職責 → [`project_guide.md`](project_guide.md)。

---

## 1. 一句話

```powershell
cd linkedin-zip-challenge
docker compose up -d --build
```

兩個容器起來（約 2 分鐘，第一次建 image 要 10–15 分鐘），然後開 <http://127.0.0.1:7440/ui>。

---

## 2. 這個 stack 裡有什麼

| 容器 | 是什麼 | 埠 | 要 GPU 嗎 |
|---|---|---|---|
| `zip_challenge_prod_app` | FastAPI ＋ Gradio ＋ 已建好的 Svelte 編輯器 ＋ **4 種 solver（含 RL）** | `7440` | ❌ 不用 |
| `zip_ollama_server` | 服務微調過的視覺語言模型，`/api/vision/solve` 靠它讀截圖 | `11435`（對外）→ `11434`（容器內）| ✅ 要 |

**app 容器不要 GPU 是刻意的**：RL 推論走 CPU 就夠（一題 4×4 約 40 毫秒），GPU 留給 ollama。
所以 app 的 log 會印 `WARNING: The NVIDIA Driver was not detected` ——**這是正常的，不用追。**

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
| Svelte | 建置階段編好，走 `/svelte-ui` | 另開一個容器跑 vite dev server（`5173`）|
| 啟動 | `docker compose up -d --build` | `docker compose -f docker-compose.dev.yml up -d --build`（或 `python start.py --dev`）|

⚠ **兩份是二選一，不是疊加**：容器名稱相同、共用 `ollama_data` volume。換一份前先 `docker compose down`。

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

## 5. 驗收：這五項都實際跑過（2026-09-12）

```
health       (200, b'{"status":"ok"}')
gradio /ui   200
svelte       200
openapi      200
DFS                      (200, '(0, 0) -> (0, 1) -> (0, 2) -> (0, 3) -> (1, 3) -> ...')
A* (heapq)               (200, '(0, 0) -> (0, 1) -> (0, 2) -> (0, 3) -> (1, 3) -> ...')
CP-SAT                   (200, '(0, 0) -> (0, 1) -> (0, 2) -> (0, 3) -> (1, 3) -> ...')
RL (behaviour cloning)   (200, '(0, 0) -> (0, 1) -> (0, 2) -> (0, 3) -> (1, 3) -> ...')
7x7 via RL   (400, '{"detail":"No RL policy for a 7x7 board; trained sizes are [4, 5, 6]."}')
```

app → ollama 的連線也實測過：

```
app -> ollama OK, models: ['zip-qwen35-4b-p4c:f16', 'gemma4:e4b-it-q8_0', 'qwen3.5:4b-q8_0',
                           'openbmb/minicpm-o2.6:latest', 'qwen2.5vl:7b', ...]
```

容器狀態：

```
zip_challenge_prod_app  Up (healthy)
zip_ollama_server       Up (healthy)
```

---

## 6. 這次修掉的四個缺陷（都是「看起來有起來、其實沒有」那種）

| 缺陷 | 症狀 | 修法 |
|---|---|---|
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

**Q：image 有 22.9 GB。**
基底是 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`（約 10 GB，含 CUDA toolkit），
`uv sync` 之後又裝了一份 `torch 2.4.1+cu121`。**app 容器其實用不到 CUDA**（GPU 在 ollama 那邊），
所以換成 slim 基底可以大幅縮小——**尚未做，也尚未驗證相依的系統函式庫夠不夠**。

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
