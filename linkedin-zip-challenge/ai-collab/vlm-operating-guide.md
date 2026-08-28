# 操作手冊 — 用微調後的模型讀圖解題

> 2026-08-29（Asia/Taipei）｜對象：**使用這個專案的人**（不是接手開發的 agent）
> 接手開發看 [`handover-vlm-parser.md`](handover-vlm-parser.md)；訓練細節看
> [`reports/2026-08-29_vl-training-reproducible.md`](reports/2026-08-29_vl-training-reproducible.md)。

---

## 0. 這份文件教你什麼

| 你想做的事 | 去 §幾 |
|---|---|
| **每天開機後要打哪幾個指令** | §1（三行） |
| 用網頁介面上傳截圖解題 | §2 |
| 用 API 解題（給程式呼叫） | §3 |
| **手把手自己測一次，含現成測資** | §3.5 ★ |
| 看懂回應裡的警告與信心旗標 | §4 ★ **最重要** |
| 換模型（微調版 ↔ 未微調版） | §5 |
| 重新產生模型（重訓之後） | §6 |
| **每次呼叫的紀錄存在哪／怎麼變成評估集** | §6.5 ★ |
| 出事了怎麼查 | §7 |

**前提**：Docker Desktop 已安裝並啟動、專案已 `uv sync`。

---

## 1. 每天開機後的三行

> **兩種跑法，先講清楚差別**：
>
> | 跑法 | 誰在容器裡 | 狀態 |
> |---|---|---|
> | **A. Ollama 在 Docker、app 在本機**（下面這三行） | 只有模型 | ✅ **2026-08-29 實測驗過，讀圖功能可用** |
> | B. 整套 Docker（`python run_docker_dev.py`） | 模型＋app＋Svelte | ⚠ **設定已就緒但未實測**：這台機器**從沒 build 過 app 的 image**，第一次要拉 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel`（約 20 GB）再 `uv sync`。網路設定已驗過（compose 會把 app 的 `OLLAMA_PROVIDER_URL` 覆寫成 `http://ollama:11434/v1`），但整條路徑沒跑過 |
>
> **A 就是 `AGENTS.md` 記載的開發跑法，建議用它。**

```powershell
cd D:\it_project\github_sync\zip-vlm\linkedin-zip-challenge
docker compose -f docker-compose.dev.yml up -d ollama
uv run python -m src.app.main
```

然後開瀏覽器：

| 網址 | 是什麼 |
|---|---|
| <http://localhost:7440/ui> | Gradio 主控台（**讀圖分頁在這裡**） |
| <http://localhost:7440/docs> | Swagger，可以直接試 API |
| <http://localhost:7440/svelte-ui> | Svelte 互動編輯器 |

> ⚠ **Docker Desktop 有開 ≠ 容器有跑。** `zip_ollama_server` 會停在 `Exited`，
> 第二行就是把它叫起來。確認用 `docker ps`，要看到 `zip_ollama_server` 是 `Up`。
> ⚠ **同一個埠不要重複起 app。** Windows 上 7440 可以同時有多個 LISTENING，
> 回應的是最舊的那個 process，症狀是「改了程式卻沒反應」。先 `netstat -ano | findstr :7440` 確認。

第一次呼叫模型會慢（要把 9.3 GB 載進顯卡，約 50 秒），之後每張圖 **3–7 秒**。

---

## 2. 網頁介面：上傳截圖解題

1. 開 <http://localhost:7440/ui>
2. 切到 **`Solve from Screenshot`** 分頁
3. 拖一張 Zip 謎題截圖進去（或按貼上）
4. Solver 選 **CP-SAT**（預設；它判定「無解」最快）
5. 想看動畫再勾 **Also render the animation**（會慢好幾秒）
6. 按 **Read and Solve**

右邊會出現四塊：

| 區塊 | 內容 |
|---|---|
| 摘要 | 哪個模型讀的、幾乘幾、**有沒有解**、**警告** |
| 解答圖 | 走法（勾了動畫才有 GIF） |
| `Layout read` | 模型讀到的盤面，**Python literal**，可直接貼到 `Puzzle Solver (Naive)` 分頁 |
| `Walls read` | 模型讀到的牆，同樣可貼 |

**後兩塊是為了「模型讀錯時你能自己修」而存在的**：把它們貼進 Naive 或 Interactive 分頁，
改掉錯的那一格或那一道牆，再解一次。

---

## 3. API：`POST /api/vision/solve`

`multipart/form-data`：

| 欄位 | 必填 | 說明 |
|---|---|---|
| `image` | ✅ | 圖片檔。支援 `.png` `.jpg` `.jpeg` `.webp` |
| `solver_name` | | `CP-SAT`（預設）／`DFS`／`A* (heapq)` |
| `include_gif` | | `true` 才產生動畫，預設 `false` |

```powershell
curl.exe -X POST http://localhost:7440/api/vision/solve `
  -F "image=@illustrations/puzzle_01.png" `
  -F "solver_name=CP-SAT"
```

回應：

```json
{
  "model_name": "zip-qwen35-4b-p4c:f16",
  "prompt_variant": "finetune",
  "grid_size": [6, 6],
  "layout": [["  ", "01", ...], ...],
  "walls": [{"cell1": [0, 4], "cell2": [1, 4]}, ...],
  "warnings": [],
  "solvable": true,
  "solver_name": "CP-SAT",
  "solution_path": "(0, 0) -> (0, 1) -> ...",
  "solution_final_image_b64": "iVBORw0KG...",
  "solution_gif_b64": null
}
```

### HTTP 狀態碼的意思不同，不要一律當「失敗」

| 碼 | 意思 | 你該做什麼 |
|---|---|---|
| **200** | 讀到了（**不代表讀對**，看 §4） | 看 `solvable` 與 `warnings` |
| **422** | 模型有回答，但不是可用的盤面 | 通常是圖裡沒有 Zip 謎題。**重試同一張沒用** |
| **503** | 模型連不上 | 回 §1 把 `zip_ollama_server` 叫起來 |
| **415** | 副檔名不支援 | 轉成 PNG／JPG |
| **404** | solver 名字打錯 | 只有那三種 |

---

## 3.5 ★ 手把手自己測一次（含現成測資）

### 步驟 0：確認服務都在

> ⚠ **路徑要對**：這條 track 的東西全在 **`zip-vlm`** 這個 worktree，不是主工作樹 `ml-workshop`。

```powershell
cd D:\it_project\github_sync\zip-vlm\linkedin-zip-challenge
docker ps                                    # 要看到 zip_ollama_server 是 Up
docker exec zip_ollama_server ollama list    # 要看到 zip-qwen35-4b-p4c:f16
uv run python -m src.app.main                # 另開一個視窗讓它一直跑
```

### 步驟 1：最簡單的方式——用 Swagger，不用打指令

1. 開 <http://localhost:7440/docs>
2. 找 **`POST /api/vision/solve`** → 按 **Try it out**
3. `image` 選一張圖（測資見下）→ 按 **Execute**
4. 看 Response body

### 步驟 2：用指令測一張

```powershell
curl.exe -X POST http://localhost:7440/api/vision/solve `
  -F "image=@illustrations/puzzle_01.png" `
  -F "solver_name=CP-SAT"
```

### 步驟 3：一次測完六張真實截圖（**這段實跑過**）

```powershell
Get-ChildItem .\illustrations\puzzle_0*.png | ForEach-Object {
  $json = curl.exe -s -X POST http://localhost:7440/api/vision/solve -F "image=@$($_.FullName)" -F "solver_name=CP-SAT" | ConvertFrom-Json
  [pscustomobject]@{
    圖片 = $_.Name
    盤面 = "$($json.grid_size[0])x$($json.grid_size[1])"
    牆   = $json.walls.Count
    有解 = $json.solvable
    警告 = $json.warnings.Count
  }
} | Format-Table -AutoSize
```

> ⚠ **不要用 `Invoke-RestMethod -Form`**——`-Form` 是 PowerShell 6.1 才有的，
> 這台是 Windows PowerShell 5.1，會直接參數錯誤。用 `curl.exe`（Win10 內建）。

### 測資 A：六張真實 LinkedIn 截圖（已在版控，`illustrations/`）

**這是預期輸出，對得起來就代表整條路是通的**（2026-08-29 實測）：

```
圖片            盤面   牆    有解 警告
puzzle_01.png 6x6 10  True  0
puzzle_02.png 6x6  0  True  0
puzzle_03.png 6x6  5 False  1     <- 故意留著的失敗案例，見 §4
puzzle_04.png 7x7 14  True  0
puzzle_05.png 6x6  4  True  0
puzzle_06.png 7x7  0  True  0
```

**`puzzle_03` 回 `有解 False` 是正確行為，不是壞掉**——模型多讀了一道牆，
安全機制把它抓出來了。這張刻意留著當「警告長什麼樣子」的示範。

### 測資 B：合成 held-out（本機 `datasets/vl/`，不進版控）

> ⚠ **找不到這個資料夾？多半是看錯 worktree。** `datasets/` 不進版控，所以它**只存在於當初建它的那個 worktree**：
>
> | Worktree | `datasets/vl/main_6x6/images/` |
> |---|---|
> | `D:\it_project\github_sync\zip-vlm\linkedin-zip-challenge` | ✅ 8,000 個檔案 |
> | `D:\it_project\github_sync\ml-workshop\linkedin-zip-challenge` | ⚠ `datasets/` 存在但是空的 |
>
> 這次所有的產出（`datasets/`、`models/`、`logs/`）都在 **`zip-vlm`** 這一邊。

**`main_6x6` 共 8,000 張，前 7,800 張是訓練資料、最後 200 張（`007800`–`007999`）是模型沒看過的 held-out。**
只測 held-out，測前面等於在測訓練資料。

四張橫跨牆數範圍的樣本，**四張都應該完全正確**：

| 編號 | 牆數 | 格子大小 | 存成 |
|---|---|---|---|
| `007811` | 0 | 116 | JPEG (q77) |
| `007817` | 4 | 100 | JPEG (q88) |
| `007841` | 8 | 72 | JPEG (q94) |
| `007807` | 12 | 72 | JPEG (q92) |

> ⚠ **副檔名是混的。** 出資料時有 35% 的機率存成 JPEG，其餘是 PNG，所以
> `007811` 是 `.jpg` 但 `007800` 是 `.png`。下面的指令用萬用字元，不寫死副檔名。

```powershell
"007811","007817","007841","007807" | ForEach-Object {
  $img  = (Get-ChildItem "datasetsl\main_6x6\images\$_.*").FullName
  $json = curl.exe -s -X POST http://localhost:7440/api/vision/solve -F "image=@$img" | ConvertFrom-Json
  [pscustomobject]@{ 檔案 = Split-Path $img -Leaf; 牆 = $json.walls.Count; 有解 = $json.solvable }
} | Format-Table -AutoSize
```

期望：牆數分別是 **0 / 4 / 8 / 12**，`有解` 全部 `True`。

**想知道某張圖的正確答案**（`label` 就是模型該吐的那個 JSON）：

```powershell
uv run python -c "import json; print(next(r['label'] for r in map(json.loads, open('datasets/vl/main_6x6/metadata.jsonl', encoding='utf-8')) if r['file_name'].startswith('images/007811')))"
```

### 測資 C：自己出一張新的（隨時要多少有多少）

```powershell
uv run python -m src.core.vl_models.dataset_builder --count 3 --name mytest --no-verify
# 圖會出現在 datasets/vl/mytest/images/，標準答案在 datasets/vl/mytest/metadata.jsonl
```

### 想量完整指標而不只是看一眼

```powershell
# 跑 20 筆 held-out 並算出四層指標
uv run python -m src.core.vl_models.run_holdout `
  --dataset datasets/vl/main_6x6 --count 20 --slice tail --prompt finetune `
  --out ai-collab/reports/artifacts/vl-p4d/mycheck.jsonl
uv run python -m src.core.vl_models.score_predictions ai-collab/reports/artifacts/vl-p4d/mycheck.jsonl
```

### 測不出來的時候

| 症狀 | 原因 |
|---|---|
| `curl` 回 `Failed to connect` | app 沒起來，回步驟 0 |
| 回 503 | `zip_ollama_server` 沒跑 |
| 第一張特別慢（快一分鐘） | 正常，在載 9.3 GB 進顯卡 |
| PowerShell 說 `-Form` 不是有效參數 | 見上面的警告，改用 `curl.exe` |

---

## 4. ★ 看懂 `solvable` 與 `warnings`（最重要的一節）

### 4.1 `solvable: false` ＝ **一定讀錯了**，不是「這題很難」

出題器是**先畫一條走完全部格子的路，再把題目挖出來**——所以**真實盤面必定有解**。
既然如此，「讀出來的盤面無解」就**證明**至少有一道牆或一個號碼讀錯了，
**而且不需要正確答案就能知道**。這是免費的自我檢查，所以做成回應的一部分。

看到 `solvable: false`：把 `layout` 與 `walls` 貼進 Interactive 分頁，自己對著截圖修。

### 4.2 `warnings` 非空 ＝ 模型幻覺出不合法的牆，已經被丟掉

例如「牆蓋在不相鄰的兩格之間」或「牆跑到格子外面」。parser 會丟掉它們並**告訴你丟了什麼**——
不告訴你才是危險的。

### 4.3 ★ 真正危險的是「看起來很正常」的那種錯

牆的兩種錯誤**後果完全不對稱**：

| 錯誤 | 後果 |
|---|---|
| **多幻覺一道牆** | 解出來的路**仍然合法**（只是被多綁了一下）；最糟是變成「無解」——**你看得見** |
| **漏讀一道牆** | 解出來的路**可能穿牆，而且畫面上一切正常**——**靜默的錯誤答案** |

⇒ **`solvable: true` ＋ `warnings: []` 不等於一定對。**
重要場合請對著原截圖掃一遍牆。

---

## 5. 換模型

模型與 prompt **必須配對**——拿微調模型去吃 baseline 的 few-shot prompt，
等於問它一個訓練時從沒看過的問題。所以兩個設定放在一起：

編輯 `.env`（**不進版控**）：

```ini
# 微調版（預設）— 合成 held-out 端到端 200/200
OLLAMA_MODEL_NAME=zip-qwen35-4b-p4c:f16
VISION_PROMPT_VARIANT=finetune

# 未微調版 — 2026-08-15 量到的最佳基線設定，牆 F1 約 0.44
# OLLAMA_MODEL_NAME=qwen3.5:4b-q8_0
# VISION_PROMPT_VARIANT=sized
```

改完**要重開 app**（設定有 `@cache`，不會熱重載）。

> 兩者的差距很大：未微調在真實截圖上端到端只有 **2/6**，微調版在合成 held-out 上是 **200/200**。
> ⚠ 但這兩個數字**不可直接比較**（不同的圖、不同的難度）——見 §8。

---

## 6. 重新產生模型（重訓之後才需要）

模型不是憑空來的：Colab 訓練只產出一個 **LoRA adapter**，要經過三步才變成 Ollama 跑得動的東西。
**這三步全部在本機做，不用回 Colab。**

### 前置：把東西放到位

```
models/
├─ base/Qwen3.5-4B/                        # 8.9 GB，見下方指令
└─ colab_finetune/p4c_qwen35_4b_zip_lora/  # 從 Google Drive 下載並解壓
```

> ⚠ Google Drive 下載大目錄時會**切成多個 zip**，而且會把大檔和小檔拆到不同包。
> **每一包都要解到同一個目錄樹**，否則你會得到一個沒有 `adapter_model.safetensors` 的資料夾。

下載 base（約 8.9 GB，只需一次）：

```powershell
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('unsloth/Qwen3.5-4B', local_dir='models/base/Qwen3.5-4B', ignore_patterns=['LICENSE','README.md','.gitattributes'])"
```

### 第 1 步：把 adapter 併回 base

```powershell
uv run python -m src.core.vl_models.merge_lora `
  --base models/base/Qwen3.5-4B `
  --adapter models/colab_finetune/p4c_qwen35_4b_zip_lora `
  --out models/merged/qwen35-4b-zip-p4c
```

約 15 秒。它會印出併了幾個張量（P4c 是 **344** 個），對不上就中止。

### 第 2 步：轉成 GGUF（文字塔 ＋ 視覺投影層兩個檔）

需要一份 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的原始碼（不用編譯，只用它的 Python 轉檔腳本）：

```powershell
# 只需 clone 一次，放哪裡都行
git clone --depth 1 https://github.com/ggml-org/llama.cpp C:\tools\llama.cpp

$env:PYTHONPATH = "C:\tools\llama.cpp\gguf-py"
uv run python C:\tools\llama.cpp\convert_hf_to_gguf.py models/merged/qwen35-4b-zip-p4c `
  --outfile models/gguf/zip-qwen35-4b-p4c-text-f16.gguf --outtype f16
uv run python C:\tools\llama.cpp\convert_hf_to_gguf.py models/merged/qwen35-4b-zip-p4c `
  --outfile models/gguf/zip-qwen35-4b-p4c-mmproj-f16.gguf --outtype f16 --mmproj
```

**兩個檔都要**：第二個是視覺投影層，少了它模型看不見圖。

### 第 3 步：匯進 Ollama

`docker-compose.dev.yml` 已經把 `./models` 唯讀掛到容器的 `/models`，所以不必複製 9 GB 進容器。

```powershell
docker exec zip_ollama_server sh -c "printf 'FROM /models/gguf/zip-qwen35-4b-p4c-text-f16.gguf\nFROM /models/gguf/zip-qwen35-4b-p4c-mmproj-f16.gguf\n' > /tmp/Modelfile && ollama create zip-qwen35-4b-p4c:f16 -f /tmp/Modelfile"
docker exec zip_ollama_server ollama show zip-qwen35-4b-p4c:f16
```

`ollama show` 的 **Capabilities 要有 `vision`**、下面要有 **Projector** 區塊。沒有就是第 2 步只轉了一個檔。

### 第 4 步：驗證它真的沒壞

```powershell
# 對 held-out 跑 20 筆（完整 200 筆約 15 分鐘）
uv run python -m src.core.vl_models.run_holdout `
  --dataset datasets/vl/main_6x6 --count 20 --slice tail --prompt finetune `
  --out ai-collab/reports/artifacts/vl-p4d/check.jsonl

uv run python -m src.core.vl_models.score_predictions ai-collab/reports/artifacts/vl-p4d/check.jsonl
```

`exact_match` 要接近 1.00。明顯掉下來就是匯出壞了，不要拿去用。

> ⚠ **不要用 `ollama create --experimental` 直接吃 safetensors 目錄。** 實測（2026-08-29，
> Ollama 0.32.13）它匯入會成功，但執行時走 **MLX runner**（Apple Silicon 專用），
> 在 Linux／NVIDIA 容器裡直接報
> `mlx runner failed: MLX not available`。**一定要走 GGUF。**

---

## 6.5 ★ 每次呼叫都會留紀錄，而且可以直接變成評估集

`POST /api/vision/solve` 每被呼叫一次，就會把**上傳的圖**和**模型原話**存進：

```
logs/vision/
├─ images/20260829-044717_949e2bfc.png
└─ metadata.jsonl
```

（`logs/` 已在 `.gitignore`，不會進版控。要關掉就把 `.env` 的 `VISION_LOG_DIR` 設成空字串。）

一行長這樣：

```json
{"file_name": "images/20260829-044717_949e2bfc.png",
 "logged_at": "2026-08-29T04:47:17+08:00",
 "image_sha256": "949e2bfc...",
 "model_name": "zip-qwen35-4b-p4c:f16", "prompt_variant": "finetune",
 "usable": true, "generation_seconds": 14.933,
 "grid_size": [6, 6], "wall_count": 10, "solvable": true, "parser_warnings": [],
 "raw_output": "{
  \"layout\": [...]}"}
```

**讀不出來的那種（HTTP 422）也會記**，`usable` 是 `false` 並附 `parse_error`——
那是最值得留的一種，因為任何評估集裡都沒有、重試也沒用。

### ★ 這個資料夾的形狀是刻意的：補一個 `label` 就能算分

欄位名跟訓練資料完全一致（`file_name`／`raw_output`／`generation_seconds`），
唯一缺的是 `label`——那是「這張圖的正確答案」，只有人看得出來。
所以你**手動幫某幾行補上 `label`**，現成的 scorer 就能直接算：

```powershell
uv run python -m src.core.vl_models.score_predictions logsision\metadata.jsonl
```

`label` 的格式就是模型該吐的那個 JSON 字串。最省事的做法是：
把模型讀出來的貼進 Gradio 的 Interactive 分頁 → 把錯的地方改對 → 拿改好的 JSON 當 `label`。

**這是目前累積「真實截圖評估集」最便宜的路**：你平常用它解謎題，順手改幾張錯的，
評估集就長出來了。（實測過：兩行補上 `label` 之後 scorer 正確算出 `EXACT MATCH 1/2`、
牆 F1 0.889 的那張就是 `puzzle_03`。）

---

## 7. 出事了怎麼查

| 症狀 | 多半是什麼 | 怎麼修 |
|---|---|---|
| API 回 **503** | `zip_ollama_server` 沒跑 | `docker compose -f docker-compose.dev.yml up -d ollama` |
| API 回 **503**，但容器有跑 | `.env` 的 URL 不對 | host 上跑 app 要 `http://127.0.0.1:11435/v1`；容器裡跑則由 compose 覆寫，不用改 |
| 第一次呼叫等了快一分鐘 | 正常，在載 9.3 GB 進顯卡 | 第二次起 3–7 秒 |
| 每次都很慢、答案還很爛 | 模型 tag 是未微調的那個 | `docker exec zip_ollama_server ollama list` 對一下，再看 §5 |
| 回答全是一大段推理文字 | 思考沒關掉 | 出貨路徑已經固定關閉。若你自己呼叫 Ollama，`/v1` **不吃 `think`**，要 `reasoning_effort="none"` |
| 改了 `.env` 沒反應 | 設定有 `@cache` | 重開 app |
| 改了程式沒反應 | 7440 上有多個 process | `netstat -ano \| findstr :7440` |
| `docker compose up` 說埠被佔 | 另一個 worktree 的 stack 開著 | **兩個 worktree 不可同時起**，先關掉另一個 |
| Gradio 分頁不見 | app 是舊版本 | 重開 app |

---

## 8. 這個模型能做到什麼、不能做到什麼

**能**：讀懂**這個專案自己畫出來的**盤面，在 200 筆沒看過的合成圖上端到端 **200/200**（全部指標 1.000）。

**真實 LinkedIn 截圖**：目前有六張的實測，結果比預期好——逐格 **1.000**、號碼召回 **1.000**、
牆 F1 **0.972**、端到端 **5/6**（未微調同一批是 0.438／2/6）。**兩張 7×7 也全對**，
即使訓練資料 100% 是 6×6。唯一失敗的那張是多幻覺了一道牆而變成無解，**被 `solvable` 旗標抓到**。

**但六張不算驗證過。** 訓練與主要評估都在自己的 renderer 上，這六張又是開發期間反覆看的同一批。
所以：

- ✅ 「合成資料訓出來會在真實截圖上失效」這個風險，**在現有證據上沒有出現**
- ❌ 但**不能**因此宣稱「已驗證可讀 LinkedIn 截圖」

**所以**：拿真實截圖用的時候，§4 那三條請當成必讀，尤其是 4.3 的「漏讀牆是靜默錯誤」。
六張裡靜默錯誤是 0 筆，但六張證明不了「不會發生」。`solvable` 旗標會抓到一部分，但抓不到全部。
