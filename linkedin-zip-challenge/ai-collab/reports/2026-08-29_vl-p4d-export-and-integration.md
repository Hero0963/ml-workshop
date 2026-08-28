# P4d／P5／P6 — 把微調模型接進產品，以及它在真實截圖上的表現

> 2026-08-29（Asia/Taipei）｜分支 `feat/vlm-parser`｜worktree `zip-vlm`
> 一句話：**匯出零損失（200/200 逐位元組與 Colab 相同、快 6.5 倍），而且在六張真實 LinkedIn 截圖上端到端 5/6——比原本預期的好很多。**
> 原始資料：[`artifacts/vl-p4d/`](artifacts/vl-p4d/)｜訓練細節：[`2026-08-29_vl-training-reproducible.md`](2026-08-29_vl-training-reproducible.md)
> 操作方式：[`../vlm-operating-guide.md`](../vlm-operating-guide.md)

---

## 1. 做了什麼

P4c 的成果原本只是 Google Drive 上的一個 LoRA adapter，產品拿不到。這一輪把它變成本機服務中的模型，並接上 API 與 UI：

```
Drive 的 LoRA adapter
   ↓ 撈回本機 + SHA-256 驗證
   ↓ merge_lora.py            併回 unsloth/Qwen3.5-4B（344 個張量）
   ↓ convert_hf_to_gguf.py    文字塔 8.42 GB + mmproj 672 MB
   ↓ ollama create            zip-qwen35-4b-p4c:f16
   ↓ puzzle_parser → /api/vision/solve → Gradio「Solve from Screenshot」分頁
```

---

## 2. 撈回來的東西是不是原來那一份

adapter 從 Drive 下載時被切成兩個 zip，**大檔與小檔被拆到不同包**（`adapter_model.safetensors` 在 002，設定檔在 001），
兩包必須解到同一個目錄樹。驗證：

| 檢查 | 結果 |
|---|---|
| 訓練資料 tar SHA-256 | **MATCH** `69c753e1…0fbf` |
| 200 筆預測 SHA-256 | **MATCH** `fc96cb30…4e22` |
| adapter 張量數 | 688（**visual 96 對 / language 248 對**，全 F32） |

最後一列是關鍵：96／248 與 P4c 訓練當下 notebook 印出的 `visual: 96/96`、`language: 248/248` **完全吻合**，
所以撈回來的是那一次 run 的 adapter，不是別的。

> ⚠ **checkpoint 只剩 800 與 975。** 訓練時設了 `save_total_limit=2`，200／400／600 早被自動刪除。
> 交接文件 §6 提議的「拿 checkpoint-200 驗證 1,600 筆是否就夠」**已經做不成**。

---

## 3. Merge：為什麼自己寫而不用 peft

adapter 的 base 是 `Qwen3.5-4B`，它的 modelling 程式碼在 `transformers` 5.x；本專案為了其他模組鎖 `transformers<5`。
為了兩個矩陣相加而升級整個堆疊不划算，而**這個運算不需要它**——直接讀寫張量即可。

`src/core/vl_models/merge_lora.py` 只實作最單純那一種，並且**拒絕**其他變體而不是算錯：
`use_dora`／`use_rslora`／`fan_in_fan_out`／`lora_bias` 皆為 false、`modules_to_save` 與
`rank_pattern`／`alpha_pattern` 皆空 ⇒ 更新就是 `W += (alpha/r) · B @ A`，scaling = 16/16 = **1.0**。
delta 在 float32 累加後再轉回 base 的 bfloat16（adapter 存的是 float32，直接用 bf16 乘會丟掉大部分訓練成果）。

**實測**：344 個張量、**14 秒**。逐張量稽核：

```
visual    merged  96/96   max|delta| 2.2400e-02
language  merged 248/248  max|delta| 6.0272e-03
non-target tensors changed: 0
```

視覺層動得比語言層大——與訓練時量到的 `max|B|` 0.272 > 0.166 **方向一致**，兩種完全不同的量法互相佐證。

### 3.1 一個會靜默弄壞模型的陷阱

adapter 目錄裡有一份 `tokenizer_config.json`，是 Colab 上 transformers **5.2.0** 重新序列化的，
`tokenizer_class` 寫著 `TokenizersBackend`——**transformers 4.x 的任何工具都載不動**，GGUF 轉檔會直接失敗。

直覺的規則「adapter 的副本優先，因為那是訓練時真正用的」在這裡是**錯的**。實測比對後發現：

| 檔案 | base vs adapter |
|---|---|
| `chat_template.jinja` | **逐位元組相同** |
| `tokenizer.json` | **逐位元組相同** |
| `processor_config.json` | **逐位元組相同** |
| `tokenizer_config.json` | 只差在 transformers 版本的序列化風格 |

也就是說**沒有任何訓練相關的東西需要保留**。所以 `merge_lora.py` 改成：用 base 的副本，
但**逐位元組驗證上面三個「會改變模型行為」的檔案一致，不一致就中止 merge**。
那正是交接文件 §7 吃過兩次虧的 train/inference 渲染陷阱——應該擋下來讓人決定，而不是用經驗法則挑一邊。

---

## 4. 匯出：safetensors 路線是死路，一定要走 GGUF

**先試最便宜的兩條，都失敗，記錄在這裡以免有人再試一次：**

| 嘗試 | 結果 |
|---|---|
| `ollama create` 用 `ADAPTER` 掛 safetensors adapter 在 `qwen3.5:4b-q8_0` 上 | ❌ `qwen3.5:4b-q8_0 is not a supported safetensors model directory (needs config.json + *.safetensors)`——base 也必須是 safetensors 目錄 |
| `ollama create --experimental` 直接吃 merge 後的 safetensors 目錄 | ⚠ **匯入會成功**（738 張量、747 層，`ollama show` 甚至顯示 `vision`），**但一執行就死**：`mlx runner failed: MLX not available: failed to load MLX dynamic library`。unquantized safetensors 走的是 **MLX runner（Apple Silicon 專用）**，Linux/NVIDIA 容器裡沒有那個函式庫。`--quantize` 支援的型別是 `int4/int8/nvfp4/mxfp4/mxfp8`，也是 MLX 那一套。 |

⇒ **必須用 llama.cpp 轉 GGUF。** 該版本（2026-04）已經註冊 `Qwen3_5ForConditionalGeneration`
（文字塔 `Qwen3_5TextModel`、視覺 `Qwen3VLVisionModel`），轉檔順利：

| 產物 | 大小 | 張量 |
|---|---|---|
| `zip-qwen35-4b-p4c-text-f16.gguf` | 8.42 GB | 426 |
| `zip-qwen35-4b-p4c-mmproj-f16.gguf` | 672 MB | 298 |

**兩個檔都要**，Modelfile 兩行 `FROM`。少了 mmproj 模型就看不見圖。
匯入後 `ollama show` 有 `vision` capability 與 `Projector`（clip、333.51M 參數）。

> 順帶一提：**unsloth#3899（vision GGUF 匯出缺陷）沒有踩到**——但那是因為這裡沒有用 unsloth 的匯出，
> 而是自己 merge 後用 llama.cpp 轉。這條路徑繞開了那個缺陷。

---

## 5. 匯出有沒有損失？沒有，一位元都沒有

拿同一批 200 筆 held-out，把**本機 GGUF 的輸出**與**Colab 上 LoRA 的輸出**逐筆對拍：

```
same files                200
local  output == label  : 200/200
colab  output == label  : 200/200
local  output == colab  : 200/200      ← 逐位元組相同
mean seconds  colab 34.5   local 5.3   speedup 6.5x
distinct local timings: 197            ← 不是常數，確實在生成
```

離線算分（走出貨用的 parser 與既有指標函式，不重寫）：

```
JSON parse rate          1.000
grid size correct        200/200
mean cell accuracy       1.000
mean waypoint recall     1.000
mean wall F1 (walled)    1.000  over 189 boards
micro wall precision     1.000
micro wall recall        1.000
EXACT MATCH              200/200  (1.000)
SOLUTION VALID           200/200  (1.000)
predicted board solvable 200   (of which wrong: 0  <- silent failures)
```

十三個牆數桶每一個都是 1.00。**P4c 完整重現。**

**6.5 倍加速的來源**正是 P4c 報告 §5.3 指出的設計錯誤被修掉了：那邊是 batch 1 逐筆生成、
344 個**未 merge** 的 LoRA adapter 讓每個 token 多 688 次 kernel 發動；這裡權重已經併進去，
而且走 llama.cpp 的推論路徑。

---

## 6. ★ 真實 LinkedIn 截圖：這一節是新資訊

到目前為止所有的話都是「它學會了我們的 renderer」。這次順手把六張**真實截圖**也跑了一遍
（用 `finetune` prompt，**沒有 few-shot 範例，所以不存在 `puzzle_01–03` 的洩題問題**）：

| 指標 | 未微調最佳設定 | **微調後** |
|---|---|---|
| JSON 可解析 | 6/6 | 6/6 |
| 格盤尺寸 | 6/6 | 6/6 |
| 逐格準確率 | 0.947 | **1.000** |
| 號碼召回 | 0.917 | **1.000** |
| 牆 F1（有牆題） | **0.438** | **0.972** |
| **端到端相符** | **2/6** | **5/6** |
| 平均延遲 | 5.4 s | 5.6 s |
| 峰值 GPU | — | 11,986 MiB（16 GB 卡剩餘充足） |

逐張：

| 圖 | 尺寸 | 逐格 | 牆 | 端到端 |
|---|---|---|---|---|
| puzzle_01 | 6×6 | 1.000 | 10/10 | ✅ |
| puzzle_02 | 6×6 | 1.000 | 0/0 | ✅ |
| **puzzle_03** | 6×6 | 1.000 | **預測 5、真實 4**（precision 0.80、recall 1.00） | ❌ 無解 |
| **puzzle_04** | **7×7** | 1.000 | **14/14** | ✅ |
| puzzle_05 | 6×6 | 1.000 | 4/4 | ✅ |
| **puzzle_06** | **7×7**（21 個號碼） | 1.000 | 0/0 | ✅ |

### 6.1 唯一的失敗是「看得見的」那一種

`puzzle_03` 的 recall 是 **1.00**——四道真牆一道都沒漏，只是**多幻覺了一道**。
後果正如 P4c 報告 §7 的推論：多牆讓盤面過度受限 ⇒ **無解** ⇒ `solvable: false` 直接抓到。

**六張裡靜默錯誤（讀錯但仍解得出來、且沒人會發現）是 0 筆。** 這是這一節最重要的一句話。

### 6.2 ★ 兩張 7×7 全對，推翻了一個寫在文件裡的預期

交接文件 §0 ⑤ 寫著：「只做 6×6……訓練資料 100% 是 6×6，微調後模型很可能看到 7×7 也答 6×6。」

**實測沒有發生。** `puzzle_04`（7×7、14 道牆）與 `puzzle_06`（7×7、21 個號碼）都是尺寸正確、逐格 1.000。
未微調時 `gemma4:e4b` 反而會把真 6×6 過度矯正成 7×7，而這個微調模型兩種尺寸都判對。

### 6.3 這個數字能撐多重的結論？——不要拿它當「已驗證真實截圖」

**n = 6，其中只有 4 張有牆。** 而且這六張從 P0 起就一直是開發時看的那批，
存在「反覆檢視同一組樣本」的選擇偏誤。**它們從未進入訓練資料**，所以不是洩題，
但**六張不足以宣稱「看得懂 LinkedIn 截圖」**。

誠實的說法是：**先前「合成資料訓出來的模型可能在真實截圖上失效」這個風險，
在現有的六張證據上沒有出現**，而且牆 F1 從 0.438 跳到 0.972。
要把它變成可靠結論仍然需要更多真實樣本——那是本人 2026-08-22 定案不做的 P3。
**這一節不是推翻那個決定，是把新證據放上檯面讓決定可以被重新考慮。**

---

## 7. 產品接線（P5／P6）

### `POST /api/vision/solve`

`multipart/form-data`：`image`（必填）、`solver_name`（預設 CP-SAT）、`include_gif`（預設 false）。

回應含解析出的 `layout`／`walls`、`warnings`、**`solvable` 信心旗標**、solver 路徑與解答圖。
狀態碼刻意分開，因為對呼叫端意義不同：

| 碼 | 意思 |
|---|---|
| 422 | 模型有回答但不可用（多半圖裡沒有謎題）——重試同一張沒用 |
| 503 | 模型連不上——可以稍後重試 |
| 415 | 副檔名不支援（傳輸層靠副檔名決定 media type） |

### Gradio `Solve from Screenshot` 分頁

Adapter，只把 UI 操作翻成 API 呼叫。輸出四塊：摘要（含警告與 `solvable`）、解答圖、
以及**可直接貼進其他分頁的 Python literal**（`layout` 與 `walls`）——這是為了「模型讀錯時使用者能自己修」。

### 每次呼叫都落盤，而且形狀是刻意的

`request_log.py` 把每次呼叫的**圖**與**模型原話**寫進 `logs/vision/`（git-ignored），
形狀刻意做成**訓練資料集的形狀**——`images/` ＋ `metadata.jsonl`，欄位名與 `dataset_builder` 一致。

`score_predictions` 需要的是 `label` ＋ `raw_output`，並帶走 `file_name` 與 `generation_seconds`；
除了 `label`（沒人能替使用者剛上傳的圖知道正確答案）之外全部都寫了。
⇒ **手動補一個 `label`，一行真實使用紀錄就變成一筆可算分的評估樣本。**

這是目前累積「真實截圖評估集」最便宜的路，而那正是 §6.3 指出的、目前唯一擋在
「已驗證可讀 LinkedIn 截圖」前面的東西。實測驗過：兩行補 `label` 後 scorer 正確算出
`EXACT MATCH 1/2`，牆 F1 0.889 的那張就是 `puzzle_03`。

**HTTP 422（讀不出來）也會記**，`usable: false` ＋ `parse_error`。那是最值得留的一種——
任何評估集裡都沒有，重試也沒用。為此 `ModelOutputError` 改成會帶著失敗的原文，
否則證據會被關在例外裡丟掉。

### 這一輪抓到的實作缺陷

`pydantic-ai` 的 `agent.run_sync()` **不能在執行中的事件迴圈裡呼叫**，而端點原本寫成 `async def`：

```
VisionBackendError: ... RuntimeError: This event loop is already running
```

**單元測試全綠卻沒抓到**，因為 stub backend 根本不碰事件迴圈——只有實際打一次 API 才會爆。
修法是改成同步 `def`，讓 FastAPI 丟進 threadpool（這個 handler 從頭到尾都是阻塞工作，本來就該如此，
順便不會卡住事件迴圈）。

**同時把 stub 改成會檢查有沒有執行中的事件迴圈**，讓這個約束以後由測試守住。
已實證這個守衛有效：把 handler 改回 `async def`，12 個測試中 **8 個失敗**。

---

## 8. 落盤位置

| 東西 | 位置 |
|---|---|
| merge 工具 | `src/core/vl_models/merge_lora.py`（＋ 16 個測試） |
| 本機 held-out 執行器 | `src/core/vl_models/run_holdout.py` |
| API | `src/app/routers/vision.py` ＋ `src/app/schemas/vision.py`（＋ 20 個測試） |
| 呼叫落盤 | `src/core/vl_models/request_log.py`（＋ 10 個測試）→ `logs/vision/`（不進版控） |
| UI | `src/ui/gradio_app.py` 的 `solve_from_image_ui`（＋ 4 個測試） |
| 200 筆本機預測與算分 | [`artifacts/vl-p4d/p4d_holdout_predictions.jsonl`](artifacts/vl-p4d/p4d_holdout_predictions.jsonl)、`…_scored.json` |
| 六張真實截圖的完整量測 | `artifacts/vl-p4d/20260829-031817_zip-qwen35-4b-p4c_f16_openai-compat.json` |
| 模型（不進版控） | `models/`：`base` 8.8 GB、`merged` 8.8 GB、`gguf` 8.5 GB、`colab_finetune` 3.8 GB |

測試：**167 passed → 214 passed, 8 xfailed**（新增 47 個），`ruff` 全綠。
