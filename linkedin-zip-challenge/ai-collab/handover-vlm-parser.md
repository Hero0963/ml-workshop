# 交接文件 — VLM Track（圖片解析）

> **接手這條 track 的 agent／developer 從這一份開始讀，讀完就能動手。**
> 最後更新：2026-08-22（Asia/Taipei）｜分支 `feat/vlm-parser`｜worktree `zip-vlm`｜對應 roadmap 第 2 項
> 其他文件是延伸閱讀，本檔會標明什麼時候該去翻哪一份。

---

## 0. 一句話現況

**讀圖這件事已經做完了，而且做滿了：微調後的模型在 200 筆合成 held-out 上四層指標全部 1.000（端到端 200/200）。剩下的不是「讀得準不準」，是「怎麼把它接進產品」——P4d 匯出、P5 Gradio 分頁、P6 `/api/vision/solve`。**

> ★ **接手前一定要知道的五件事：**
>
> **① 那個 200/200 現在還碰不到。** 微調成果是一個 **LoRA adapter，只存在 Google Drive 上**
> （`colab_finetune/p4c_qwen35_4b_zip_lora`，168 MB）。`puzzle_parser.parse_puzzle_image()`
> 今天走的仍然是 Ollama 上**未微調**的模型，實力是牆 F1 0.438、端到端 2/6。
> **要讓產品拿到那個 200/200，P4d（匯出）是唯一的路，也是下一步。**
>
> **② 評估集已經飽和，這是好消息帶來的壞消息。** 所有指標都是 1.000 ⇒ **這把尺再也量不出差異**。
> 任何後續改動（視覺層消融、CoD、減少訓練量、batch 調整）在它上面都會是 1.000。
> **要重獲鑑別力，只能把合成資料變難**（視覺雜訊、多種渲染風格、模擬截圖壓縮與縮放、更大盤面）。
> 沒做這件事之前，不要宣稱任何「改進」。
>
> **③ 不做真實截圖，就用自產的合成資料。** 2026-08-22 本人明示定案（原本 P3 是「暫緩」，現在是「不做」）。
> **代價要講清楚**：所有數字證明的是「**學會了我們的 renderer**」，**不證明**「看得懂 LinkedIn 截圖」。
> 這是明示接受的取捨。**不要再提議收真實截圖**，要做由本人開口。
>
> **④ 推論 prompt 必須用 `build_inference_prompt()` 產生**，不可自己拼。它從**訓練那條渲染路徑**
> 推導，by construction 相符。這個設計已經救過兩次——兩次都是因為我對訓練渲染的描述是錯的，
> 而方法不依賴描述正確（詳見 §7）。
>
> **⑤ 只做 6×6。** renderer 與 builder 都吃 size 清單，要加只是一個旗標，但訓練資料 100% 是 6×6，
> 微調後模型很可能看到 7×7 也答 6×6。

---

## 1. 開工前照這個順序讀

| 順序 | 檔案 | 什麼時候讀 |
|---|---|---|
| 1 | **本檔的 §0（五件必知）→ §6（下一步）** | 一定，而且先讀這兩節 |
| 2 | [`reports/2026-08-22_vl-p4c-results.md`](reports/2026-08-22_vl-p4c-results.md) | 一定。P4c 的完整結果、五項對抗性檢查、四個缺陷、硬體對照 |
| 3 | [`roadmap.md`](roadmap.md) 的「已定案，不要再重開的決策」表 | 一定，快速掃過。**那張表是為了不讓你重蹈已經踩過的坑** |
| 4 | [`../AGENTS.md`](../AGENTS.md) | 動手前。環境、驗證、紅線、回報格式 |
| 5 | [`plans/2026-08-15_track-vlm-parser.md`](plans/2026-08-15_track-vlm-parser.md) | 要看 P0–P6 各階段的 done 條件時 |
| 6 | [`project_guide.md`](project_guide.md) | 要改 `src/app/`／`src/ui/` 時——**P5/P6 一定會用到** |
| 7 | [`reports/2026-08-15_vl-p0-p1-baseline.html`](reports/2026-08-15_vl-p0-p1-baseline.html)（瀏覽器開） | 要看**未微調** baseline 的完整數字與方法時 |
| 8 | [`reports/2026-08-15_vlm-model-survey.html`](reports/2026-08-15_vlm-model-survey.html) | 想知道選型的完整推理時。⚠ 部分內容已被第 7 份與 dev_log 修正 |
| 9 | [`dev_log.md`](dev_log.md) | 只在需要歷史脈絡時。**1,400+ 行，用日期或關鍵字搜，不要整份讀** |

> **只想動手、不想讀完的話**：§0 ＋ §6 ＋ 第 2 份報告，這三個就夠開工。
> P4c 的 notebook 本身寫得很囉唆是刻意的——每個和 P4a 不一樣的地方都在 markdown cell 裡講了為什麼，
> 而且**它保留了那一輪的全部執行輸出當存證**。

---

## 2. 環境建置（照抄可跑）

```powershell
# 已經有 worktree 的話跳過這段
cd D:\it_project\github_sync\ml-workshop
git worktree add ..\zip-vlm -b feat/vlm-parser main
Copy-Item .\linkedin-zip-challenge\.env ..\zip-vlm\linkedin-zip-challenge\.env   # .env 不進版控，必須複製

# 每次開工
cd D:\it_project\github_sync\zip-vlm\linkedin-zip-challenge   # ★ 一定要進子專案再跑 uv
uv sync
uv run pytest                 # 基線 167 passed, 8 xfailed
uv run ruff check .           # 應為 All checks passed!
```

> ⚠ **`uv` 路徑陷阱**：repo 根 `ml-workshop/.venv` 是 py3.9 devtools，**不能拿來跑這個子專案**。
> 一律 `cd linkedin-zip-challenge` 之後才 `uv run`。

### Ollama（Docker）

```powershell
cd D:\it_project\github_sync\zip-vlm\linkedin-zip-challenge
docker compose -f docker-compose.dev.yml up -d ollama
docker exec zip_ollama_server ollama --version     # 應為 0.32.13 或更新
docker exec zip_ollama_server ollama list          # 應看到 gemma4:e4b / qwen3.5:4b-q8_0 等
```

- 容器名 **`zip_ollama_server`**、host 埠 **11435**（不是 11434，11434 被本機另一專案佔用，**不要改回去**）。
- 模型放在具名 volume `linkedin-zip-challenge_ollama_data`，**所有 worktree 共用同一份**，不必重拉。
- ⚠ **兩個 worktree 不可同時起 docker stack**（容器名與埠全機唯一）。要跟 RL track 協調。

### Colab（只有要再訓練或匯出時才需要）

VS Code → **Select Kernel → Colab → L4（付費）**。不必裝 WSL2，官方 CLI 的 Windows 限制繞得過去。
`drive.mount()` 若回 `credentials-propagation ... Bad Request`，**同一個 session 在瀏覽器也開一個 Colab 分頁**再重跑那格。

---

## 3. 已完成且已驗證的事實（**不要重驗，浪費時間**）

**推論環境**

- 容器 Ollama 已由 0.16.1 更新到 **0.32.13**；2025-10 的 15GB 舊模型完整存活。
- 已從官方 library 拉下 `gemma4:e4b`、`gemma4:e4b-it-q8_0`、`qwen3.5:4b`、`qwen3.5:4b-q8_0`。
- **四種模型／量化組合全部 100% 載入 GPU、無 CPU offload**，峰值最高 9582 MiB
  → **16GB 顯卡對 4B 級 Q8 不是瓶頸**。
- Gradio 由 5.49.1 升到 6.15.1，已用 Chrome headless 與主工作樹對拍，**UI 無回歸**。

**訓練（P4c，2026-08-22）**

- **975 步 / 1.56 h / 5.77 s/step / 峰值 VRAM 20.90 of 22.03 GiB (94.9%) / 約 2.4 CU。**
- 視覺層有學到且動得比語言層大：`visual` max|B| **0.272** vs `language` **0.166**（96/96 與 248/248 非零）。
- adapter 168 MB ＋ checkpoint 200/400/600/800/975 **都在 Drive 上**。
- **held-out 200 筆四層指標全 1.000**，並通過五項對抗性檢查（見報告 §2）。

**硬體對照（決定下一輪在哪裡跑時用）**

- 本機 **RTX 4070 Ti SUPER**：bf16 **90.06 TFLOP/s**、頻寬 **588 GB/s**、VRAM 15.99 GiB。
- Colab **L4**：bf16 **64.11 TFLOP/s**、頻寬 **300 GB/s**（GDDR6，非 HBM）、VRAM 22.03 GiB。
- ⇒ **L4 不是比較快，是 VRAM 比較大**（本機快 1.4 倍）。訓練峰值 20.90 GiB ⇒ **本機在 batch 2 完全塞不下**。
- 訓練期 MFU 約 **42%**，頻寬只佔約 4% ⇒ **訓練不缺頻寬**。
- **batch size 往上調反而虧**：圖是變動尺寸的，batch 2 已浪費 18.9% 算力在 padding，batch 4/8 要多付 18%/29%。

## 4. 三個最重要的結論

### 4.1 未微調的瓶頸純粹是「牆」，而微調把它解掉了

未微調最佳設定（見 §5）在六張真實截圖上：逐格 0.947、號碼 0.917、尺寸 6/6，**但牆 F1 只有 0.438**，端到端僅 2/6。
**版面與數字早已接近滿分，過不了關的原因只有牆。**

P4c 之後，合成 held-out 上牆 F1（有牆題）**1.000**、端到端 **200/200**。
⚠ 兩者**不可直接比較**（真實截圖 vs 合成圖），但瓶頸被解決的幅度是明確的。

**牆的偽陽性和漏檢都致命，但失敗模式不同**——這是管線設計推出來的：

| 錯誤 | 集合關係 | 後果 |
|---|---|---|
| **多幻覺牆**（偽陽性） | 預測 ⊇ 真實 | 解出來的路**仍然合法**；風險是過度受限導致「無解」——**看得見的失敗** |
| **漏讀牆**（偽陰性） | 預測 ⊉ 真實 | 解出來的路**可能穿牆，而且沒有任何跡象**——**靜默的錯誤答案** |

⇒ **出貨時 recall 比 precision 更要命**，而且**求解器本身就是免費的驗證器**：出題器先畫 Hamiltonian path 再挖題，所以真實盤面必定有解 ⇒ **預測盤面無解就一定是讀錯了，不需要 ground truth 也知道**。`score_predictions.py` 的 `solvable` / `solvable_but_wrong` 就是量這個，**P6 端點應該把它做成回應的一部分**。

### 4.2 設定要逐模型實測，不可跨模型沿用

| 介入 | `qwen3.5:4b-q8_0` | `gemma4:e4b` |
|---|---|---|
| 關閉思考（`--no-think`） | JSON 3/6 → **6/6**、快 5.8 倍 | 各項**變差** |
| 格盤尺寸指示＋7×7 範例（`--prompt sized`） | 逐格 0.924 → **0.961**、相符 1/6 → **2/6** | 尺寸 4/6 → **3/6**，把真 6×6 過度矯正成 7×7 |

量化也一樣：`qwen3.5:4b` 的 **Q4 完全吐不出 JSON**，Q8 版面全對；`gemma4:e4b` 反而 Q8 比 Q4 差。

### 4.3 模型世代的選擇是被「尺寸」決定的

Qwen3.6 最小 27B、Qwen3.7 無開放權重、Qwen3.8 只有 27B；27B 在 Q4 約 17GB，**超過本機 16GB**。
**Qwen3.5 是唯一有 ≤10B 尺寸的 Qwen 世代**；Gemma 4 同理（官方 vision 微調只支援 E2B／E4B）。

---

## 5. 未微調 baseline 怎麼重跑（比較的參考點）

```powershell
cd D:\it_project\github_sync\zip-vlm\linkedin-zip-challenge

# 未微調最佳設定
uv run python -m src.core.vl_models.benchmark `
  --model qwen3.5:4b-q8_0 --no-think --prompt sized
```

常用旗標：`--images puzzle_01 puzzle_04`、`--client pydantic-ai`（走出貨路徑）、`--out-dir <dir>`、`--seed`／`--temperature`／`--num-ctx`。
**微調後的模型要用 `--prompt finetune`。**

**原始資料落盤在** `ai-collab/reports/artifacts/`：

| 目錄 | 內容 |
|---|---|
| `vl-benchmark/` | P0 矩陣（4 種模型／量化 × 冷暖） |
| `vl-baseline-p1/`／`-nothink/`／`vl-prompt-sized/` | P1 baseline 與兩個消融 |
| `vl-client-crosscheck/` | native vs pydantic-ai 傳輸層對照 |
| **`vl-p4c/`** | **P4c 的 200 筆原始預測 ＋ 算分結果** |

離線算分（Colab 只吐原始輸出，指標一律在本機算，**不在 notebook 裡重寫指標**）：

```powershell
uv run python -m src.core.vl_models.score_predictions ai-collab\reports\artifacts\vl-p4c\p4c_holdout_predictions.jsonl
```

---

## 6. 下一步

讀圖已經達標，**接下來三件都是「把它接進產品」**。建議順序就是下面的順序，因為 P5／P6 沒有 P4d 就只能接到未微調的模型。

### P4d — 把 adapter 變成本機跑得動的模型 ★ 先做這個

**問題**：微調成果現在只是 Drive 上的一個 LoRA adapter，本機的 Ollama 用不到。

**路線**：

1. 在 Colab 上 merge adapter 進 base，匯出 **GGUF** → 拉回本機給 Ollama。
   ⚠ **已知風險 unsloth#3899（vision 匯出缺陷）**，匯出後**一定要實測**視覺能力沒壞：
   拿 `datasets/vl/main_6x6` 的幾張圖跑 `benchmark.py --prompt finetune`，對得上才算成功。
2. GGUF 失敗就走 **`transformers` backend** —— `backends.py` 目前只有兩個 Ollama 傳輸，
   要新增第三個（本機直接載 transformers ＋ adapter）。介面照 `VisionBackend` 抄，
   **傳輸層只有這一份正本，不要在別處重寫**。

**Done 條件**：`puzzle_parser.parse_puzzle_image()` 走微調後的模型，對合成圖能重現 P4c 的水準。

### P5 — Gradio 上傳分頁

`src/ui/gradio_app.py` 是 **Adapter**，只負責把 UI 操作翻成 API 格式，**邏輯不要塞進去**。
上傳圖 → 呼叫 P6 的端點 → 顯示解析出的盤面 ＋ 解答（沿用既有的 GIF/PNG 視覺化）。
`ParseResult.warnings`（被丟掉的幻覺牆）**要顯示給使用者**，不要吞掉。

### P6 — `/api/vision/solve`

`src/app/routers/` ＋ `src/app/schemas/` **成對改**，跑 `src/app/tests/`。
回應建議包含：解析出的 `Puzzle`、solver 的路徑、`warnings`、以及 **§4.1 那個「預測盤面是否有解」的信心旗標**。
模型不在時要有明確錯誤訊息（`VisionBackendError` / `ModelOutputError` 已經備好，**不要讓端點靜默壞掉**）。

### 想再訓練的話，先讀這個

**評估集已飽和（§0 ②）。在把合成資料變難之前，任何再訓練都量不出好壞。** 前置工作：
更多視覺雜訊、多種渲染風格、模擬截圖的壓縮與縮放失真、更大盤面。
`render_puzzle.py` 與 `dataset_builder.py` 都吃參數，改動範圍不大。

**唯一還有鑑別力的現成實驗**：Drive 上還有 `checkpoint-200/400/600/800`。loss 在第 250 步就到雜訊地板，
若 checkpoint-200 對同一批 200 筆 held-out 也是滿分 ⇒ **1,600 筆就夠了，本輪 4/5 的訓練量是白付的**。
它比較的是「不同模型」而非「同一個滿分」，所以仍量得出東西，成本只有一次**批次**推論。

---

## 7. 陷阱清單（都是實際踩過的）

| 陷阱 | 說明 |
|---|---|
| **`uv` 路徑** | repo 根 `.venv` 是 py3.9 devtools。一律 `cd linkedin-zip-challenge` 再 `uv run` |
| **★ 推論 prompt 不可自己拼** | 訓練與推論曾有**兩處**渲染不一致（thinking 區塊、text/image 順序），代價是「內容全對、格式全錯」（JSON 4/4 vs 0/3）。**一律用 `build_inference_prompt()`**，它從訓練渲染路徑推導。**它救過我兩次，兩次都是因為我對訓練渲染的描述是錯的** |
| **★ 兩包資料集 seed 只差 1 ＝ 同一批資料** | `draw_recipe` 用 `random.Random(seed + index)`，seed 差 1 的兩包是同一亂數流**位移一格**。`smoke_6x6`(20260823) 與 `main_6x6`(20260822) 實測 **120/120 渲染 recipe 相同、82/120 標籤相同**。**要獨立資料集，seed 要差得夠遠，或直接從同一包切 disjoint 的 slice** |
| **★ Colab 的 Drive FUSE 只在關檔時才上傳** | 逐行 `flush()` 只推到 FUSE 層，雲端上**看不到檔案**，斷線就全沒了。**寫本機 `/content`，每批用 `shutil.copy` 覆蓋到 Drive**（copy 會開檔關檔才會觸發上傳） |
| **★ 短跑量出的 s/step 不能拿來外推** | P4c 的 lr=0 五步短跑量到 37.59 s/step、投影 10.18 h，實際 5.77 s/step / 1.56 h（**差 6.5 倍**）——第一步吃掉全部編譯與 autotune 成本。**丟掉第一步再平均，或標成「上限」** |
| **★ 評估不要用 batch 1 逐筆生成** | P4c 推論 3.0 CU > 訓練 2.4 CU，本末倒置。batch 1 解碼是**每 token 固定成本綁死**（只跑到 roofline 27%），批次化近乎線性加速。**下一輪一律批次 ＋ 推論前 merge LoRA** |
| **VRAM 峰值要跑滿才準** | P4a 50 步量到 16.57 GiB (75%)，P4c 跑滿 975 步是 **20.90 GiB (94.9%)**——短跑沒抽到 998px 的大圖。**別用短跑的 VRAM 決定 batch size** |
| **cell magic 必須在第一行** | 在 `%%capture` 上面放註解，Jupyter 會報 `Line magic function not found`，整格不執行、後面全部連鎖失敗 |
| **煙霧測試不要用 `%%capture`** | 它會吞掉安裝輸出，裝失敗時變成後面莫名其妙的 import 錯誤 |
| **Drive 掛載會 400** | 新 runtime 第一次 `drive.mount()` 可能回 `credentials-propagation ... Bad Request`。**同一個 session 在瀏覽器也開一個分頁，再重跑那格** |
| **VS Code 對遠端 Colab kernel 的 Interrupt 不可靠** | 2026-08-22 實測按了沒反應。改用 **Colab 網頁分頁 → Runtime → Interrupt execution**。⚠ **絕對不要按 Terminate／Disconnect**——那不是優雅關閉，開著的檔案不會上傳 |
| **VS Code 不顯示 cell 編號** | 助理讀 `.ipynb` 看到的是 JSON 索引，使用者畫面上沒有。**指路要用 markdown 標題** |
| **`torch.cuda.is_bf16_supported()` 會騙人** | 預設 `including_emulation=True`，在無 bf16 硬體的 T4 上照樣回 `True`。**一律問 `including_emulation=False`** |
| **`seed` 不保證決定性** | `gemma4:e4b` Q4 冷、暖兩次結果不同。**對照實驗要多次重複取平均** |
| **牆 F1 會被無牆題灌水** | 無牆題預測 0 道就白拿 F1 = 1.0。**看 `mean_wall_f1_walled_only`**，或直接看 `exact_match` / `solution_valid_on_truth` |
| **few-shot 有洩題** | `puzzle_01`–`03` 的答案就寫在 baseline prompt 裡，它們的成績不能當泛化證據 |
| **改 prompt 要開新變體** | baseline prompt（`prompt_baseline.build_puzzle_prompt()`）**已凍結**，報告數字都是對著它量的。新想法加進 `prompt_variants.py` 用 `--prompt` 切換，改動會被雜湊測試擋下 |
| **關思考的旋鈕逐傳輸層不同** | `/api/chat` 吃 `think`，**`/v1` 完全忽略它**、要 `reasoning_effort="none"`。翻譯已收進 `backends.py`，**不要在別處重寫傳輸** |
| **同一個埠不要重複起 app** | Windows 上 7440 會同時有多個 LISTENING，回應的是最舊的那個 process，症狀是「改了程式卻沒反應」。先 `netstat -ano \| grep :7440` 確認 |
| **P1 的 GPU 峰值不可用** | 連續跑兩個模型時前一個仍常駐，數字被污染。**VRAM 只採用報告 §2 的 P0 數字** |

---

## 8. 待本人決定（agent 不要自己決定）

1. ~~**微調要不要付 Colab CU**~~ → **已定案（2026-08-22）：可付費**，走 Qwen3.5-4B ＋ L4 ＋ bf16 LoRA。
2. ~~**P3 真實驗證集**~~ → **已定案（2026-08-22）：不做**。就用自產的合成資料。
   **不要再提議收真實截圖**；要做由本人開口。代價見 §0 ③。
3. ~~**P4c 要不要順便跑 CoD 變體**~~ → **已定案（2026-08-22）：不跑**，先只做無 CoD。
   `generate_cod_dataset.generate_chain_of_draft_str()` 還在，未來要試隨時可用——但注意 §0 ② 的飽和問題，
   **現在跑也比不出好壞**。
4. **新增套件**一律由本人 `uv add`，agent 只負責說要裝什麼、為什麼。
   `pyproject.toml`／`uv.lock` 是與 RL track 的**共用衝突點**。
5. **2025-10 的舊訓練資產只在 Google Drive**（`colab_finetune/` 的 `cod_dataset_*.zip`、`all_trained_runs/`、
   `trained_models/`）。本機已無副本。要不要撈回來重用**由本人決定**——但 P4c 之後價值不高。

---

## 9. 這條 track 動過的檔案

| 檔案 | 說明 |
|---|---|
| `src/core/vl_models/backends.py` | 傳輸層**唯一正本**。關思考的旋鈕在兩條 HTTP 路徑上不同，翻譯只在這裡 |
| `src/core/vl_models/puzzle_parser.py` | 正式 parser：走 `settings`、`loguru`、失敗用例外不回 `None`、幻覺牆會被丟掉並回報 |
| `src/core/vl_models/prompt_baseline.py` | **凍結**的 baseline prompt，逐位元組被雜湊測試釘住 |
| `src/core/vl_models/prompt_variants.py` | prompt 變體（`sized`、`finetune`）。`FINETUNE_INSTRUCTION` 是訓練與其推論共用的短指令 |
| `src/core/vl_models/schema.py` | 標籤與推論共用同一個 Pydantic 契約，`to_prompt_json` 逐行對齊 few-shot 範例 |
| `src/core/vl_models/render_puzzle.py` | LinkedIn 風格 renderer；格線淺灰、牆粗黑（舊版兩者同色）；Pillow 內建字型跨平台一致 |
| `src/core/vl_models/dataset_builder.py` | 自己抽 0–12 道牆、增強、CP-SAT 驗證、SHA-256 產物摘要與 `--check` |
| `src/core/vl_models/benchmark.py` | 四層指標量測工具（對真實截圖），透過 `backends.py` |
| **`src/core/vl_models/score_predictions.py`** | **離線算分**：把 Colab 產的 `predictions.jsonl` 在本機算成四層指標。走正式 parser ＋ benchmark 的指標函式，**不重寫指標**。含 `path_is_legal()`、`solution_valid_on_truth`、`solvable_but_wrong`、按牆數分層 |
| `src/core/vl_models/final_puzzle_parser.py` | **SCRATCHPAD**（未刪），改為 re-export |
| `notebooks/colab_smoke_test.ipynb` | Colab kernel 的 GPU/bf16 煙霧測試（第 8 格是 matmul 對照，量新卡用它） |
| `notebooks/p4a_finetune_smoke.ipynb` | P4a 訓練煙霧測試，**保留執行輸出當存證** |
| `notebooks/p4a_verify_e0_e1.ipynb` | E0（渲染修法對照）＋ E1（解析度定價），載入已存 adapter，**不訓練** |
| **`notebooks/p4c_finetune_8000.ipynb`** | **P4c 正式訓練**：lazy 資料集、lr=0 的不污染短跑、Drive checkpoint／resume、只輸出原始預測不算指標。**保留全部執行輸出** |
| `src/core/tests/vl_models/` | 91 個測試，全部 mock 或小規模，**不需要 Ollama 或 GPU** |
| `ai-collab/reports/2026-08-22_vl-p4c-results.md` | **P4c 完整結果報告** |
| `ai-collab/reports/artifacts/vl-p4c/` | P4c 的 200 筆原始預測 ＋ 算分結果 |
| `.env.example`／`pytest.ini`／`pyproject.toml`／`uv.lock` | 設定與相依修正 |

**沒有動到**：`src/core/rl/`（RL track 的地盤）、`src/core/puzzle_generation/`、`src/core/utils.py`（共用模組）、
`src/core/solvers/`、`src/app/`、`src/ui/`。**P5／P6 會是第一次動到後兩者。**
