# 交接文件 — VLM Track（圖片解析）

> **接手這條 track 的 agent／developer 從這一份開始讀，讀完就能動手。**
> 最後更新：2026-08-22（Asia/Taipei）｜分支 `feat/vlm-parser`｜worktree `zip-vlm`｜對應 roadmap 第 2 項
> 其他文件是延伸閱讀，本檔會標明什麼時候該去翻哪一份。

---

## 0. 一句話現況

**P0–P2 與 P4a 已完成：8,000 張合成資料集已生成並上傳、Colab L4 通路打通、煙霧測試通過，並修掉一個會讓 P4c 比較失效的渲染缺陷。下一步是 P4c——第一次真正的訓練（8,000 筆 × 1 epoch，約 2.09 小時 / 3.2 CU）。**

> ★ **接手前一定要知道的三件事（2026-08-22）：**
>
> **① 範圍已被本人縮小，這是決定不是疏漏。** 現階段**只追求「學會我們畫的圖」**，真實截圖（P3）暫緩。
> 所有評估用合成 held-out。**代價：量不到 domain gap**——數字證明「學會這個 renderer」，
> 不證明「看得懂 LinkedIn 截圖」。
>
> **② 推論 prompt 必須用 `build_inference_prompt()` 產生**，不可自己拼。訓練與推論曾有兩處渲染
> 不一致（thinking 區塊、text/image 順序），代價是「內容全對但格式全錯」。
>
> **③ 只做 6×6。** renderer 與 builder 都吃 size 清單，要加只是一個旗標，但訓練資料 100% 是 6×6，
> 微調後模型很可能看到 7×7 也答 6×6。

---

## 1. 開工前照這個順序讀

| 順序 | 檔案 | 什麼時候讀 |
|---|---|---|
| 1 | **本檔的 §0（三件必知）→ §6（下一步 P4c）** | 一定，而且先讀這兩節 |
| 2 | [`roadmap.md`](roadmap.md) 的「已定案，不要再重開的決策」表 | 一定，快速掃過。**那張表是為了不讓你重蹈已經踩過的坑** |
| 3 | [`../AGENTS.md`](../AGENTS.md) | 動手前。環境、驗證、紅線、回報格式 |
| 4 | [`plans/2026-08-15_track-vlm-parser.md`](plans/2026-08-15_track-vlm-parser.md) | 要看 P2–P6 各階段的 done 條件時 |
| 5 | [`reports/2026-08-15_vl-p0-p1-baseline.html`](reports/2026-08-15_vl-p0-p1-baseline.html)（瀏覽器開） | 要看**未微調** baseline 的完整數字與方法時。⚠ 其中「Q4 完全不行」的結論**可能被混淆**，見 `dev_log.md` 2026-08-22 的 open question |
| 6 | [`project_guide.md`](project_guide.md) | 要改 `src/app/`／`src/ui/` 時 |
| 7 | [`reports/2026-08-15_vlm-model-survey.html`](reports/2026-08-15_vlm-model-survey.html) | 想知道選型的完整推理時。⚠ 部分內容已被第 5 份與 dev_log 修正 |
| 8 | [`dev_log.md`](dev_log.md) | 只在需要歷史脈絡時。**1,000+ 行，用日期或關鍵字搜，不要整份讀**。2026-08-22 那一則是本次 session 的完整紀錄 |

> **只想動手、不想讀完的話**：§0 ＋ §6 ＋ roadmap 的決策表，這三個就夠開工。其他等需要時再翻。

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
uv run pytest                 # 基線 136 passed, 8 xfailed
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

---

## 3. 已完成且已驗證的事實（**不要重驗，浪費時間**）

- 容器 Ollama 已由 0.16.1 更新到 **0.32.13**；2025-10 的 15GB 舊模型完整存活。
- 已從官方 library 拉下 `gemma4:e4b`、`gemma4:e4b-it-q8_0`、`qwen3.5:4b`、`qwen3.5:4b-q8_0`。
- **四種模型／量化組合全部 100% 載入 GPU、無 CPU offload**，峰值最高 9582 MiB
  → **16GB 顯卡對 4B 級 Q8 不是瓶頸**（計畫書風險 #5 的推論端已解除）。
- 量測工具 `src/core/vl_models/benchmark.py` 的評分邏輯**已自檢**（六題 ground truth 當完美預測 ＋ 8 組邊界案例全過）。
- Gradio 由 5.49.1 升到 6.15.1，已用 Chrome headless 與主工作樹對拍，**UI 無回歸**。

---

## 4. 三個最重要的結論

### 4.1 瓶頸幾乎純粹是「牆」，不是認字

目前最佳未微調設定（見 §5）在六張真實截圖上：

| 指標 | 成績 |
|---|---|
| JSON 可解析率 | **6/6** |
| 格盤尺寸判斷 | **6/6** |
| 逐格準確率 | **0.961** |
| 號碼格召回 | **0.917** |
| **牆 F1（有牆題）** | **0.438** ← 就是這個 |
| 端到端與 ground truth 相符 | **2/6** |

**版面與數字已接近讀滿分，端到端過不了關的原因只有牆。**
→ **P2 合成資料的重心要放在牆**：粗細、壓在格線上的位置、與格線的對比、以及 0–16 的數量分布
（出題器預設只產 2–5 道，`puzzle_04` 實際有 14 道）。其他增強次要。

**牆的偽陽性和漏檢一樣致命**：`gemma4:e4b` 在 `puzzle_03` 上真牆 4/4 全中，卻因為多幻覺 2 道而整題無解。
**評估牆一定要同時看 precision 與 recall，不能只看 recall。**

### 4.2 設定要逐模型實測，不可跨模型沿用

兩個零成本介入的效果**方向完全相反**：

| 介入 | `qwen3.5:4b-q8_0` | `gemma4:e4b` |
|---|---|---|
| 關閉思考（`--no-think`） | JSON 3/6 → **6/6**、快 5.8 倍 | 各項**變差** |
| 格盤尺寸指示＋7×7 範例（`--prompt sized`） | 逐格 0.924 → **0.961**、相符 1/6 → **2/6** | 尺寸 4/6 → **3/6**，把真 6×6 過度矯正成 7×7 |

量化選擇也一樣：`qwen3.5:4b` 的 **Q4 完全吐不出 JSON**（思考 16,505 字元後撞頂），Q8 卻版面全對；
`gemma4:e4b` 反而是 **Q8 比 Q4 差**。

### 4.3 模型世代的選擇是被「尺寸」決定的

Qwen3.6 最小 27B、Qwen3.7 無開放權重、Qwen3.8 只有 27B／2.4T；27B 在 Q4 約 17GB，**超過本機 16GB**。
**Qwen3.5 是唯一有 ≤10B 尺寸的 Qwen 世代**；Gemma 4 同理（官方 vision 微調只支援 E2B／E4B）。
DiffusionGemma 則 Q4 就要 18GB、不在 Ollama library、微調要 A100，**三重不符**。

完整世代表（發布日、尺寸、能否本機跑、能否免費微調）與**重驗方法**見報告 §9。

---

## 5. 目前最佳設定與怎麼重跑

```powershell
cd D:\it_project\github_sync\zip-vlm\linkedin-zip-challenge

# 目前最佳未微調設定 —— 這就是微調必須超越的門檻
uv run python -m src.core.vl_models.benchmark `
  --model qwen3.5:4b-q8_0 --no-think --prompt sized

# 對照：gemma4 最佳設定（預設思考、baseline prompt）
uv run python -m src.core.vl_models.benchmark --model gemma4:e4b
```

常用旗標：`--images puzzle_01 puzzle_04`（只跑部分）、`--client pydantic-ai`（走出貨路徑）、
`--out-dir <dir>`（分開存放）、`--seed`／`--temperature`／`--num-ctx`。

**原始資料落盤在** `ai-collab/reports/artifacts/`：

| 目錄 | 內容 |
|---|---|
| `vl-benchmark/` | P0 矩陣（4 種模型／量化 × 冷暖） |
| `vl-baseline-p1/` | P1 baseline（6 張圖 × 2 模型） |
| `vl-baseline-p1-nothink/` | 關閉思考的消融 |
| `vl-prompt-sized/` | sized prompt 的消融 |
| `vl-client-crosscheck/` | native vs pydantic-ai 傳輸層對照 |

每個 JSON 都含 seed、原始模型輸出、token 數、GPU 取樣，可完全追溯。

---

## 6. 下一步：P4c（8,000 筆正式訓練）

**資料已就緒**，Drive 上有兩包（`我的雲端硬碟/colab_finetune/`）：

| 檔案 | 內容 | sha256 |
|---|---|---|
| `zip_vl_6x6_smoke120_20260822.tar` | 120 筆，P4a 用 | `4424ecc8…f266` |
| `zip_vl_6x6_8000_20260822.tar` | **8,000 筆，P4c 用** | `69c753e1…0fbf` |

兩包內都有 `metadata.jsonl`（`file_name` ＋ `label`）與 `manifest.json`（seed、分布、SHA-256）。
`uv run python -m src.core.vl_models.dataset_builder --check <dir>` 可重驗。

### 照這個做

1. **從 `notebooks/p4a_finetune_smoke.ipynb` 複製一份**改成 P4c，換掉 `ARCHIVE`／`DATASET`／
   `EXPECTED_SHA256`，並把 `max_steps=50` 換成 `num_train_epochs=1`。
2. **★ 推論一律用 `build_inference_prompt()`**（在 `notebooks/p4a_verify_e0_e1.ipynb` 裡，
   E0 那格）。不要用 `apply_chat_template(..., add_generation_prompt=True)` 直接拼。
3. **⚠ 不要照抄 P4a 的資料載入方式。** 它把 120 張圖全部解碼成 PIL 物件放進 Python list；
   8,000 張約 **10 GB 系統記憶體**，而標準 Colab VM 只有約 12.7 GB。**要改成 lazy**
   （`imagefolder` 的 `datasets.Dataset` ＋ transform），而且**先用短跑驗證再開長跑**。
4. **加 checkpoint／resume。** 2.09 小時的 session 有斷線風險，`SFTConfig` 的
   `save_steps` ＋ `output_dir` 指到 Drive。
5. **評估**：用合成 held-out（從 8,000 切，或另外生一包新 seed 的），指標用
   `benchmark.py` 的四層。微調模型要用 `--prompt finetune`。

### 停損與門檻

- **要打敗的基準**（2026-08-22 實測，六張真實截圖、`--prompt sized`、未微調 `qwen3.5:4b-q8_0`）：
  逐格 **0.947**、牆 F1（有牆題）**0.470**、尺寸 **6/6**、端到端 **2/6**。
- ⚠ 但依 §0 的範圍決定，P4c 的評估在**合成 held-out** 上做，兩者**不可直接比較**。
  P4a 的 held-out 已達牆 F1 **0.958**（116 筆、50 步），所以 P4c 的合格線應該顯著高於此，
  否則代表加資料沒有用。
- **★ 端到端正確要求每一道牆都對**：平均 6 道牆時，每道牆正確率 p=0.85 只換得到 38% 題目全對，
  要 80% 需要 **p ≈ 0.96**。**別用「牆 F1 0.85」當門檻，那太鬆。**

### 之後

P4b（視覺層消融）只在 P4c 失敗時才做——已有弱證據視覺層有效（`max|B|` 0.114 > 語言層 0.059）。
P4d 匯出（GGUF 有已知風險 unsloth#3899，保留 transformers fallback）。
P5 剩 Gradio 上傳分頁、P6 是 `/api/vision/solve`。

## 7. 陷阱清單（都是實際踩過的）

| 陷阱 | 說明 |
|---|---|
| **`uv` 路徑** | repo 根 `.venv` 是 py3.9 devtools。一律 `cd linkedin-zip-challenge` 再 `uv run` |
| **★ 推論 prompt 不可自己拼** | 訓練與推論曾有**兩處**渲染不一致：thinking 區塊（訓練是 `<think>

</think>

`，推論預設是沒關閉的 `<think>
`）與 **text/image 順序**（訓練 `[text, image]`、推論 `[image, text]`）。代價是「內容全對、格式全錯」。**一律用 `build_inference_prompt()`**，它從訓練那條渲染路徑推導 |
| **cell magic 必須在第一行** | 在 `%%capture` 上面放註解，Jupyter 會報 `Line magic function '%%capture' not found`，整格不執行、套件一個都沒裝、後面全部連鎖失敗。**notebook 產生器已加 assert 擋這件事** |
| **煙霧測試不要用 `%%capture`** | 它會吞掉安裝輸出，裝失敗時變成後面莫名其妙的 import 錯誤。官方 notebook 有它是為了畫面乾淨，我們刻意拿掉 |
| **Drive 掛載會 400** | 新 runtime 第一次 `drive.mount()` 可能回 `credentials-propagation ... Bad Request`。授權流程原本設計在 Colab 網頁跑，從 VS Code 驅動時沒地方跳。**修法：同一個 session 在瀏覽器也開一個分頁，再重跑那格**（2026-08-22 實測有效） |
| **VS Code 不顯示 cell 編號** | 助理讀 `.ipynb` 看到的是 JSON 索引，使用者畫面上沒有。**指路要用 markdown 標題**，不要說「第 13 格」 |
| **`mcp__ide__executeCode` 需要 notebook 是 active editor** | 使用者一在對話框打字就失焦，十次有六次打不到。**可靠做法是請使用者自己跑，助理讀 `.ipynb` 裡存下的輸出** |
| **`torch.cuda.is_bf16_supported()` 會騙人** | 預設 `including_emulation=True`，在無 bf16 硬體的 T4 上照樣回 `True`。**一律問 `including_emulation=False`** |
| **`seed` 不保證決定性** | `gemma4:e4b` Q4 冷、暖兩次結果不同。**對照實驗要多次重複取平均**，不能用單次下結論 |
| **牆 F1 會被無牆題灌水** | `puzzle_02`／`puzzle_06` 無牆，預測 0 道就白拿 F1 = 1.0，把平均從 0.268 抬到 0.512。**看 `mean_wall_f1_walled_only`** |
| **few-shot 有洩題** | `puzzle_01`–`03` 的答案就寫在 prompt 裡。它們的成績**不能當泛化證據**；目前 2/6 相符中只有 `puzzle_05` 是真的 |
| **不要拿評估圖當 prompt 範例** | 7×7 範例是**合成的**（seed 20260815），刻意不用 `puzzle_04`／`06` |
| **survey 會過期得很快** | 前一份報告寫「Qwen3.8 權重未上架」，**一天內就上架了**。版本宣稱一律回一手來源重查，方法見報告 §9.4 |
| **改 prompt 要開新變體** | baseline prompt（`prompt_baseline.build_puzzle_prompt()`）**已凍結**，報告數字都是對著它量的。新想法加進 `prompt_variants.py` 並用 `--prompt` 切換。改動會被 `test_prompt_baseline.py` 的雜湊測試擋下 |
| **關思考的旋鈕逐傳輸層不同** | `/api/chat` 吃 `think`，**`/v1` 完全忽略它**、要 `reasoning_effort="none"`。2026-08-22 實測：同一模型 9.7s／reasoning 1392 字元 → 0.9s／0。翻譯已收進 `backends.py`，**不要在別處重寫傳輸** |
| **同一個埠不要重複起 app** | 2026-08-22 踩到：連續 spawn 幾次 `python -m src.app.main` 後，Windows 上 7440 同時有多個 LISTENING 項目，**回應的是最舊的那個 process**，害我以為 `npm run build` 沒生效。症狀是「改了程式卻沒反應」。先 `netstat -ano \| grep :7440` 確認只有一個活著，或換個埠 `APP_PORT=7441` 驗證 |
| **P1 的 GPU 峰值不可用** | 連續跑兩個模型時前一個仍常駐，數字被污染。**VRAM 只採用報告 §2 的 P0 數字** |

---

## 8. 待本人決定（agent 不要自己決定）

1. ~~**微調要不要付 Colab CU**~~ → **已定案（2026-08-22）：可付費**，因此走
   **Qwen3.5-4B ＋ L4 ＋ bf16 LoRA**（它未微調就只剩「牆」要學）。免費 T4／Gemma 4 E4B 是備案。
2. ~~**要不要 push 到 origin**~~ → **已過期**：2026-08-22 查證 `origin/main` 已含本 track 的
   `b24d8bd`（`git merge-base --is-ancestor feat/vlm-parser main` 為真），成果早就在 origin 上。
   同日把 `feat/vlm-parser` fast-forward 到 `main`（純祖先，不會丟 commit）。
3. **新增套件**一律由本人 `uv add`，agent 只負責說要裝什麼、為什麼。
   `pyproject.toml`／`uv.lock` 是與 RL track 的**共用衝突點**。
4. ~~**P3 真實驗證集**~~ → **已明示暫緩（2026-08-22）**：現階段只追求「學會我們畫的圖」。
   要恢復就是收 30–50 張真實截圖 ＋ 人工標註；在那之前，**任何「能讀懂真實截圖」的宣稱都沒有證據**。
5. **P4c 要不要順便跑 CoD 變體**（把 chain-of-draft 放進 `<think>` 區塊，草稿可由標籤程式化生成，
   `generate_cod_dataset.generate_chain_of_draft_str()` 已存在）。一輪約 1.5–2 CU，兩個都跑不到 8% 餘額。
   支持它的證據：`--prompt sized` 加一句「先數行列」就讓尺寸從 4/6 變 6/6。
6. **2025-10 的舊訓練資產只在 Google Drive 上**（`我的雲端硬碟/colab_finetune/`：
   `cod_dataset_20251024_170006.zip` 14.3MB、`finetune_dataset_20251024_*.zip`、
   `all_trained_runs/`、`trained_models/`）。**本機已無副本**（2026-08-22 全碟搜尋，只剩一個
   `generate_finetune_dataset.cpython-311.pyc` 殘骸；產生它的 `.py` 已於 2025-10-28 的 `422643d` 刪除）。
   要不要撈回來重用、以及 `trained_models/` 裡是什麼，**由本人決定**（見 §6 的 P2 說明）。

---

## 9. 這條 track 動過的檔案

| 檔案 | 說明 |
|---|---|
| `src/core/vl_models/benchmark.py` | **新增**。四層指標量測工具，兩種傳輸層。2026-08-22 改為呼叫 `backends.py`，不再自帶傳輸 |
| `src/core/vl_models/prompt_variants.py` | **新增**。prompt 變體（`sized`），baseline 不動 |
| `src/core/vl_models/backends.py` | **新增 2026-08-22**。傳輸層唯一正本；關思考的旋鈕在兩條路徑上不同，翻譯只在這裡 |
| `src/core/vl_models/puzzle_parser.py` | **新增 2026-08-22**。正式 parser：走 `settings`、`loguru`、失敗用例外不回 `None`、幻覺牆會被丟掉並回報 |
| `src/core/vl_models/prompt_baseline.py` | **新增 2026-08-22**。凍結的 baseline prompt 從 scratchpad 搬出來，逐位元組相同（雜湊測試釘住） |
| `src/core/vl_models/final_puzzle_parser.py` | **標為 SCRATCHPAD 2026-08-22**（未刪）。改為 re-export，`__main__` 走新 parser |
| `src/core/vl_models/render_puzzle.py` | **新增 2026-08-22**。LinkedIn 風格 renderer；格線淺灰、牆粗黑（舊版兩者同色）；字型用 Pillow 內建 Aileron，跨平台一致 |
| `src/core/vl_models/schema.py` | **新增 2026-08-22**。標籤與推論共用同一個 Pydantic 契約，`to_prompt_json` 逐行對齊 few-shot 範例 |
| `src/core/vl_models/dataset_builder.py` | **新增 2026-08-22**。自己抽 0–12 道牆（不動共用模組）、增強、CP-SAT 驗證、SHA-256 產物摘要與 `--check` |
| `notebooks/colab_smoke_test.ipynb` | **新增 2026-08-22**。給 VS Code 的 Colab kernel 用的 GPU/bf16 煙霧測試 |
| `notebooks/p4a_finetune_smoke.ipynb` | **新增 2026-08-22**。P4a 訓練煙霧測試，改自官方 Unsloth notebook。**檔案裡保留了那一輪的執行輸出當存證** |
| `notebooks/p4a_verify_e0_e1.ipynb` | **新增 2026-08-22**。E0（渲染修法對照）＋ E1（解析度定價），載入已存的 adapter，**不訓練** |
| `src/core/vl_models/prompt_variants.py` | 2026-08-22 增 `FINETUNE_INSTRUCTION`（微調與其推論共用的短指令，421 字元 vs baseline few-shot 2,124）與 `--prompt finetune` |
| `src/core/tests/vl_models/` | **新增 2026-08-22**。60 個測試，全部 mock 或小規模，不需要 Ollama 或 GPU |
| `pytest.ini` | 2026-08-22 註冊 `slow` mark |
| `.env.example` | 2026-08-22 補上 `OLLAMA_HOST_PORT`／`OLLAMA_PROVIDER_URL`／`OLLAMA_MODEL_NAME`（先前只有三個 APP 變數） |
| `ai-collab/reports/2026-08-15_vl-p0-p1-baseline.html` | **新增**。完整報告 |
| `ai-collab/reports/artifacts/` | **新增**。所有原始量測資料 |
| `ai-collab/handover-vlm-parser.md` | **新增**。本檔（2026-08-15 由 `handover.md` 改名，因為 RL track 也有自己的交接文件） |
| `ai-collab/roadmap.md`／`dev_log.md` | 更新 |
| `pyproject.toml`／`uv.lock` | 相依修正，理由見 `build(zip)` commit |

**沒有動到**：`src/core/rl/`（RL track 的地盤）、`src/core/puzzle_generation/`、`src/core/utils.py`（共用模組）、
`src/core/solvers/`、`src/app/`、`src/ui/`。
