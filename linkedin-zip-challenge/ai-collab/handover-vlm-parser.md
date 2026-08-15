# 交接文件 — VLM Track（圖片解析）

> **接手這條 track 的 agent／developer 從這一份開始讀，讀完就能動手。**
> 最後更新：2026-08-15（Asia/Taipei）｜分支 `feat/vlm-parser`｜對應 roadmap 第 2 項
> 其他文件是延伸閱讀，本檔會標明什麼時候該去翻哪一份。

---

## 0. 一句話現況

**計畫書的 P0（部署煙霧測試）與 P1（未微調 baseline）已完成並落盤，另外做完兩個零訓練成本的改善；下一步是 P2（合成資料 pipeline），而 P2 的重心已依實測結果從「全面模仿 LinkedIn 風格」修正為「把牆畫對」。**

---

## 1. 開工前照這個順序讀

| 順序 | 檔案 | 什麼時候讀 |
|---|---|---|
| 1 | **本檔** | 一定 |
| 2 | [`reports/2026-08-15_vl-p0-p1-baseline.html`](reports/2026-08-15_vl-p0-p1-baseline.html)（瀏覽器開） | 一定。所有數字、方法、限制、模型世代全表都在裡面 |
| 3 | [`plans/2026-08-15_track-vlm-parser.md`](plans/2026-08-15_track-vlm-parser.md) | 一定。P2–P6 的階段任務與 done 條件 |
| 4 | [`roadmap.md`](roadmap.md) 的「已定案，不要再重開的決策」表 | 一定 |
| 5 | [`../AGENTS.md`](../AGENTS.md) | 動手前。環境、驗證、紅線、回報格式 |
| 6 | [`project_guide.md`](project_guide.md) | 要改程式時 |
| 7 | [`reports/2026-08-15_vlm-model-survey.html`](reports/2026-08-15_vlm-model-survey.html) | 想知道選型的完整推理時。⚠ 它有部分內容已被上面第 2 份修正（見 §7） |
| 8 | [`dev_log.md`](dev_log.md) | 只在需要歷史脈絡時，**用關鍵字搜，不要整份讀**（900+ 行） |

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
uv run pytest                 # 基線 46 passed
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

## 6. 下一步：P2（合成資料 pipeline）

計畫書 §5 的 P2 原文照做，但**重心依 §4.1 調整**：

1. `src/core/vl_models/render_puzzle.py` — LinkedIn 風格 renderer（PIL，專案已依賴）。
   **優先把牆畫對**：粗黑短棒、壓在格線上、與格線有明確對比。
2. `src/core/vl_models/dataset_builder.py` — 出題 → 渲染 → 增強 → HF datasets 格式。
3. **牆數分布改成 0–16**（出題器預設 2–5 與真實不符）。
   ⚠ `src/core/puzzle_generation/` 是**與 RL track 共用的模組，只讀不改**；要改先提出。
   合成時用參數覆寫，不要動預設值。
4. 增強：尺寸 5×5–8×8、亮／暗色、±2° 旋轉、JPEG 60–95、縮放、有無按鈕、2–3 種字型。
5. **seed 一律落盤**。

**Done 條件**：8,000 張圖＋標籤；隨機抽 20 張人眼比對真實截圖；所有標籤反向通過 solver 驗證可解。

之後才是 P3（真實驗證集）→ P4（SFT）→ P5（整合）→ P6（一步到位 endpoint）。

### 微調（P4）的兩個非預設決定

- **順序**：成本非硬限制 → **先訓 Qwen3.5-4B**（`Qwen3_5_(4B)_Vision.ipynb`），它只剩牆要學；
  **只能用免費層 → 訓 Gemma 4 E4B**（`Gemma4_(E4B)-Vision.ipynb`，QLoRA 10GB）。
  原因是 Unsloth 官方明講不建議對 Qwen3.5 做 QLoRA，而免費 T4 沒有 bf16。
- **視覺層消融要優先做**：Unsloth 建議先 `finetune_vision_layers = False` 省記憶體，
  但本任務的失敗是**純視覺的**，凍住視覺層很可能什麼都學不到。
  另：官方註明 Gemma 4 E2B/E4B 多模態訓練 **loss 13–15 是正常的**，不要誤判成發散。

---

## 7. 陷阱清單（都是實際踩過的）

| 陷阱 | 說明 |
|---|---|
| **`uv` 路徑** | repo 根 `.venv` 是 py3.9 devtools。一律 `cd linkedin-zip-challenge` 再 `uv run` |
| **`seed` 不保證決定性** | `gemma4:e4b` Q4 冷、暖兩次結果不同。**對照實驗要多次重複取平均**，不能用單次下結論 |
| **牆 F1 會被無牆題灌水** | `puzzle_02`／`puzzle_06` 無牆，預測 0 道就白拿 F1 = 1.0，把平均從 0.268 抬到 0.512。**看 `mean_wall_f1_walled_only`** |
| **few-shot 有洩題** | `puzzle_01`–`03` 的答案就寫在 prompt 裡。它們的成績**不能當泛化證據**；目前 2/6 相符中只有 `puzzle_05` 是真的 |
| **不要拿評估圖當 prompt 範例** | 7×7 範例是**合成的**（seed 20260815），刻意不用 `puzzle_04`／`06` |
| **survey 會過期得很快** | 前一份報告寫「Qwen3.8 權重未上架」，**一天內就上架了**。版本宣稱一律回一手來源重查，方法見報告 §9.4 |
| **改 prompt 要開新變體** | baseline prompt（`final_puzzle_parser.build_puzzle_prompt()`）**已凍結**，報告數字都是對著它量的。新想法加進 `prompt_variants.py` 並用 `--prompt` 切換 |
| **P1 的 GPU 峰值不可用** | 連續跑兩個模型時前一個仍常駐，數字被污染。**VRAM 只採用報告 §2 的 P0 數字** |

---

## 8. 待本人決定（agent 不要自己決定）

1. **微調要不要付 Colab CU**——決定先訓 Qwen3.5-4B（付費 L4）還是 Gemma 4 E4B（免費 T4）。
2. **要不要 push 到 origin**——目前只有本機 commit，尚未 push。
3. **新增套件**一律由本人 `uv add`，agent 只負責說要裝什麼、為什麼。
   `pyproject.toml`／`uv.lock` 是與 RL track 的**共用衝突點**。

---

## 9. 這條 track 動過的檔案

| 檔案 | 說明 |
|---|---|
| `src/core/vl_models/benchmark.py` | **新增**。四層指標量測工具，兩種傳輸層 |
| `src/core/vl_models/prompt_variants.py` | **新增**。prompt 變體（`sized`），baseline 不動 |
| `ai-collab/reports/2026-08-15_vl-p0-p1-baseline.html` | **新增**。完整報告 |
| `ai-collab/reports/artifacts/` | **新增**。所有原始量測資料 |
| `ai-collab/handover-vlm-parser.md` | **新增**。本檔（2026-08-15 由 `handover.md` 改名，因為 RL track 也有自己的交接文件） |
| `ai-collab/roadmap.md`／`dev_log.md` | 更新 |
| `pyproject.toml`／`uv.lock` | 相依修正，理由見 `build(zip)` commit |

**沒有動到**：`src/core/rl/`（RL track 的地盤）、`src/core/puzzle_generation/`、`src/core/utils.py`（共用模組）、
`src/core/solvers/`、`src/app/`、`src/ui/`。
