# 任務計畫書 — Track VLM：圖片解析整合進主流程

> **給接手這條 track 的 agent。這是你的作戰計畫，不是背景資料。**
> 建立日期：2026-08-15（Asia/Taipei）｜對應 roadmap 第 2 項｜姊妹 track：[track-rl-solver](2026-08-15_track-rl-solver.md)
> **技術分析與選型理由不在本檔**，在 [`../reports/2026-08-15_vlm-model-survey.html`](../reports/2026-08-15_vlm-model-survey.html)（用瀏覽器開）。
> 本檔只講：怎麼建環境、做哪些階段、每階段怎麼算完成。

---

## 0. 一句話目標

**把「使用者上傳一張 Zip 謎題截圖 → 得到可餵給 solver 的 `Puzzle` 資料」做成正式、可測試、可部署的功能**，取代目前 `src/core/vl_models/` 裡的實驗腳本堆。

**明確不做**（做了就是超出範圍）：

- ❌ 不碰 `src/core/rl/`（那是另一條 track，會衝突）
- ❌ 不做「模型端到端直接解題」（那是 RL track 的路線 B）
- ❌ 不把 9 種 solver 掛上 API（roadmap #1，本人已明示暫緩）
- ❌ 不自己 `uv add`（見 §6）

---

## 1. 開工前必讀（照順序，不要跳）

1. 本檔（你正在讀）
2. [`../roadmap.md`](../roadmap.md) — 現況與**已定案不要再重開的決策**
3. [`../../AGENTS.md`](../../AGENTS.md) — 子專案規範正本（環境、驗證、紅線、回報格式）
4. [`../reports/2026-08-15_vlm-model-survey.html`](../reports/2026-08-15_vlm-model-survey.html) — 選型與方案的完整推理
5. [`../project_guide.md`](../project_guide.md) — 架構（要動程式時再讀）
6. `../dev_log.md` — 600+ 行，**不要整份讀**，用關鍵字搜（例如 `minicpm`、`tool-calling`）

---

## 2. Worktree 環境建置 ★（新 worktree 一定會踩的坑）

新 worktree 只會拿到**進版控的檔案**。以下東西**不會跟過來**，必須手動處理：

| 項目 | 狀態 | 怎麼辦 |
|------|------|--------|
| `.env` | ❌ 不進版控 | **必須從主工作樹複製**，否則 `pydantic` 會因缺 `ollama_*` 設定而行為異常（2025-10 踩過） |
| `.venv` | ❌ 不進版控 | 各 worktree 各自 `uv sync`（會抓 `torch==2.4.1+cu121`，數 GB，第一次很慢） |
| `models/`、`logs/`、`datasets/`、`puzzle_dataset/` | ❌ 不進版控 | 新 worktree 是空的，需要就自己產 |
| `illustrations/puzzle_01..06.png` | ✅ **有**進版控 | 這 6 張真實截圖是你唯一的現成評估素材，直接可用 |
| Ollama 模型（15GB） | Docker volume | 不隨 worktree，但已 pin 成全機共用（見下） |

```powershell
# --- 由本人執行（建 worktree）---
cd D:\it_project\github_sync\ml-workshop
git worktree add ..\zip-vlm -b feat/vlm-parser main

# --- 複製不進版控的設定 ---
Copy-Item .\linkedin-zip-challenge\.env ..\zip-vlm\linkedin-zip-challenge\.env

# --- agent 從這裡開始 ---
cd ..\zip-vlm\linkedin-zip-challenge
uv sync
uv run pytest                 # 基線：2026-08-08 紀錄為 46 passed，先確認仍然全綠再往下
uv run ruff check .
```

> **`uv run` 一律先 `cd linkedin-zip-challenge`**。repo 根的 `.venv` 是 py3.9 devtools，拿來跑本專案會 import 失敗。

### 2.1 Docker / Ollama（2026-08-15 已實測可用）

```powershell
cd <worktree>\linkedin-zip-challenge
docker compose -f docker-compose.dev.yml pull ollama    # 必要：容器內是 0.16.1，太舊跑不動新模型
docker compose -f docker-compose.dev.yml up -d ollama
docker exec zip_ollama_server ollama list               # 應看到 3 個 2025-10 的舊模型
curl http://127.0.0.1:11435/api/tags                    # 應回 200
```

已驗證事實（**不要重驗，浪費時間**）：

- 容器內 `nvidia-smi` 看得到 `RTX 4070 Ti SUPER, 16376 MiB` → **GPU 直通可用**
- volume 已 pin 成 `linkedin-zip-challenge_ollama_data`，**任何 worktree 都指向同一份 15GB 模型**
- 容器名 `zip_ollama_server`、host 埠 `11435`（`OLLAMA_HOST_PORT` 可改）。
  **11434 與 `ollama_server` 這個名字被本機另一個專案佔用，不要改回去。**

> ⚠ **兩個 worktree 不可同時起 docker stack**——容器名與 host 埠是全機唯一，會撞。與 RL track 協調。

---

## 3. 現況與已驗證事實（不要重驗）

- 核心功能完整可跑：9 種 solver、FastAPI、Gradio、Svelte 編輯器、出題器、GIF/PNG 視覺化、Docker 雙環境。
- 目前的 VL 流程是**實驗腳本**：`src/core/vl_models/final_puzzle_parser.py` 用 `pydantic-ai` ＋ `output_type=str` ＋ few-shot prompt 打 Ollama，再用 regex 摳 JSON。**能跑，但沒有正式介面、沒有測試、沒進 API/UI。**
- 目標 schema 是 `SimplePuzzleOutput`：`layout`（2D 字串陣列，`"  "` 空格／`"01"` 數字／`"xx"` 障礙）＋ `walls`（相鄰格對清單）。轉換函式 `parse_puzzle_layout()` 已存在於 `src/core/utils.py`。
- 真實輸入長相（我開圖確認過 `puzzle_04.png`）：圓角網格、實心黑圓＋白數字、**壓在格線上的粗黑短棒當牆**、下方有 Undo/Hint 按鈕、可能有游標殘影。**與現有 `save_solution_as_image()` 畫出來的風格差很遠。**
- 出題器 `generate_puzzle()` 預設只產 **2–5 道牆**（`puzzle_generator.py:10-11`），但 `puzzle_04.png` 實測有 **14 道**。合成資料要修這個分布。

---

## 4. 已定案，不要重開

| 決策 | 理由 |
|------|------|
| **VL 用「混合策略」，不用 tool-calling** | 2025-10-24 實測：`bsahane/Qwen2.5-VL-7B` 支援 tools 但視覺壞掉；`minicpm-o2.6` 視覺正常但不支援 tools。結論是 `output_type=str` ＋ prompt engineering 自己 parse |
| **第一步的產物給人類玩** | 解析出來的 `Puzzle` 要能進互動介面讓人自己解，不是只拿來自動求解 |
| **牆一律叫 `walls`** | 不可與 `blocked_cells`（障礙格）混用 |
| **lint 只用 `ruff`** | 不要加回 black/isort |
| **不進 root uv workspace** | 鎖 Python 3.11 ＋ torch cu121，與根的 3.9 衝突 |
| **Ollama 容器名／埠不改回 `ollama_server`／11434** | 會與本機另一專案衝突（2026-08-15 確認） |

---

## 5. 階段任務

> 每階段做完就更新 `../roadmap.md` 與 `../dev_log.md`，不要囤到最後。

### P0 — 部署煙霧測試（0.5 天）★ 先做這個

**為什麼先做**：微調成功 ≠ 跑得起來。先確認整條「拿模型 → 本機推論」的路是通的，再投資訓練。

- [ ] `docker compose -f docker-compose.dev.yml pull ollama` 更新映像
- [ ] 從 **Ollama 官方 library** 拉未微調模型（官方模型內嵌視覺張量，繞開已知的 mmproj 匯入問題）：
      主力 `Qwen3.5-4B`，對照 `Gemma 4 E4B`
- [ ] 用 `illustrations/puzzle_01.png` 跑一次，確認模型看得懂圖
- [ ] 量：VRAM 峰值、單張延遲、Q4 vs Q8 的差異

**Done**：三個數字（VRAM／延遲／量化差異）落盤成表，寫進 `../reports/`。至少一個模型能對 `puzzle_01.png` 產生合理輸出。

### P1 — Baseline（0.5 天）

- [ ] 用**現有的** `build_puzzle_prompt()` few-shot prompt，對 `illustrations/puzzle_01..06.png` 六張全跑
- [ ] 量四層指標：① JSON 可解析率 ② 逐格準確率／wall F1 ③ **端到端**（解析結果餵 CP-SAT 能否解出與 ground truth 一致的路徑）④ 延遲

**Done**：一張 baseline 表落盤。**沒有這張表就不准進 P2**——否則之後無法證明微調有沒有用。

> 💡 **若 baseline 已經很好（例如端到端 5/6 以上），先回報本人**：P2–P4 的兩三天可能不必做，直接跳 P5 整合。

### P2 — 資料 pipeline（2 天）

- [ ] 新增 `src/core/vl_models/render_puzzle.py`：**LinkedIn 風格** renderer（PIL，專案已依賴）
      — 圓角網格、實心黑圓＋白數字、粗黑牆棒、UI chrome 可選
- [ ] 新增 `src/core/vl_models/dataset_builder.py`：出題 → 渲染 → 增強 → HF datasets 格式
- [ ] 牆數分布修正到 **0–16**（現有預設 2–5 與真實不符）
- [ ] 增強：尺寸 5×5–8×8、亮/暗色、±2° 旋轉、JPEG 60–95、縮放、有無按鈕、2–3 種字型
- [ ] **seed 落盤**（repo 規範：測資與 seed 要可追溯）

**Done**：產出 8,000 張圖＋標籤；隨機抽 20 張人眼比對真實截圖；所有標籤能反向通過 solver 驗證（生成的題目都可解）。

### P3 — 真實驗證集（1 天，人工）

- [ ] 收集 30–50 張真實 Zip 截圖 ＋ 人工標註

**Done**：入庫（大檔不進版控），標註經 solver 驗證可解。
**紅線：這批只准用來評估，絕不能進訓練集，也不要照著它逐一調 renderer 到過擬合。**

### P4 — SFT（1–2 天，Colab）

- [ ] fork Unsloth 官方 vision notebook（**不要自己寫訓練 loop**）
- [ ] 同一份資料訓兩個家族：`Qwen3.5-4B`（bf16 LoRA）與 `Gemma 4 E4B`（QLoRA）
- [ ] 消融（便宜，順手做）：`finetune_vision_layers` True vs False —— 差距直接告訴你瓶頸是不是在視覺層

**Done**：兩者四層指標**均優於 P1 baseline**，並排比較表落盤；權重存 `models/`（不進版控）。

### P5 — 整合（1–2 天）

- [ ] 新增正式 `src/core/vl_models/puzzle_parser.py`（`image → Puzzle`）
- [ ] backend 抽象：`backends/transformers_backend.py` 與 `backends/openai_compat_backend.py`，同一介面，用 `src/settings.py` 切換
- [ ] 單元測試：**VL 呼叫要能 mock，測試不依賴 Ollama 或 GPU**
- [ ] Gradio 上傳分頁
- [ ] 模型不在時要有明確錯誤訊息，**不要靜默壞掉**

**Done**：roadmap 第 2 項的 done 條件全數達成；`uv run pytest` 全綠；`experiment_*.py` 明確標為 scratchpad。

### P6 — 一步到位 endpoint（1–2 天）

- [ ] `POST /api/vision/solve`：圖片 → 解析 → CP-SAT → 回傳 `{puzzle, path, image_url, parse_warnings}`
- [ ] **回傳 `puzzle` 讓使用者能修**：接既有 Svelte WYSIWYG 編輯器，解析錯一兩格時拖兩下就好

**Done**：上傳真實截圖 → 拿到正確解答圖；解析錯誤可手動修正後重解。

---

## 6. 相依與外部資源

- **套件一律由本人 `uv add`**，agent 只負責說「要裝什麼、為什麼」。
- ⚠ **`pyproject.toml` / `uv.lock` 是兩條 track 的共用衝突點**。要動相依前先在回報中提出，**不要自己改檔**。
- Colab：Colab CLI 官方只支援 Linux/macOS（Windows 要 WSL2）；**Windows 最省事的是 VS Code 的 Colab kernel 擴充**。
- ⚠ 免費 T4 是 Turing 架構、**不支援 bf16**，而 Unsloth 不建議對 Qwen3.5 用 QLoRA → 想用免費層就走 Gemma 4。

---

## 7. 與 RL track 的協作

| 面向 | 情況 | 約定 |
|------|------|------|
| 程式碼 | 幾乎不重疊：你動 `src/core/vl_models/`、`src/app/`、`src/ui/`；RL 動 `src/core/rl/` | 各自為政 |
| **共用模組** | `src/core/utils.py`（`Puzzle`／`parse_puzzle_layout`／`calculate_fitness_score`）、`src/core/puzzle_generation/` | **兩邊都只讀不改**；真要改先提出 |
| **文件** | `roadmap.md`、`dev_log.md` 兩邊都會寫 | dev_log 各自加自己的 `## YYYY-MM-DD` 區塊；roadmap 只改自己那一項；合併衝突時**兩邊都保留** |
| **相依** | `pyproject.toml`／`uv.lock` | 序列化，由本人統一處理 |
| **Docker** | 容器名與埠全機唯一 | 不要同時起 stack |

---

## 8. 回報與文件更新義務

完成任務時要包含：**改了哪些檔**、實際跑過的指令與**輸出關鍵行**、有沒有更新 `roadmap.md` 與 `dev_log.md`、**逐項確認的 done 條件**。

- 沒跑就說沒跑，不要推測輸出、不要拿記憶中的數字當實測。
- 測試失敗**不要改測試讓它變綠**——先判斷是環境漂移、ground truth 錯、還是真的壞了。
- git：commit／push／PR **都需要當次授權**；不在 `main` 直接開發；禁 force push。
- **不永久刪除任何檔案**：要移除的移進 `../../../soft-delete/<時間戳>/<原相對路徑>` 並回報還原方式。

---

## 9. 風險（依機率排序）

1. **Domain gap**（高）：模型只認得自己畫的圖 → 靠增強與真實驗證集把關。
2. **牆的召回率低**（高）：牆最容易漏；評估要把 wall F1 單獨列出來。
3. **微調後 GGUF 匯出異常**（中）：unsloth#3899 有此回報且未載明修法 → 保留 transformers 直跑當 fallback。
4. **免費 T4 跑不動所選模型**（中）：見 §6 的 bf16 限制。
5. **本機 16GB 訓 4B 偏緊**（中）：降圖片解析度／batch 1／梯度檢查點，再不行上 Colab。
