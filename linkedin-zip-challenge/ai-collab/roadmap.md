# 現況與下一步 — linkedin-zip-challenge

> **新 session 的第一站。** 每次工作告一段落就更新這裡（現況一句話、下一步順序、進度日誌加一列）。
> 架構與啟動方式看 [project_guide.md](project_guide.md)、完整開發歷程看 [dev_log.md](dev_log.md)、規範看 [../AGENTS.md](../AGENTS.md)。
> 最後更新：2026-08-22

## 現況（2026-08-08）

核心功能**已完成且可跑**：9 種 solver、FastAPI 後端、Gradio 主控台、Svelte WYSIWYG 編輯器、程序化出題、GIF／PNG 視覺化、Docker 雙環境。

> **本專案自 2025-10-30 之後停了約 9 個月**，2026-08-08 重建 AI 協作骨架（本檔＋`AGENTS.md`＋`project_guide.md`＋`commands.txt`）並**完成端到端復原驗證**：
> `uv sync` 乾淨（resolved 190／audited 172）、Python **3.11.13**、`uv run pytest` **46 passed in 8.10s**、`ruff` 全綠；
> 服務實際起來打過 API（三種 solver 對 `puzzle_01` 都回傳與 ground truth **逐格相同的 36 步路徑**），
> Gradio `/ui`、Svelte `/svelte-ui`、Swagger `/docs` 三個頁面都用 Chrome headless 驗過渲染。
> **基線是好的，可以直接接著開發。**

| 項目 | 狀態 |
|------|------|
| 精確解 solver（DFS／A\*(heapq)／A\*(SortedList)／CP-SAT） | ✅ 完成，有 ground-truth 單元測試 |
| 啟發式 solver（Monte Carlo／SA／GA／Tabu／PSO／ACO） | ✅ 完成，各有 smoke test |
| Fitness function（`calculate_fitness_score`） | ✅ 完成，含非連續路徑懲罰 |
| FastAPI 後端（`/api/solver/solve`、`/api/echo`） | ✅ 完成 |
| Gradio 主控台（`/ui`，4 分頁） | ✅ 完成：出題／naive 解題／互動解題／Echo |
| Svelte 互動編輯器（`/svelte-ui`，Canvas WYSIWYG） | ✅ 完成，已整合進 FastAPI 靜態掛載 |
| 程序化出題（隨機回溯 Hamiltonian path ＋ 多核） | ✅ 完成 |
| Docker 雙環境（dev hot-reload／prod multi-stage） | ✅ 完成，2025-10-30 驗證過 |
| **API 只掛了 3/9 種 solver** | ⚠ 已知落差（`src/app/routers/solver.py` 的 `SOLVERS` 只有 DFS／A\*(heapq)／CP-SAT）；**Gradio 與 Svelte 兩邊的下拉選單也同樣只有 3 種**（2026-08-08 實測確認） |
| Swagger `/docs` 的 Echo 端點重複兩份 | 🐞 小 bug：`main.py` 掛 router 時給 `tags=["Echo"]`，而 `echo.py` 的 router 自帶 `tags=["echo"]`，FastAPI 合併後產生兩個群組 |
| Svelte UI 的 Instructions 顯示原始 Markdown | 🐞 小 bug：`**middle**`／`**border**` 直接印出星號，該處沒有走 Markdown 渲染 |
| RL solver（`src/core/rl/`） | ⏸ **刻意暫停**（2025-10-15，見下方決策） |
| VL 圖片解析（`src/core/vl_models/`） | ✅ **讀圖已完成（2026-08-22 P4c）**：微調後合成 held-out **200/200 全項滿分**。⚠ 但成果是 Drive 上的 LoRA adapter，**還沒接進產品**——待 P4d 匯出 ＋ P5 Gradio 分頁 ＋ P6 `/api/vision/solve` |
| 環境復原驗證（9 個月未動） | ✅ **2026-08-08 完成**：46 tests passed、ruff 全綠 |

## 下一步

> **★ 2026-08-08 本人定案的執行順序：先做 #2（VLM），再做 #3（RL），之後才換 `board-game-rl`。**
> 本專案已被指定為當前的 side project 主菜（取代原排的 Transformer 0→1 教材）。
> **#1 明確被跳過**——它最快見效，但本人選擇先做 VLM／RL。這是明示決定，開工時直接從 #2 起手，
> 不必再重提 #1；要順手做由本人開口。下面的編號維持技術優先序，不代表執行順序。

1. **把 9 種 solver 全部掛進 API**（技術上最划算，但**本人已決定暫緩**）
   - 為什麼：`src/core/solvers/` 有 9 種實作、每種都有測試，但 `src/app/routers/solver.py` 的 `SOLVERS` dict 只暴露 3 種。
     2026-08-08 實測佐證：打 `POST /api/solver/solve` 指定 `Simulated Annealing` → **404 `Solver 'Simulated Annealing' not found.`**；
     Svelte 前端 bundle 內的下拉選項字串也只有 `DFS`／`A* (heapq)`／`CP-SAT`。**六種啟發式解法目前完全無法從介面觸及。**
   - 注意：啟發式 solver 需要 `attempts` 參數，API schema（`src/app/schemas/solver.py`）要一併擴充，且要考慮逾時（同步阻塞可能拖很久）。
     改完 Svelte 的下拉要重新 `npm run build` 才會反映。
   - Done 條件：`SOLVERS` 含全部 9 種、schema 支援 `attempts`、Gradio 與 Svelte 兩邊下拉都可選、新增對應 API 測試且全綠。

2. **VL 圖片解析整合進主流程** ← **★ 讀圖已達標（P4c 完成）；剩下的是接進產品：P4d → P5 → P6**
   - **✅ P5 程式骨幹已完成（2026-08-22）**：`backends.py`（傳輸層唯一正本）、`puzzle_parser.py`（正式 parser）、
     `prompt_baseline.py`（凍結 prompt 搬出 scratchpad，雜湊釘住）、`src/core/tests/vl_models/`（39 個 mock 測試）。
     測試 76 → **115 passed, 8 xfailed**，ruff 全綠。
   - **★ 修掉一個會讓 P4 比較失效的缺陷**：關思考的開關**從來沒接到正式流程走的那條傳輸層**。
     根因不是「忘了傳」，是**兩個 HTTP 介面吃的旋鈕不同**——`/api/chat` 吃 `think`，
     `/v1` **完全忽略 `think`**、要 `reasoning_effort="none"`。實測 `qwen3.5:4b-q8_0`：
     修前 `pydantic-ai` **66.5s／JSON 0/2** vs `native` 4.1s／2/2；修後 **5.5s／2/2**，兩條一致。
     **P1 報告的好數字全是量在 native 上的，而正式 parser 走的是另一條。**
   - **★ 量化不是這個任務的限制**：`qwen3.5:4b-q8_0` 峰值僅 **7,992 MiB / 16,376 MiB**，
     4B 連 F16（約 8GB 權重）都放得下 ⇒ **沒有「為了塞進去而被迫量化」的壓力**，部署階段可把量化這個變數消掉。
   - **✅ P2 前半已完成（2026-08-22）**：`render_puzzle.py`（LinkedIn 風格，**格線淺灰／牆粗黑**）、
     `schema.py`（標籤 ＝ 推論格式，不再需要轉換）、`dataset_builder.py`（自己抽 0–12 道牆、
     增強、CP-SAT 驗證、SHA-256 可驗證）。測試 115 → **136 passed**。
     **★ 範圍縮到 6×6**（本人 2026-08-22 決定），但 renderer／builder 都吃 size 清單，之後要加只是一個旗標。
   - **★ 舊資料集不能拿來訓練（2026-08-22 實測 4,000 對）**：標籤正確（60/60 可解），但
     **格盤 100% 都是 6×6**、**牆只有 2–5 道**、**零障礙格**，而且**牆和格線同為黑色**——
     它教的辨識線索在真實截圖上不存在。**可當 P4a 煙霧測試的輸入，不可當正式訓練集。**
   - **★ 字型可攜性已解**：`ImageFont.load_default(size=N)` 回傳 Pillow 內建的 Aileron 可縮放字型，
     不依賴系統字型 ⇒ 之前「Colab 上會畫出不同的圖」的風險解除。
   - **★ 2025-10 真的跑過一輪微調**，資料與權重**只在 Google Drive**（`colab_finetune/`），
     本機全碟搜尋無副本。`trained_models/` 裡是什麼**尚未確認**。
   - **★ Colab 付費已定案（2026-08-22）** ⇒ 走 **Qwen3.5-4B ＋ L4 ＋ bf16 LoRA**。
   - **✅ P4a 煙霧測試已完成（2026-08-22）**：L4 實測 **7.54 s/step**、峰值 VRAM 16.6/22.0 GiB、
     loss 1.011 → 0.0019、adapter 168MB 存到 Drive。**1 epoch（8,000 筆）＝ 2.09 小時 ＝ 3.2 CU。**
     視覺層確認有訓到（`visual` 96/96 `lora_B` 非零，且動得比語言層大）。
   - **★ 抓到一個會讓 P4c 比較失效的缺陷**：訓練與推論的 prompt 渲染不一致（① thinking 區塊
     ② text/image 順序）。同一個 adapter、同樣 4 張 held-out，只換 prompt：
     **修好 = JSON 4/4、版面全對、牆 F1 0.958（3 題每道牆都對）；壞掉 = 0/3 完全不可用。**
     修法 `build_inference_prompt()` 從**訓練那條渲染路徑**推導，不依賴人工同步兩邊的參數。
   - **✅✅ P4c 已完成（2026-08-22）——完整結果見
     [reports/2026-08-22_vl-p4c-results.md](reports/2026-08-22_vl-p4c-results.md)**：
     975 步 / 1.56 h / 5.77 s/step / 峰值 VRAM 20.90 of 22.03 GiB / 約 2.4 CU；
     **200 筆合成 held-out 四層指標全部 1.000**（JSON 200/200、逐格 1.000、牆 F1 1.000、端到端 200/200），
     連 24 題 12 道牆的都一道不差。滿分已通過**五項對抗性檢查**（切分正確、圖片位元組零重複、
     生成時間 200 個相異值、獨立重算一致）。
   - **★ 抓到一個會讓 P4c 數字失真的洩題**：原本打算拿 120 筆的 `smoke_6x6` 當 held-out，
     實測它與訓練用的 `main_6x6` **渲染 recipe 120/120 相同、標籤 82/120 相同**——
     兩包 seed 只差 1，而 `draw_recipe` 用 `random.Random(seed + index)`，等於整包位移一格。
     改成從 8,000 的尾巴切 200 筆當 held-out，訓練 7,800 筆。
   - **⚠ 評估集已飽和**：全部 1.000 ⇒ 這把尺再也量不出差異，**後續任何改動在它上面都會是 1.000**。
     要重獲鑑別力只能把合成資料變難（視覺雜訊、多種渲染風格、模擬截圖失真、更大盤面）。
   - **⚠ 那個 200/200 現在還碰不到**：成果是 Drive 上的 LoRA adapter，
     `puzzle_parser.parse_puzzle_image()` 今天走的仍是 Ollama 上**未微調**的模型。
   - **下一步：P4d（匯出，讓本機用得到）→ P5（Gradio 上傳分頁）→ P6（`/api/vision/solve`）。**
     細節與 done 條件在 [handover-vlm-parser.md](handover-vlm-parser.md) §6。
   - 📋 **接手請直接讀計畫書：[plans/2026-08-15_track-vlm-parser.md](plans/2026-08-15_track-vlm-parser.md)**
     （worktree 環境建置、P0–P6 分階段 done 條件、與 RL track 的協作約定）
   - ✅ **P0＋P1 已完成（2026-08-15，分支 `feat/vlm-parser`，worktree `zip-vlm`）**：
     實測報告 [reports/2026-08-15_vl-p0-p1-baseline.html](reports/2026-08-15_vl-p0-p1-baseline.html)；
     量測工具 `src/core/vl_models/benchmark.py`；原始資料 `reports/artifacts/`。
     - 部署路徑打通：容器 Ollama 0.16.1 → **0.32.13**，四種模型／量化組合**全部 100% 上 GPU**（峰值最高 9582 MiB）
       → **16GB 顯卡對 4B 級 Q8 不是瓶頸**，原風險 #5 的推論端解除。
     - **未微調 baseline：端到端最好只有 1/6**（`qwen3.5:4b-q8_0` ＋關閉思考），`gemma4:e4b` 為 **0/6**。P2–P4 有明確必要性。
     - ★ **方向修正：剩下的問題幾乎純粹是「牆」。** 最佳設定的逐格 0.924、號碼召回 0.910、格盤尺寸 6/6 全對，
       **但牆 F1 只有 0.410**。**P2 合成資料的重心要從「全面模仿」收斂到「把牆畫對」**（粗細、壓在格線上的位置、對比、0–16 數量分布）。
       且**牆的偽陽性與漏檢一樣致命**——`puzzle_03` 真牆 4/4 全中，卻因多幻覺 2 道而無解。
     - ★ **兩個零訓練成本的介入已完成並生效**：①關閉思考 ②加入格盤尺寸指示＋合成 7×7 範例（`--prompt sized`）。
       `qwen3.5:4b-q8_0` 由 JSON 3/6、逐格 —、延遲 44.5s 推進到 **JSON 6/6、尺寸 6/6、逐格 0.961、端到端相符 2/6、5.4s**。
       **但同樣兩個介入都讓 `gemma4:e4b` 變差**（後者甚至把真 6×6 過度矯正成 7×7）→ **設定要逐模型實測，不可共用**。
     - ⚠ **`seed` ＋ `temperature=0` 不保證決定性**（`gemma4:e4b` Q4 冷、暖兩次結果不同）→ 對照實驗要多次重複取平均。
     - ⚠ **`uv add` 在本專案會解析失敗**（cu121 index 排在 PyPI 之前，而 `index-strategy` 寫在只對 `uv pip` 生效的
       `[tool.uv.pip]`）。暫以 `uv add --index-strategy unsafe-best-match` 繞過；**RL track 動 torch 時會踩到同一顆雷**。
   - **方案已定案（2026-08-15，報告 v2）**：見 [reports/2026-08-15_vlm-model-survey.html](reports/2026-08-15_vlm-model-survey.html)
     ——選型分流（**免費 Colab T4 → Gemma 4 E4B QLoRA；付費 L4／A100 → Qwen3.5-4B bf16 LoRA**，
     兩者同資料各訓一輪做對照）、合成資料 pipeline、Unsloth 生態（250+ notebook）、OCR 專用模型評估、
     本機部署地雷（unsloth#3899、ollama#14730）、八階段執行計畫與 done 條件。**開工從該報告的 P0（部署煙霧測試）起手。**
   - **關鍵限制**：免費 T4 是 Turing 架構、不支援 bf16，而 Unsloth 不建議對 Qwen3.5 用 QLoRA
     → 想用免費層就走 Gemma 4。本機 16GB 可訓到 4B（bf16 LoRA 10GB），9B（22GB）必須上雲；
     **但 9B 的 Q4 推論只吃約 6GB，本機部署不是瓶頸**。
   - 現況：`src/core/vl_models/` 是實驗腳本堆，技術路線已驗證（見下方決策），但沒有正式的 parser 進 API／UI。
   - 起手處：`final_puzzle_parser.py`／`parser.py`／`prompts.py` 是最接近成品的三支；
     `experiment_*.py` 是驗證用的 scratchpad，不要當生產程式碼改。
   - ⚠ 需要本機 Ollama 跑得起來（`ollama_model_name`／`ollama_provider_url` 在 `.env`，不進版控）。
     模型不在時要有明確的失敗訊息，不要讓 API 靜默壞掉。
   - Done 條件：一個穩定的 `image → Puzzle dict` 函式 ＋ 單元測試（VL 呼叫要能 mock，測試不依賴 Ollama）
     ＋ Gradio 上傳分頁；實驗腳本標明為 scratchpad。

3. **RL 重啟前置研究** ← **★ VLM 之後接這個（可與 VLM 交錯進行：VLM 等 Colab 時本機推 RL）**
   - 🤝 **接手第一站：[handover-rl-solver.md](handover-rl-solver.md)**——自足的交接文件（環境建置、已驗證事實、
     已定案決策、程式地圖、下一步 A2 的具體做法、陷阱清單）。讀完那一份就能動手。
   - 📋 作戰計畫在 [plans/2026-08-15_track-rl-solver.md](plans/2026-08-15_track-rl-solver.md)
     （分階段 done 條件、協作約定；**§4 有 2026-08-15 的 curriculum 修訂說明**）
   - **✅ Done 條件已達成（2026-08-15）**：[reports/2026-08-15_rl-restart-plan.html](reports/2026-08-15_rl-restart-plan.html)
     ——含路線 A（action masking ＋ MaskablePPO ＋ curriculum）與路線 B（GRPO/GSPO，用既有 solver 當 verifier）。
   - **✅ A0 環境健全性已完成（2026-08-15，分支 `feat/rl-masked-ppo`）**：
     [reports/2026-08-15_a0-env-v1-findings.md](reports/2026-08-15_a0-env-v1-findings.md)
     ——**env v1 不是難學，是「餵標準答案也不會過關」**：合法一筆畫解重播 **0/7 終止**（`reset()` 把起點當成待收集的
     waypoint 1，而收集判定只在移動後執行，合法解不重踩起點 ⇒ 索引永遠停在 0）；同一條解答前面插一步「踩回起點」的
     **非法**繞路則 **6/6 終止並拿 +999.01**。**v1 的獎勵與 Zip 規則反相關，最高分策略在定義上就是作弊。**
     這修正了 restart plan §2.4：合法正樣本的機率不是「趨近於零」，是**恰好為零**。
     **2-cycle 假說實驗成立**（8 步只有 2 個相異觀測；2 狀態確定性策略跑 69 步到 truncated 都沒逃出）⇒ v2 設計前提不必重審。
     附帶發現：出題器在奇數盤有 parity 失敗（5×5 起點掃描 偶數色 13/13 成功、奇數色 0/12），約 2.5% 的 seed 會回 `None`，
     **A1 的資料生成迴圈必須換 seed 重試**。`rl_env.py` 與舊 checkpoint 未動（留作 v2 對照）。
   - **★ 2026-08-15 本人拍板兩件事**（已同步進計畫書）：
     ① **curriculum 改成「全程一筆畫 ＋ 反向 curriculum」**，不再走「先允許倒車、三階段收緊」。
     理由：一筆畫在構造上必定可解（出題器先畫 Hamiltonian path 再挖題）；**禁止重踩 ⇒ 2-cycle 定義上不可能發生**，
     `visit_count`／`visit_recency` 兩個 channel 直接砍掉；原案 Phase 1/2 的「成功」不是合法 Zip 解，
     而反向 curriculum 本來就能解稀疏訊號問題。備案（部分覆蓋給分 → 才考慮開放倒車）由 A1 量到的死路率決定。
     ② **相依走選項 B**：`uv add sb3-contrib==2.7.1`，**不動 `torch==2.4.1+cu121`**。
     查證：contrib 2.7.1 只要求 SB3≥2.7.0，而 SB3 2.7.0／2.8.0 都只要求 `torch>=2.3`（SB3 2.9 才要 ≥2.8）；
     wheel 內確認有 `ppo_mask` 與 `MaskablePPO`。**待本人執行 `uv add`；A1 不需要它。**
   - **✅ A1 已完成（2026-08-15）**：`src/core/rl/rl_env_v2.py`（一筆畫 env ＋ action masking ＋ 反向 curriculum，21 個測試全過，
     **合法性定義對齊 `dfs.py:96-105`**）、`generate_dataset_v2.py`（保留 solution path，舊腳本 `generate_rl_dataset.py:59` 會丟掉它）、
     `baselines.py`。**5,100 題資料集已生成**（4/5/6 各 1700，train/val/test = 8:1:1）。
     **生成成本修掉 18 倍**：出題器預設 `timeout_per_attempt=20s` 都花在證明「起點 parity 不對所以無解」，
     實測成功的搜尋 ≤0.415s 就完成 → 改 0.5s 後，5,100 題從預估 23 小時降到 **45 秒**；7×7 也只要 35 秒／100 題（**已不是瓶頸**）。
     這是呼叫端參數，沒有改共用模組。
   - **Baseline 已落盤**（510 題 held-out × 20 局，`logs/rl_baselines/`）：
     masked random **4×4 8.8%／5×5 0.9%／6×6 0.0%**；greedy **10.2%／3.7%／0.8%**；失敗中 90–100% 是死路。
     **greedy 就是距離型 shaping 的天花板，6×6 就崩掉——等於用實驗證實了報告 §2.2**（原本只是靜態論證）。
   - **相依已安裝**：`sb3-contrib==2.7.1`（需加 `--index-strategy unsafe-best-match`，因為專案的 index 策略寫在 `[tool.uv.pip]`，`uv add` 不吃）。
     裝完確認 `torch 2.4.1+cu121` 與 SB3 2.7.0 都沒被動到、`MaskablePPO` 可 import、測試全綠。
   - **下一步是 A2**（4×4 ＋ 反向 curriculum ＋ MaskablePPO 訓練），這是第一個真的會訓練的階段。
   - **根因已升級為機制層解釋**：`ch_path` 是二值、步數不在觀測裡 → 在兩個已訪格間震盪時觀測序列變成 `o_A, o_B, o_A…`，
     確定性策略必然卡死；非法移動則是更退化的單點迴圈。**所以不是調 reward 權重的問題**。
     另外觀測只給「下一個」waypoint，長程規劃在資訊上本來就不可能。
   - **實驗已重新設計（報告 v2）**：觀測改 9 channel（含 **visit 次數**）＋6 純量（含**已用步數比例**，
     嚴格單調 → 數學上排除觀測循環）；reward 改**冰湖式**（成功 +1、其餘 0，「越快越好」由 γ=0.99 表達，
     不用每步扣分）；位能從「距離下一個號碼」換成「**覆蓋率**」（與真目標同構）；
     三階段 curriculum：**寬鬆允許倒車 → 倒車開始收費 → 一筆畫（mask 已訪格）**。
   - **開工前要拍板的相依決策**：`MaskablePPO` 在 `sb3-contrib`，其最新版需 SB3≥2.9 而 SB3 2.9 需 `torch>=2.8`，
     與本專案鎖的 `torch==2.4.1+cu121` 衝突。三個選項（升 torch／找舊版 contrib／自寫 masked PPO）見報告 §5.2。
   - 先讀 `../more_simple_reinforcement_learning/` 的 DQN 與 PPO 章節，再看 AlphaGo／AlphaZero 架構。

4. **測試報告目錄整理**（雜務）
   - `run_tests.bat` 會把報告寫進 `src/core/tests/reports/`，堆久了會亂。決定要不要納入 `.gitignore`。

## 已定案，不要再重開的決策

| 決策 | 理由 |
|------|------|
| **RL 暫停，不是放棄** | 2025-10-15。根因是 **deterministic policy loop**：距離型 reward shaping 造成「獎勵陷阱」，即使把權重從 0.1 降到 0.01 仍會讓策略卡在小迴圈。訓練期看起來好只是 ε-greedy 的隨機性意外把 agent 撞出迴圈。**不要再靠調 reward 權重硬解**，要先補理論 |
| **模型世代選擇由「尺寸」決定，不是由「效能」決定** | 2026-08-15 一手查證（Ollama registry 探測＋library 頁）：**Qwen3.6 最小 27B、Qwen3.7 無開放權重、Qwen3.8 只有 27B**，27B 在 Q4 約 17GB **超過本機 16GB**；**Qwen3.5 是唯一有 0.8B／2B／4B／9B 小尺寸的世代**。Gemma 4 同理——vision 微調官方只支援 E2B／E4B。**所以不是「3.5 比較好」，是新世代沒出跑得動的尺寸。** ⚠ 前一份 survey 報告寫「Qwen3.8 權重尚未上架」，該權重已於 2026-08-14/15 上架（僅 27B），結論不變但該句已過期 |
| **微調順序：成本非硬限制 → 先訓 Qwen3.5-4B；只能用免費層 → 訓 Gemma 4 E4B** | 2026-08-15 實測後修訂（原本一律推 Gemma 4）。兩者都有官方 Unsloth vision notebook（`Qwen3_5_(4B)_Vision.ipynb`／`Gemma4_(E4B)-Vision.ipynb`），差別全在**起點與穩定度**：Qwen 未微調就已尺寸 6/6、逐格 0.961，**只剩「牆」要學**；Gemma 得同時學會尺寸、數字、牆。且**兩次零成本介入（關思考、加尺寸指示）都是幫 Qwen、害 Gemma**——Gemma 甚至把真 6×6 過度矯正成 7×7，對 prompt 擾動不穩定。**唯一翻轉條件**：Qwen3.5 的 QLoRA 被 Unsloth 官方勸退、只剩 bf16 LoRA，而免費 T4 無 bf16 → 想免費就只能選 Gemma 4 E4B（QLoRA 10GB） |
| **未微調門檻 ＝ `qwen3.5:4b-q8_0` ＋ 關思考 ＋ `--prompt sized`** | 2026-08-15。該設定六張圖達 JSON 6/6、尺寸 6/6、逐格 0.961、號碼 0.917、牆 F1(有牆) 0.438、端到端相符 2/6、平均 5.4s。**停損規則：微調後若端到端贏不過它，代表該換家族或換方法，而不是加訓練量。** ⚠ 相符的 2 題中 `puzzle_03` 屬洩題（答案在 few-shot 裡），**真正具泛化意義的只有 `puzzle_05` 1 題** |
| **視覺層消融要優先做，不照 Unsloth 預設** | 2026-08-15。Unsloth 建議先 `finetune_vision_layers = False` 省記憶體，但本任務的失敗模式是**純視覺的**（數字與版面早已讀對，錯的是壓在格線上的細黑牆棒）→ 凍住視覺層很可能什麼都學不到。另：官方註明 Gemma 4 E2B/E4B 多模態訓練 **loss 13–15 是正常的**（純文字版才 1–3），不知道會誤判成發散 |
| **`think` 開關逐模型實測，不可跨模型沿用** | 2026-08-15 實測：關閉思考讓 `qwen3.5:4b-q8_0` JSON 解析率 3/6→6/6、快 5.8 倍；同一開關讓 `gemma4:e4b` 全面變差。量化選擇亦然（qwen 需 Q8，gemma4 的 Q8 反而比 Q4 差） |
| **`pydantic-ai` 釘 `==1.107.5`** | 2026-08-15。v1 維護線最新（2026-08-14）；1.2.1 缺三個已 backport 的資安修補，且只有 1.107+ 才有官方文件現行示範的 `OllamaModel`。**不上 2.x**——它 10 天內連發 8 個 minor，不是穩定目標 |
| **DiffusionGemma 只觀察，不進 VLM track** | 2026-08-15 查證：雖然真的吃圖（Google 模型卡：text/image/video 輸入），但 Q4 推論最低 **18GB > 本機 16GB**、**不在 Ollama library**（registry 404）、微調官方示範要 **A100** → 與「本機推論＋免費 T4 微調」的前提全數衝突。唯一值得追蹤處是 Unsloth 那本 **26B-A4B Sudoku GRPO** notebook，與 RL 路線 B 同構（同樣卡 A100） |
| **★★ 不做真實截圖，就用自產的合成資料** | **2026-08-22 本人明示定案**，把原本的「暫緩」升級為「不做」。P4c 之後合成 held-out 已 200/200 全滿分，讀圖告一段落。**代價要一直講清楚**：所有數字證明的是「學會了我們的 renderer」，**不證明**「看得懂 LinkedIn 截圖」。**agent 不要再提議收真實截圖**，要做由本人開口 |
| **★ 評估集飽和 ⇒ 再訓練前必須先把合成資料變難** | 2026-08-22。P4c 四層指標全部 1.000，**這把尺失去鑑別力**——視覺層消融、CoD、減少訓練量、batch 調整在它上面都會是 1.000，比不出好壞。唯一還有鑑別力的現成實驗是拿 Drive 上的 `checkpoint-200` 對同一批 held-out 再算一次（loss 第 250 步就到雜訊地板，若也滿分代表 1,600 筆就夠、本輪 4/5 訓練量白付） |
| **★ 評估一律批次生成，且推論前先 merge LoRA** | 2026-08-22 實測代價。P4c 用 batch 1 逐筆生成，**推論 3.0 CU 比訓練 2.4 CU 還貴**，本末倒置。機制：batch 1 解碼是每 token 固定成本綁死（只跑到 roofline 的 27%，344 個未 merge 的 adapter 每 token 多 688 次 kernel 發動），**批次化近乎線性加速，估 2 小時 → 15 分鐘** |
| **★ 短跑量到的 s/step 與 VRAM 都不能外推** | 2026-08-22 兩次踩到。lr=0 五步短跑量到 37.59 s/step、投影 10.18 h，實際 **5.77 s/step / 1.56 h**（差 6.5 倍，第一步吃掉全部編譯成本）；P4a 50 步的 VRAM 是 16.57 GiB (75%)，跑滿 975 步是 **20.90 GiB (94.9%)**（短跑沒抽到大圖）。**丟掉第一步再平均；別用短跑的 VRAM 決定 batch size** |
| **★ Colab 的 Drive FUSE 只在關檔時才上傳** | 2026-08-22 實測。逐行 `flush()` 只推到 FUSE 層，檔案在 drive.google.com 上**完全看不到**，斷線就全沒了——「逐行寫入所以斷線也保得住」是錯的。**寫本機 `/content`，每批用 `shutil.copy` 覆蓋到 Drive**（copy 會開檔關檔才觸發上傳）|
| **★ 現階段只追求「學會我們畫的圖」，不追真實截圖** | **2026-08-22 本人明示決定**（不是疏漏）。P3 的 30–50 張真實截圖暫緩，評估一律用**合成 held-out**。**代價要講清楚：這樣量不到 domain gap**——所有數字只證明「學會這個 renderer」，不證明「看得懂 LinkedIn 截圖」。要主張後者，P3 是唯一途徑 |
| **★ 兩個資料集的 seed 差 1 ＝ 同一批資料，不是兩批** | 2026-08-22 實測。`dataset_builder.draw_recipe` 用 `random.Random(seed + index)`，所以 seed 差 1 的兩包是同一個亂數流**位移一格**：`smoke_6x6`(20260823) 與 `main_6x6`(20260822) **120/120 渲染 recipe 相同、82/120 標籤相同**（只有 `generate_puzzle` 的 wall-clock 不決定性讓 38 筆長得不一樣）。**P4c 的 held-out 因此改成從同一包切 disjoint 的 200 筆**，不另外生一包「新 seed」的——要真的獨立，seed 得差得夠遠，而切 slice 是零成本又保證不重疊的做法 |
| **指標只准有一份實作，Colab 只產原始輸出** | 2026-08-22。P4c 的 notebook 不算任何分數，只把模型原始輸出寫成 `predictions.jsonl`；算分在本機由 `score_predictions.py` 走`puzzle_parser.parse_model_output` ＋ `benchmark.score_layout`/`score_walls` 完成。理由與傳輸層那次同構——**當初 benchmark 與 parser 各寫一份才會漂移**，在 notebook 裡重寫一份指標就是同一個錯誤換個地方犯 |
| **推論 prompt 一律從訓練那條渲染路徑推導** | 2026-08-22 實測。訓練與推論曾有**兩處**不一致（thinking 區塊、text/image 順序），代價是同一個模型「內容全對但格式全錯」。`build_inference_prompt()` 用 sentinel 切訓練渲染字串，**by construction 相符**，不靠人工同步參數——事實證明有效：我當時對訓練渲染的描述是錯的，但修法照樣正確 |
| **不要為了省時間縮訓練圖** | 2026-08-22 量過：縮到 384px 省一半（7.54 → 3.89 s/step），但 1 epoch 只值 3.2 CU（3.3% 餘額），而真實截圖是 ~920×1018 ⇒ 訓練縮圖會製造新的 train/inference 不一致。**槓桿留著，等有真實驗證集能確認牆沒被縮掉再用。** 另注意：640 完全沒省到（patch 取整），768 反而貴 33% |
| **Colab 走 VS Code 的 Colab kernel，不裝 WSL2** | 2026-08-22 實測成功：助理透過 `mcp__ide__executeCode` 直接在 Colab 的 `/content` 上執行。官方 Colab CLI 只支援 Linux/macOS（issue #12 的 `fcntl`），本來以為 Windows 非 WSL2 不可——**不必**。人只要點一次連線 |
| **微調走 Qwen3.5-4B ＋ bf16 LoRA（付費 L4）** | 2026-08-22 實測 L4：`compute_cap 8.9`、**22.03 GiB**、**bf16 原生支援**。matmul 實測 bf16 **64.11 TFLOP/s** vs fp32 12.43（5.2 倍），確認是硬體加速而非模擬。⚠ 9B 需 22GB 對上 22.03 GiB **是臨界值，不要當可行方案** |
| **`torch.cuda.is_bf16_supported()` 不可直接採信** | 2026-08-22。簽章是 `(including_emulation: bool = True)`，**免費 T4（Turing，無 bf16 硬體）也回 `True`**；問 `including_emulation=False` 才回 `False`。照預設值會得出「免費層也能訓 bf16」——跑得動但沒加速。**一律問原生** |
| **資料集的可重現單位是「產物」，不是「指令」** | 2026-08-22 實測。`generate_puzzle` 用 **wall-clock** 中止隨機回溯，被切斷的嘗試消耗的亂數量與完成的不同 ⇒ 同 seed 兩次跑，0.5s 預算下 **30 筆差 8 筆**，改 5s 仍差 3 筆且慢 3 倍（6×6 搜尋時間重尾：中位數 0.07s、最大 5.4s）。根治要把搜尋改成「以工作量計量」，那在唯讀的共用模組裡。**改成用 SHA-256 驗證產物**（`--check`），這才是真正需要的性質：確認 Colab 上那份等於本機檢查過的那份 |
| **`multiprocessing.Pool` 在本機 Windows 不能用** | 2026-08-22：32 筆 10 分鐘零產出，有沒有 CP-SAT 都一樣，`spawn` 會在每個子行程重新 import 模組。**已移除，不留會卡死的旗標**。單執行緒 200 筆／72 秒，8,000 筆約 45–50 分鐘，一次性成本可接受 |
| **傳輸層只有一份實作，且關思考的旋鈕逐介面翻譯** | 2026-08-22 實測。Ollama 的 `/api/chat` 吃 `think`，`/v1` **忽略 `think`**、只認 `reasoning_effort="none"`（9.7s／reasoning 1392 字元 → 0.9s／0）。當初 benchmark 與 parser 各寫一份傳輸，才會讓 `--no-think` 在正式路徑上變成無聲 no-op，代價是 **66.5s／JSON 0/2**。現在 `backends.py` 是唯一正本，**不要在別處重寫傳輸** |
| **VL 採「混合策略」，不用 tool-calling** | 2025-10-24 實測：`bsahane/Qwen2.5-VL-7B-Instruct` 支援 tool-calling 但**視覺模組壞掉**（回傳結構正確但空的物件）；`openbmb/minicpm-o2.6` 視覺正常但**不支援 tools**（400 Bad Request）。結論：用 `pydantic-ai` 但把 `output_type` 設成 `str`，靠 prompt engineering 要模型吐 JSON 字串再自己 parse——已驗證完全成功 |
| **Docker 雙環境（dev／prod 分開）** | 2025-10-28。`docker-compose.dev.yml` 兩容器＋volume mount 換 hot-reload；`docker-compose.yml` 單一 multi-stage 映像檔換 production 乾淨。不要合併成一個 |
| **牆一律叫 `walls`** | 曾與 `blocked_cells` 混用造成前後端資料格式不符。`walls` ＝牆，`blocked_cells`／障礙是另一回事，不可互換 |
| **lint 只用 `ruff`** | 2025-10-01。`black`＋`isort`＋`ruff` 三者互相打架造成格式化無限迴圈，已移除前兩者，`ruff` 一手包辦 lint／import 排序／格式化 |
| **`pytest` 路徑設定放 `pytest.ini`** | 用 `pythonpath = .`，不要回頭在 `conftest.py` 動 `sys.path`（那是舊做法，會在測試收集階段炸 `ModuleNotFoundError`） |
| **不進 root uv workspace** | 本專案鎖 Python 3.11 ＋ `torch==2.4.1`／cu121 自訂 index，與 repo 根的 3.9 衝突。維持獨立 `.venv`＋`uv.lock`（見 `../AGENTS.md §5`） |
| **不做「兵器庫式」solver 比較報告** | 目前無此需求；要比較請先確認目的是效能數據還是教學說明 |

## 開放問題（想到就補，不急著答）

- 啟發式 solver 掛進 API 後，逾時要怎麼處理？（同步阻塞 vs 背景任務）
- `puzzle_dataset/`／`zip_puzzles/` 已累積多批資料集且不進版控，要不要定個保留策略？
- Svelte 前端目前沒有自己的測試，值得補嗎？

## 進度日誌（摘要；完整版見 [dev_log.md](dev_log.md)）

| 日期 | 事件 |
|------|------|
| 2025-09-20 | 專案初始化：DFS 回溯解題器 ＋ pytest ＋ loguru 測試報告流程 |
| 2025-09-21 ~ 09-23 | 輸入系統重構（文字版面 → parser）、6 題 ground truth 進 `conftest.py`、GIF 動畫視覺化 |
| 2025-09-25 | A\* ＋ CP-SAT（AddCircuit ＋ dummy node）完成；solver 移進 `src/core/solvers/` |
| 2025-10-01 | 啟發式框架：fitness function ＋ Monte Carlo baseline；工具鏈統一為 `ruff` |
| 2025-10-04 | 啟發式擴充：SA（truncate-and-regrow 修好路徑不連續）、GA（no-crossover）、Tabu、PSO |
| 2025-10-08 | 程序化出題：隨機回溯 Hamiltonian path ＋ retry/decrement ＋ 內部逾時；多核資料集腳本 |
| 2025-10-12 ~ 10-13 | RL 深度除錯：確診 deterministic policy loop，兩次調 reward shaping 皆失敗 |
| 2025-10-15 | **RL 暫停**；專案轉向服務化（FastAPI ＋ Gradio） |
| 2025-10-16 | 服務化第一階段：FastAPI ＋ `pydantic-settings` ＋ routers/schemas 分層 ＋ Gradio 多分頁 |
| 2025-10-20 | Gradio 互動編輯器（WYSIWYG）、即時預覽、大量 bug 修正 |
| 2025-10-21 | `run_docker_dev.py` 一鍵啟動開發環境 |
| 2025-10-22 ~ 10-24 | VL 模型驗證：兩個模型各缺一半能力 → **混合策略**（prompt engineering 產 JSON）驗證成功 |
| 2025-10-28 | Production-ready 重構：設定集中化、DRY（`prepare_solver_input`）、Svelte 整合、Docker 雙環境 |
| 2025-10-30 | 文件精修 ＋ 本機／Docker dev／Docker prod 三種環境全部實測驗證 |
| 2026-08-08 | **建立 AI 協作骨架**：`AGENTS.md`／`CLAUDE.md`／`ai-collab/`（roadmap・project_guide・dev_log・commands），`dev_log.md` 移入 `ai-collab/`，補 `.python-version` |
| 2026-08-15 | **兩份方案報告**（純研究，未動程式）：VLM 選型與微調計畫、RL 重啟方案（含 2025-10 policy loop 的機制層根因） |
| 2026-08-22 | **VLM P2 ＋ P4a**：8,000 張合成資料集（新 renderer：格線淺灰／牆粗黑、Pillow 內建字型跨平台）、標籤 ＝ 推論 schema；Colab L4 打通（VS Code kernel，不用 WSL2）；P4a 煙霧測試 **7.54 s/step**、修好訓練／推論渲染不一致後 **held-out 牆 F1 0.958** |
| 2026-08-22 | **VLM P4c 完成，讀圖告一段落**：7,800 筆 × 1 epoch（975 步 / 1.56 h / ~2.4 CU），**200 筆合成 held-out 四層指標全 1.000**，通過五項對抗性檢查；本人定案**不做真實截圖**。⚠ 評估集已飽和；成果仍在 Drive 上待 P4d 匯出。報告見 `reports/2026-08-22_vl-p4c-results.md` |
| 2026-08-22 | **VLM P4c 備妥**：`p4c_finetune_8000.ipynb`（lazy 資料集，實測整包 materialise 要 10.7 GB / VM 只有 12.7 GB；lr=0 的不污染短跑；Drive checkpoint／resume）＋ `score_predictions.py`（離線算分，不重寫指標，19 個測試）；查出 `smoke_6x6` 與訓練集**渲染 recipe 120/120 相同、標籤 82/120 相同**，held-out 改為從 8,000 切 200 筆 |
| 2026-08-22 | **VLM P5 程式骨幹 ＋ 傳輸層缺陷修復**：`backends.py`／`puzzle_parser.py`／`prompt_baseline.py` ＋ 39 個 mock 測試；查出關思考在 `/v1` 要用 `reasoning_effort` 而非 `think`（66.5s → 5.5s、JSON 0/2 → 2/2）；補 `.env.example`、修 worktree 的 `/svelte-ui` 404；盤點出 renderer 已存在、2025-10 訓練資產只在 Drive |
