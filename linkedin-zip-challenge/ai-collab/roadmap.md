# 現況與下一步 — linkedin-zip-challenge

> **新 session 的第一站。** 每次工作告一段落就更新這裡（現況一句話、下一步順序、進度日誌加一列）。
> 架構與啟動方式看 [project_guide.md](project_guide.md)、完整開發歷程看 [dev_log.md](dev_log.md)、規範看 [../AGENTS.md](../AGENTS.md)。
> 最後更新：2026-08-15

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
| VL 圖片解析（`src/core/vl_models/`） | 🧪 **實驗性 scratchpad**，技術路線已驗證可行但未整合進主流程 |
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

2. **VL 圖片解析整合進主流程** ← **★ 現在做這個（P0／P1 已完成，下一步 P2）**
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
   - 📋 **接手請直接讀計畫書：[plans/2026-08-15_track-rl-solver.md](plans/2026-08-15_track-rl-solver.md)**
     （worktree 環境建置、A0–A5 分階段 done 條件、**A2 前要拍板的 torch 相依決策**）
   - **✅ Done 條件已達成（2026-08-15）**：[reports/2026-08-15_rl-restart-plan.html](reports/2026-08-15_rl-restart-plan.html)
     ——含路線 A（action masking ＋ MaskablePPO ＋ curriculum）與路線 B（GRPO/GSPO，用既有 solver 當 verifier）。
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
