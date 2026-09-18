# LinkedIn Zip 解謎挑戰

> **狀態：2026-09-19 收尾，停止開發。** 哪些能用、怎麼驗的、還有什麼沒做，都在
> [收尾報告](./ai-collab/reports/2026-09-19_project-wrap-up.md)。

把 LinkedIn「Zip」解謎從頭做到尾：**出題**、從截圖**讀題**、以及**解題**。
三件事由同一個 FastAPI 服務提供，外加 Gradio 主控台與 Svelte 畫布編輯器，
**整套服務一行指令就起得來**。

其中兩塊是學出來的。**微調過的視覺語言模型**負責把截圖變成盤面——那是**讀題**，不是解題。
**解題本身有兩條路線，而比較它們正是這個專案的重點**：**傳統搜尋**（其中 CP-SAT 是精確解）
對上**用模仿學習訓練的神經策略**（一步一步把路走出來）。

精確解法在速度與正確性上早就贏了，所以學習型 solver 的價值在於回答
「**學習在哪裡有用、在哪裡沒用、你要怎麼知道**」。所有量測（包含推翻我們自己假設的那些）
都在 [`ai-collab/`](./ai-collab/)。

英文版說明：[`README.md`](./README.md)

---

## 遊戲規則

畫一條連續的線，把每一個可走的格子**恰好走過一次**。

*   必須覆蓋所有可走格，任何格子不能重複走。
*   必須**依序**經過編號格（1 → 2 → 3 …），並**停在最大的數字上**。
*   不能穿牆（`|` 或 `—`）。

正式地說，這是**帶順序約束的 Hamiltonian path**——一般情況下 NP-hard，
也正因為如此，在這麼小的盤面上比較「精確解法 vs 學出來的解法」才有意思。

![解題動畫](./solution.gif)

---

## 快速開始

需要 **Docker**（Docker Desktop，或有 Compose V2 的 Docker Engine）以及主機上**任何 Python 3.9 以上**——
啟動腳本只用標準函式庫。app image 約 6 GB。

```bash
git clone https://github.com/Hero0963/ml-workshop.git
cd ml-workshop/linkedin-zip-challenge
python start.py
```

`start.py` 會：從範本建立 `.env` → 起整台機器唯一的 Ollama 容器 → 建並起**這個 checkout 的** app →
**等到 API 真的回應為止** → 告訴你哪些能用、哪些不能。然後打開：

| | |
|---|---|
| Gradio 主控台 | <http://127.0.0.1:7440/ui> |
| Svelte 編輯器 | <http://127.0.0.1:7440/svelte-ui/> |
| API 文件 | <http://127.0.0.1:7440/docs> |

| 指令 | 作用 |
|---|---|
| `python start.py` | 正式環境：程式碼烤進 image |
| `python start.py --dev` | 開發環境：熱更新 ＋ Svelte dev server（`:5173`）|
| `python start.py --status` | 現在跑著什麼、少了哪些選配 |
| `python start.py --down` | 停掉並移除容器 |

**第一次建置要一兩分鐘**——作者機器上從空快取實測 88 秒（2026-09-19），大部分時間在下載 Python wheel。
之後從啟動到 API 回 200 約 10 秒。

**全新 clone 能用什麼。** 出題、編輯盤面、9 種 solver 中的 8 種**直接能用**。
另外兩樣需要**不在這個 repo 裡的模型權重**（見[模型權重](#模型權重)），缺了也會誠實降級：
RL solver 回 `503` 並指出缺哪個 checkpoint；讀截圖回 `503` 並說出缺哪個模型。
`start.py` 會先講清楚這兩件事。讀截圖另外需要 **NVIDIA GPU**；沒有的話 `start.py` 會說明，其他服務照常起來。

完整操作手冊（每個容器在幹嘛、驗收指令、失敗怎麼查）：
[`ai-collab/deployment-guide.md`](./ai-collab/deployment-guide.md)。

---

## 模型權重

兩個模型都**不在版控裡**：視覺模型有 9.1 GB；RL 策略雖然只有 14 MB，但**無法逐位元重生**
（訓練資料是程序化生成的，出題器用牆鐘逾時，重跑會得到另一包題目）。

| 模型 | 大小 | 狀態（2026-09-19）| 缺了會怎樣 |
|---|---|---|---|
| RL 策略 `bc_multi_456_e6` | 14 MB | **尚未發佈**。預計放本 repo 的 GitHub Release | `RL (behaviour cloning)` 回 503，其他 solver 正常 |
| 視覺模型 `zip-qwen35-4b-p4c:f16` | 9.1 GB（兩個 GGUF）| **尚未發佈**。預計放 Hugging Face model repo | `/api/vision/solve` 回 503 |

發佈之後各自只要下載一次：RL 檔放到 `models/rl_a2/bc_multi_456_e6/checkpoints/model_final.zip`（**不用重啟**）；
兩個 GGUF 用兩行的 Modelfile 匯入 Ollama 容器。確切指令、校驗值，以及**為什麼視覺模型不能用 `ollama run hf.co/...` 直接拉**，
都在 [`ai-collab/model-weights.md`](./ai-collab/model-weights.md)。

在那之前的後備：RL 可以自己訓練（先生成資料集，再用 GPU 訓練約 5 分鐘，見 `ai-collab/model-weights.md` §7；會是另一個模型，數字接近但不會相同）；讀圖可以把 `.env` 改成未微調的
`qwen3.5:4b-q8_0` 配 `VISION_PROMPT_VARIANT=sized`——數字與版面讀得好，但會漏牆（牆 F1 約 0.44）。

---

## 這個專案提供什麼

### 1. 出題

程序化產生器**先畫出一條 Hamiltonian path，再把題目從它身上刻出來**，所以每一題保證有解
（但**不保證唯一解**：生成的 6×6 訓練題有 87.5% 不只一條解）。
Gradio 主控台、Svelte 編輯器，以及決定性的資料集產生器（`src/core/rl/generate_dataset_v2.py`）都能出題——
後者**把解答一起存下來**，這正是後來模仿學習得以成立的原因。

### 2. 讀圖

`POST /api/vision/solve` 吃一張圖，微調過的視覺語言模型把它讀成盤面，再交給 solver 解。
回應**除了答案還帶 `warnings` 與 `solvable` 旗標**——因為「讀錯」才是真正要擔心的失敗模式，
而這兩個欄位就是你發現它的方式。

### 3. 解題

`POST /api/solver/solve` 吃盤面與 solver 名稱；`GET /api/solver/list` 列出可用的名稱。
下表 10 種 solver 中 **9 種上線**；粒子群最佳化保留實作但不開放。

---

## Solver 清單

所有 solver 都在 `src/core/solvers/`，共用同一套題目表示法（`src/core/utils.py`）。
API、截圖端點、Gradio 下拉選單與 Svelte 編輯器的清單來自**同一份 registry**（`src/core/solvers/registry.py`）；
編輯器透過 `GET /api/solver/list` 取得。

| Solver | 類型 | 已上線 | 說明 |
|---|---|---|---|
| **CP-SAT** | 精確 | ✅ | 約束求解器。最快且永遠正確——預設值 |
| **DFS** | 精確 | ✅ | 深度優先 ＋ 剪枝 |
| **A\*** | 精確 | ✅ | 同一個搜尋空間的最佳優先 |
| **RL（行為克隆）** | 學習 | ✅ **需權重** | 4×4～6×6 的神經策略。**不精確也不保證**——見下 |
| 蟻群演算法 | 啟發式 | ✅ | **不精確也不保證**——見下 |
| 基因演算法 | 啟發式 | ✅ | 同上 |
| 模擬退火 | 啟發式 | ✅ | 同上 |
| 禁忌搜尋 | 啟發式 | ✅ | 同上 |
| 蒙地卡羅 | 啟發式 | ✅ | 同上。基準線：彼此獨立的隨機走法 |
| 粒子群最佳化 | 啟發式 | ❌ | **有實作、有量測，但不上線**：它靠「交換格子」移動，會把路徑拆成不相鄰的步，6×6 幾乎解不出來（`ai-collab/reports/2026-09-19_pso-not-served.md`） |

五種上線的啟發式是**為了比較，不是為了效能**——在這麼小的盤面上 CP-SAT 每一項都贏。
它們不管有沒有解出來，都會交回「看過最好的那條路」，所以 registry 會重跑到答案通過
`src/core/solvers/verify.py` 為止，**每個請求固定 5 秒**，用完就回「could not find a solution」。
對啟發式來說這代表**它放棄了，不代表盤面無解**。預算**刻意不開成 API 參數**：各演算法計算努力的單位各不相同，
固定預算才能拿來比較（`ai-collab/reports/2026-09-12_heuristic-solvers-on-the-api.md`）。

---

## 兩個模型

### 視覺：從截圖讀出盤面

| | |
|---|---|
| **選了哪個模型** | **Qwen3.5-4B**（Apache-2.0），LoRA 微調後由 Ollama 以 `zip-qwen35-4b-p4c:f16` 服務 |
| **為什麼選它** | **有官方微調 notebook ＋ 比較有機會成功**；4B 能完整載進 16 GB 顯卡、不需 CPU offload。同一個模型的 **Q4 根本吐不出合法 JSON** ⇒ **量化比參數量更關鍵** |
| **怎麼訓練的** | 用產生器渲染的合成截圖，在 Colab L4 上跑 LoRA：**975 步、1.56 小時**，峰值 VRAM 20.9／22.0 GiB。之後 merge 回 base、轉 GGUF、匯入 Ollama |
| **解決了什麼** | 未微調的瓶頸**純粹是「牆」**：真實截圖上逐格 0.947、號碼 0.917，但**牆 F1 只有 0.438**、端到端 2/6。微調後合成 held-out 端到端 **200/200** |
| **收尾時再驗一次** | 2026-09-19 現場新生成 6 張（2–12 道牆、亮／暗主題）＋ held-out 4 張：**10/10**——版面、牆、以及能解**標準答案那張盤**的路徑三項全對 |
| **匯出代價** | **零**：同一批 200 張圖，本機與 Colab 輸出**逐位元組相同**，而且**快 6.5 倍**（34.5s → 5.3s／張）|

⚠ 目標是**讀懂這個專案自己畫的盤面**，數字證明的也是這件事；**不證明**看得懂任何 LinkedIn 截圖
（六張真實截圖端到端 5/6——太少，不足以宣稱）。合成 held-out 已經**飽和在 1.000**，分辨不出兩種做法的差別；
下一步若要繼續是**把評估變難**，不是把模型變大。

⚠ **模型 tag 與 prompt variant 必須配對**（微調 tag 要配 `VISION_PROMPT_VARIANT=finetune`）。
配錯的失敗是**安靜的**：HTTP 200，但盤面全空。

### 強化學習：一步一步把路走出來

環境是一筆畫：觀測是 **8 張 8×8 特徵平面 ＋ 一個純量向量**，動作是**四個方向**，
非法動作在策略看到之前就被 **action mask** 拿掉。獎勵是冰湖式的——解開 `+1`，其餘 0。

| | |
|---|---|
| **模型** | 三層 padded 3×3 卷積（64 channel、不做 pooling）→ 256 維特徵 → policy 與 value 兩個 head。**117 萬參數**，其中 89.7% 是攤平用的全連接層 |
| **現在怎麼訓練** | **行為克隆**。每一題都附解答 ⇒ 約 **117 萬組 `(盤面, 下一步)`** 就是現成的監督式資料集。遮罩後的 cross-entropy、**6 epochs**（訓練越久策略越尖、多樣性越少）、batch 512 ⇒ **單卡約 5.5 分鐘** |
| **它取代了什麼** | MaskablePPO ＋ 反向 curriculum：**800 萬步、約 2,000 秒**，在**每個盤面、每個推論設定**都輸給幾分鐘的監督式訓練。拿 PPO 微調克隆好的策略，單次變準、best-of-32 變差 |
| **一個模型三種盤面** | **單一策略同時服務 4×4／5×5／6×6。** 對上同資料訓練的單尺寸專用模型：單次 deterministic **三個盤面全贏**（+0.032 ~ +0.037）、best-of-32 打平、推論更便宜 |

**服務中那個策略的成績**（held-out test，1,931／2,001／2,000 題）。**best-of-32** 的意思是：最多抽樣 32 次完整嘗試、
取第一個通過驗證的——**這在這裡是合法手段**，因為 Zip 的解可以自我驗證。第一次成功就停，所以平均成本遠低於 32。

| 盤面 | greedy 對照 | 單次 deterministic | best-of-32 | 平均嘗試次數 |
|---|---|---|---|---|
| 4×4 | 0.1156 | 0.9410 | **0.9953** | 1.41 |
| 5×5 | 0.0346 | 0.7701 | 未量 | — |
| 6×6 | 0.0046 | 0.5430 | **0.9465** | 5.24 |

目標（best-of-32：4×4 ≥ 0.90、6×6 ≥ 0.85）**達成**。2026-09-19 起所有判定器都要求路徑**停在最大數字**；
這把更嚴的尺讓 6×6 best-of-32 從 0.948 變 0.9435（上表是舊尺）。

**這到底說明了什麼。** 策略大約是 greedy 的 **8 倍（4×4）到 118 倍（6×6）**，
加一點推論預算就能解掉大多數題目——但它**既不精確也不保證**，而 CP-SAT 在這兩點上都贏它。
這條 track 最誠實的總結是：**這個問題其實幾乎不需要強化學習**——
獎勵極度稀疏、完美示範免費、解可自我驗證、遮罩後平均分支只有 1.5，
**RL 擅長的三件事，這個問題一件都不需要**。**知道什麼時候「不要」用 RL，是這條 track 最扎實的收穫。**

**為什麼單次嘗試到不了 100%。** 6×6 每局約 **14 次真正的選擇**，所以 deterministic 解題率 0.543 代表
服務中的策略每次選對約 **95.8%**（`0.958 ^ 14.22 ≈ 0.543`）；要到 0.85 得 98.9%，錯誤率要砍 3.7 倍。致命的那一步通常在卡住之前好幾步就犯了：
連完美的一步前瞻都只值 +0.03。缺的是一個會說「這個局面已經解不開了」的 value——而這題和圍棋不同，
**精確解可以免費替它標答案**——以及把搜尋放進訓練迴圈。完整分析、「只准走一次」這條規則對不對、
以及接下來該試什麼的排序，在 [`ai-collab/reports/2026-09-19_rl-where-next.md`](./ai-collab/reports/2026-09-19_rl-where-next.md)。

名詞從零解釋（行為克隆、DAgger、PPO 微調、AlphaZero 式自我改進迴圈）：[`ai-collab/notes/`](./ai-collab/notes/)。

---

## 疑難排解

| 症狀 | 原因與處理 |
|---|---|
| `could not select device driver "nvidia" with capabilities: [[gpu]]` | 沒有 NVIDIA GPU 或沒裝 NVIDIA container toolkit。屬預期：`start.py` 會跳過視覺模型繼續起其他服務 |
| Apple Silicon Mac 或 ARM Linux 上建置卡在 `uv sync` | torch 2.4.1+cu121 只有 x86_64 的 Linux wheel。compose 已釘 `platform: linux/amd64`，Docker 應該會用模擬建置——很慢，而且**沒在 ARM 機器上測過** |
| `port is already allocated` | 別的 checkout 的 stack（或 `uv run` 起的 server）佔著。`python start.py --status` 看是誰；在 `.env` 換 `APP_PORT` |
| Windows：容器都 healthy，但 `http://127.0.0.1:7440` 沒回應 | `%USERPROFILE%\.wslconfig` 裡的 `networkingMode=mirrored` 會讓 Docker Desktop 的埠轉發失效（[microsoft/WSL#10494](https://github.com/microsoft/WSL/issues/10494)）。註解掉那行，再 `wsl --shutdown` |
| 讀截圖回 503 | 視覺模型不在 Ollama 裡（見[模型權重](#模型權重)），或 Ollama 還在載入——第一次呼叫約一分鐘 |
| RL solver 回 503 | `models/` 底下沒有 checkpoint（見[模型權重](#模型權重)）|

---

## 專案結構

```
linkedin-zip-challenge/
├── start.py                  # 一鍵啟動：從 clone 到服務跑起來
├── docker-compose.yml        # 正式環境的 app；每個 checkout 各一組
├── docker-compose.dev.yml    # 開發環境的 app（＋ 熱更新、Svelte dev server）
├── docker-compose.ollama.yml # 整台機器唯一的 ollama，所有 checkout 共用
├── .devcontainer/
│   ├── Dockerfile            # 多階段：先建 Svelte，再建 Python app
│   └── Dockerfile.dev        # 開發用 image（uvicorn --reload）
├── .env.example              # 設定範本；`.env` 由它產生
│
├── src/
│   ├── app/                  # FastAPI 應用
│   │   ├── main.py           # App、CORS、靜態檔掛載、Gradio 掛載
│   │   ├── routers/          # echo／solver／vision 端點
│   │   └── schemas/          # 請求與回應模型
│   ├── core/
│   │   ├── solvers/          # 傳統 solver、共用 registry、獨立裁判
│   │   ├── puzzle_generation/# 出題器（先畫路徑再刻題目）
│   │   ├── vl_models/        # 截圖 → 盤面：prompt、後端、評分
│   │   ├── rl/               # 環境、訓練、服務端
│   │   │   ├── rl_env_v2.py            # 環境：遮罩、獎勵、curriculum
│   │   │   ├── train_behaviour_cloning.py
│   │   │   ├── train_maskable_ppo.py
│   │   │   ├── train_config.py         # 所有訓練設定的唯一來源
│   │   │   └── solver_service.py       # 唯一的服務端檔案
│   │   ├── tests/            # 核心、solver、RL、出題的測試
│   │   └── utils.py          # Puzzle 型別、parser、評分、視覺化
│   ├── ui/gradio_app.py      # Gradio 主控台（API 的 Adapter）
│   ├── custom_components/    # Svelte 畫布編輯器
│   └── settings.py           # 設定的唯一來源
│
├── ai-collab/                # 開發文件（見下）
├── illustrations/            # 截圖與範例題
├── models/  datasets/  logs/ # 大型本機產物——不進版控
└── pyproject.toml            # 相依套件，由 uv.lock 鎖定
```

---

## 不用 Docker 的跑法

```bash
cd linkedin-zip-challenge
uv sync                                     # Python 3.11，由 .python-version 指定
cp .env.example .env
uv run python -m src.app.main               # http://127.0.0.1:7440/ui
```

Svelte 編輯器要先建置才會出現在 `/svelte-ui`：

```bash
cd src/custom_components/puzzle_editor/frontend && npm install && npm run build
```

讀圖需要一個提供視覺模型的 Ollama；`.env` 的 `OLLAMA_PROVIDER_URL` 指向 compose 對外開的埠，
所以只跑那一個容器就夠。

---

## 介面怎麼用

Gradio 主控台（`/ui`）一個分頁對應一種能力：

*   **Generate Puzzle**——隨機出題，可指定障礙格數量。
*   **Puzzle Solver (Naive)**——直接貼上盤面與牆的文字表示。
*   **Puzzle Solver (Interactive)**——畫布編輯器：點格子加編號、障礙、牆，然後解。
*   **Solve from Screenshot**——上傳圖片；答案會附 `warnings` 與 `solvable` 旗標。
*   **Echo Test**——確認後端還活著。

介面截圖在 [`illustrations/`](./illustrations/)。

---

## 開發

```bash
cd linkedin-zip-challenge
uv sync
uv run pytest            # 2026-09-19 實測 330 passed, 1 skipped, 8 xfailed
uv run ruff check .
```

那 8 個 `xfail` 是**刻意的**：它們釘住已退役的 v1 環境的缺陷，所以「意外通過」代表有人動了它。
那 1 個 skip 需要 `models/` 底下有 RL checkpoint（在掛了 checkpoint 的 app 容器裡會跑到：331 passed）。

要對一個跑著的服務做端到端驗收——每一種上線的 solver 都走 HTTP，回傳的路徑一律用獨立裁判判、不信 `200`——
用 [`ai-collab/reports/artifacts/wrap-up-acceptance/`](./ai-collab/reports/artifacts/wrap-up-acceptance/) 裡的腳本。

### 文件地圖

| 文件 | 內容 |
|---|---|
| [`ai-collab/reports/2026-09-19_project-wrap-up.md`](./ai-collab/reports/2026-09-19_project-wrap-up.md) | **收尾報告**：現況、驗收、還沒做的。**從這裡開始** |
| [`ai-collab/roadmap.md`](./ai-collab/roadmap.md) | 現況、已定案的決策、各 track 的歷程 |
| [`ai-collab/model-weights.md`](./ai-collab/model-weights.md) | 兩個模型的權重該放哪、怎麼安裝 |
| [`ai-collab/project_guide.md`](./ai-collab/project_guide.md) | 架構、模組職責、各環境怎麼啟動 |
| [`ai-collab/deployment-guide.md`](./ai-collab/deployment-guide.md) | Docker：每個容器在幹嘛、驗收指令、失敗排查 |
| [`ai-collab/notes/`](./ai-collab/notes/) | **做中學筆記**：方法是什麼、數字怎麼判讀、推論怎麼跑 |
| [`ai-collab/reports/`](./ai-collab/reports/) | 每個實驗一份報告——方法、數字、以及**推翻了什麼** |
| [`ai-collab/handover-rl-solver.md`](./ai-collab/handover-rl-solver.md) | 接手 RL track：帶證據的死路地圖、下一步 |
| [`ai-collab/handover-vlm-parser.md`](./ai-collab/handover-vlm-parser.md) | 接手視覺 track |
| [`ai-collab/handover-solvers.md`](./ai-collab/handover-solvers.md) | 接手 solver registry 與 API |
| [`ai-collab/dev_log.md`](./ai-collab/dev_log.md) | 完整開發日誌（很長——用搜尋的，不要整份讀）|
| [`AGENTS.md`](./AGENTS.md) | 本子專案的協作規範（給人與 AI 共用）|

---

## 未來可以做的

2026-09-19 起停止開發。接下來該做什麼、為什麼，依序列在[收尾報告](./ai-collab/reports/2026-09-19_project-wrap-up.md) §10；簡版：

*   **發佈兩個模型的權重**——唯一需要作者帳號的一步。
*   **學習型 solver**：把做一半的 ExIt 對照做完；改用單一的「步數預算」量尺（4n² 步內解開，允許重來與倒車）；
    再用精確解標註「可不可解」來訓練 value，拿它引導搜尋。
*   **視覺**：把合成評估變難（已飽和在 1.000）。
*   **把出題器當成 benchmark**，拿來測通用的 computer-use agent：無限量的新題、精確的裁判、agent 能拖曳的畫布
    （[survey](./ai-collab/reports/2026-09-19_computer-use-agents-and-zip.md)）。
