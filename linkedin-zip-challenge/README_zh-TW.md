# LinkedIn Zip 解謎挑戰

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
*   必須**依序**經過編號格（1 → 2 → 3 …）。
*   不能穿牆（`|` 或 `—`）。

正式地說，這是**帶順序約束的 Hamiltonian path**——一般情況下 NP-hard，
也正因為如此，在這麼小的盤面上比較「精確解法 vs 學出來的解法」才有意思。

![解題動畫](./solution.gif)

---

## 快速開始

從 clone 到服務跑起來：

```bash
git clone https://github.com/Hero0963/ml-workshop.git
cd ml-workshop/linkedin-zip-challenge
python start.py
```

`start.py` 只需要 Python 3 與**已啟動的 Docker**。它會：從範本建立 `.env` → 確認整台機器唯一的 ollama 在跑 →
建並起**這個 checkout 的** app → **等到 API 真的回應為止** → 然後告訴你哪些能用、哪些不能：

```
$ docker compose -f docker-compose.ollama.yml up -d
$ docker compose -f docker-compose.yml -p zip-app-ml-workshop up -d --build
Waiting for http://127.0.0.1:7440/api/echo/health
API is up.
Ollama is up. Models: ['zip-qwen35-4b-p4c:f16', ...]
RL solver weights found (models/rl_a2/bc_multi_456/checkpoints/model_final.zip).

  Gradio console   http://127.0.0.1:7440/ui
  Svelte editor    http://127.0.0.1:7440/svelte-ui/
  API docs         http://127.0.0.1:7440/docs
```

| 指令 | 作用 |
|---|---|
| `python start.py` | 正式環境：程式碼烤進 image |
| `python start.py --dev` | 開發環境：熱更新 ＋ Svelte dev server（`:5173`）|
| `python start.py --status` | 現在跑著什麼、少了哪些選配 |
| `python start.py --down` | 停掉並移除容器 |

**第一次建置要幾分鐘**——作者機器上實測 110 秒，大部分時間在下載 wheel；全新的機器還要加上建 Svelte 前端。
之後從啟動到 API 回 200 約 10 秒。

**兩個選配缺了會誠實降級**：沒有 Ollama 容器 ⇒ 讀截圖不能用，其餘正常；
`models/` 底下沒有 checkpoint ⇒ RL solver 回 **503**，其他 solver 不受影響。
`start.py` 會直接講缺哪一個，而不是讓你之後才踩到。

完整操作手冊（每個容器在幹嘛、五項驗收指令與實際輸出、失敗怎麼查）：
[`ai-collab/deployment-guide.md`](./ai-collab/deployment-guide.md)。

---

## 這個專案提供什麼

### 1. 出題

程序化產生器**先畫出一條 Hamiltonian path，再把題目從它身上刻出來**，所以每一題保證有解。
Gradio 主控台、Svelte 編輯器，以及決定性的資料集產生器
（`src/core/rl/generate_dataset_v2.py`）都能出題——後者**把解答一起存下來**，
這正是後來模仿學習得以成立的原因。

### 2. 讀圖

`POST /api/vision/solve` 吃一張圖，微調過的視覺語言模型把它讀成盤面，再交給 solver 解。
回應**除了答案還帶 `warnings` 與 `solvable` 旗標**——因為「讀錯」才是真正要擔心的失敗模式，
而這兩個欄位就是你發現它的方式。

### 3. 解題

`POST /api/solver/solve` 吃盤面與 solver 名稱。專案實作了 **10 種 solver**，目前 **4 種掛上 API**。

---

## Solver 清單

所有 solver 都在 `src/core/solvers/`，共用同一套題目表示法（`src/core/utils.py`）。
API、截圖端點與 Gradio 下拉選單的清單來自**同一份 registry**（`src/core/solvers/registry.py`）。

| Solver | 類型 | 已上線 | 說明 |
|---|---|---|---|
| **CP-SAT** | 精確 | ✅ | 約束求解器。最快且永遠正確——預設值 |
| **DFS** | 精確 | ✅ | 深度優先 ＋ 剪枝 |
| **A\*** | 精確 | ✅ | 同一個搜尋空間的最佳優先 |
| **RL（行為克隆）** | 學習 | ✅ | 神經策略。**不精確也不保證**——見下 |
| 蟻群演算法 | 啟發式 | — | 已實作且有測試，未掛上 API |
| 基因演算法 | 啟發式 | — | 同上 |
| 粒子群最佳化 | 啟發式 | — | 同上 |
| 模擬退火 | 啟發式 | — | 同上 |
| 禁忌搜尋 | 啟發式 | — | 同上 |
| 蒙地卡羅 | 啟發式 | — | 同上 |

六種啟發式**刻意不上線**：在這麼小的盤面上它們每一項都輸給 CP-SAT，
所以「把它們掛上去」是被記錄下來、刻意延後的項目，不是漏做。

---

## 兩個模型

### 視覺：從截圖讀出盤面

| | |
|---|---|
| **選了哪個模型** | **Qwen3.5-4B**（Q8），LoRA 微調後由 Ollama 以 `zip-qwen35-4b-p4c:f16` 服務 |
| **為什麼選它** | **有官方微調 notebook ＋ 比較有機會成功**；4B 的 Q8 能完整載進 16 GB 顯卡、不需 CPU offload（峰值 9.6 GB）。同一個模型的 **Q4 根本吐不出合法 JSON** ⇒ **量化比參數量更關鍵** |
| **怎麼訓練的** | 用產生器渲染的合成截圖，在 Colab L4 上跑 LoRA：**975 步、1.56 小時、MFU 約 42%**，峰值 VRAM 20.9／22.0 GiB。之後 merge 回 base、轉 GGUF、匯入 Ollama |
| **解決了什麼** | 未微調的瓶頸**純粹是「牆」**：真實截圖上版面 0.947、號碼 0.917，但**牆 F1 只有 0.438**、端到端 2/6。微調後在 held-out 有牆題上**牆 F1 = 1.000**、端到端 **200/200** |
| **匯出代價** | **零**：同一批 200 張圖，本機與 Colab 輸出**逐位元組相同**，而且**快 6.5 倍**（34.5s → 5.3s／張）|

⚠ held-out 是**合成資料**且指標已經**飽和在 1.000**，它已經分辨不出兩種做法的差別。
那條 track 的下一步是**把評估變難**，不是把模型變大。

⚠ **模型 tag 與 prompt variant 必須配對**（微調 tag 要配 `VISION_PROMPT_VARIANT=finetune`）。
配錯的失敗是**安靜的**：HTTP 200，但盤面全空。

### 強化學習：一步一步把路走出來

環境是一筆畫：觀測是 **8 張 8×8 特徵平面 ＋ 一個純量向量**，動作是**四個方向**，
非法動作在策略看到之前就被 **action mask** 拿掉。獎勵是冰湖式的——解開 `+1`，其餘 0。

| | |
|---|---|
| **模型** | 三層 padded 3×3 卷積（64 channel、不做 pooling）→ 256 維特徵 → policy 與 value 兩個 head。**117 萬參數**，其中 89.7% 是攤平用的全連接層 |
| **現在怎麼訓練** | **行為克隆**。每一題都附解答 ⇒ 約 **117 萬組 `(盤面, 下一步)`** 就是現成的監督式資料集。遮罩後的 cross-entropy、10 epochs、batch 512 ⇒ **單卡 7.8 分鐘** |
| **它取代了什麼** | MaskablePPO ＋ 反向 curriculum：**800 萬步、約 2,000 秒**，而且在**每個盤面、每個推論設定**都輸給 7.8 分鐘的監督式訓練 |
| **一個模型三種盤面** | **單一策略同時服務 4×4／5×5／6×6——這正是目標，而且站得住。** 對上用同資料訓練的單尺寸專用模型：單次 deterministic **三個盤面全贏**（+0.032 ~ +0.037）、best-of-32 打平、**每個盤面的推論都更便宜**、訓練成本也比三個加起來略低。而 5×5 在此之前**根本沒有模型** |

**結果**（held-out test，1,931／2,001／2,000 題）。**best-of-32** 的意思是：最多抽樣 32 次完整嘗試、
取第一個通過驗證的——**這在這裡是合法手段**，因為 Zip 的解可以自我驗證。
第一次成功就停，所以平均成本遠低於 32。

| 盤面 | greedy 對照 | 單次 deterministic | best-of-32 | 平均嘗試次數 |
|---|---|---|---|---|
| 4×4 | 0.1156 | 0.9404 | **0.9917** | 1.55 |
| 5×5 | 0.0346 | 0.7496 | **0.9495** | 3.59 |
| 6×6 | 0.0046 | 0.5205 | **0.8535** | 7.76 |

**這到底說明了什麼。** 策略大約是 greedy 的 **8 倍（4×4）到 110 倍（6×6）**，
加一點推論預算就能解掉大多數題目——但它**既不精確也不保證**，而 CP-SAT 在這兩點上都贏它。
這條 track 最誠實的總結是：**這個問題其實幾乎不需要強化學習**——
獎勵極度稀疏、完美示範免費、解可自我驗證、遮罩後平均分支只有 1.5，
**RL 擅長的三件事，這個問題一件都不需要**。
**知道什麼時候「不要」用 RL，是這條 track 最扎實的收穫。**

瓶頸也是量出來的而不是喊出來的：6×6 每局約 **14 次真正的選擇**，策略每次選對的機率是 **93.9%**，
而 `0.939 ^ 14.22 = 0.409`——正好就是實測的 deterministic 解題率。
要 deterministic 過 0.85 需要單步 **98.9%**，也就是**錯誤率砍 5.4 倍**。
**任何新想法都先拿這個數字對照。**

完整報告：[`ai-collab/reports/2026-09-12_rl-wrap-up.md`](./ai-collab/reports/2026-09-12_rl-wrap-up.md)。
名詞從零解釋（行為克隆、DAgger、PPO 微調、AlphaZero 式自我改進迴圈）：
[`ai-collab/notes/`](./ai-collab/notes/)。

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
│   │   ├── solvers/          # 九種傳統 solver ＋ 共用 registry
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
uv run pytest            # 2026-09-12 實測 276 passed, 8 xfailed
uv run ruff check .
```

那 8 個 `xfail` 是**刻意的**：它們釘住已退役的 v1 環境的缺陷，
所以「意外通過」代表有人動了它。

### 文件地圖

| 文件 | 內容 |
|---|---|
| [`ai-collab/roadmap.md`](./ai-collab/roadmap.md) | 現況、下一步、已定案的決策。**從這裡開始** |
| [`ai-collab/project_guide.md`](./ai-collab/project_guide.md) | 架構、模組職責、各環境怎麼啟動 |
| [`ai-collab/deployment-guide.md`](./ai-collab/deployment-guide.md) | Docker：每個容器在幹嘛、驗收指令、失敗排查 |
| [`ai-collab/notes/`](./ai-collab/notes/) | **做中學筆記**：方法是什麼、數字怎麼判讀、推論怎麼跑 |
| [`ai-collab/reports/`](./ai-collab/reports/) | 每個實驗一份報告——方法、數字、以及**推翻了什麼** |
| [`ai-collab/handover-rl-solver.md`](./ai-collab/handover-rl-solver.md) | 接手 RL track：帶證據的死路地圖、下一步 |
| [`ai-collab/handover-vlm-parser.md`](./ai-collab/handover-vlm-parser.md) | 接手視覺 track |
| [`ai-collab/dev_log.md`](./ai-collab/dev_log.md) | 完整開發日誌（很長——用搜尋的，不要整份讀）|
| [`AGENTS.md`](./AGENTS.md) | 本子專案的協作規範（給人與 AI 共用）|

---

## 下一步

*   **從克隆好的權重接 PPO 微調**——這是唯一可能推翻「這題不需要 RL」的實驗；
    擋在前面的問題（訓練 value head 會不會傷到策略）已經量過並排除。
*   **把視覺評估集變難**——合成 held-out 已飽和在 1.000；加視覺雜訊、多種渲染風格、更大盤面，
    才能讓它重新有鑑別力。
*   **把六種啟發式 solver 掛上 API**，做同尺度的比較。
