# 專案收尾報告 — LinkedIn Zip Challenge（2026-09-19）

> **這個 side project 在 2026-09-19 告一段落：停止開發與實驗，只做修正與收尾。**
> 本報告是收尾的總帳：現況、這次修了什麼、怎麼驗的、沒做完的是什麼。細節拆在三份子報告與一份權重文件：
>
> | 子文件 | 回答什麼 |
> |---|---|
> | [`../model-weights.md`](../model-weights.md) | 兩個模型的權重該怎麼提供、拿到之後怎麼裝 |
> | [`2026-09-19_rl-where-next.md`](2026-09-19_rl-where-next.md) | RL 走到哪、為什麼到不了 100%、「只准走一次」對不對、AlphaZero 可以改什麼、還能怎麼走 |
> | [`2026-09-19_computer-use-agents-and-zip.md`](2026-09-19_computer-use-agents-and-zip.md) | GPT-6 等級的 computer-use agent 會怎麼解 Zip |
> | [`../plans/2026-09-19_project-wrap-up.md`](../plans/2026-09-19_project-wrap-up.md) | 收尾過程的逐步紀錄（含所有實測的原始觀察）|
>
> 驗收角度：**作者**（§9.1）、**repo 訪客**（§9.2）、**二次開發者**（§9.3）。所有數字都是實測或引用既有產物，估計值會標明。

---

## 0. 一頁結論

| 功能 | 陌生人 `git clone` ＋ `python start.py` 之後 | 證據 |
|---|---|---|
| **出題**（程序化生成，保證有解）| ✅ 直接可用 | 收尾驗收用它現場生成 4／5／6×6 並全部解出 |
| **編輯盤面讓 AI 解**（Gradio 互動編輯、Svelte 畫布）| ✅ 直接可用；9 種 solver | 5 個頁面 200、9 種 solver 經 HTTP 驗收（§2）|
| **上傳截圖讓 VLM 解** | ⚠ **程式可用，但權重要另外取得**（9.1 GB，未上傳）| 有權重時：全新合成 6 張＋held-out 4 張 **10/10 全對** |
| **多種 solver** | ✅ 3 種精確解＋5 種啟發式直接可用；**RL solver 要另外取得 14 MB 權重** | 無權重回 503（明確訊息），放入權重不必重啟即可解題 |

| 亮點 | 結論一句話 |
|---|---|
| **VLM（微調）** | 目標是「解我們自己做的圖」：**達成**。合成 held-out 200/200，收尾時再用全新生成的圖驗一次 10/10 |
| **RL solver** | 目標門檻（best-of-32：4×4 ≥ 0.90、6×6 ≥ 0.85）**達成**：0.9953／0.9465。但最好的模型是**模仿學習**訓的，不是正牌 RL；**單次嘗試到不了 100%**，原因與後續路線在 RL 子報告 |

**沒做完、而且需要本人動手的只有一件**：把兩個模型的權重上傳（要 GitHub／Hugging Face 帳號）。步驟已寫好（§5）。

---

## 1. 這次收尾做了什麼

### 1.1 git：所有進度收進 `main`

開工時這台電腦有 5 個 worktree（`ml-workshop`、`zip-infra`、`zip-rl`、`zip-solvers`、`zip-vlm`）、本機 10 條分支、remote 8 條分支。
**所有分支都已經是 `main` 的祖先**，但有兩個 worktree 留著**沒 commit 的已完成工作**：

| worktree | 內容 | 處置 |
|---|---|---|
| `zip-solvers` | PSO「包 5 秒預算」量測結果（4×4 1.000、6×6 0.030）＋ 4 份文件 | 以 patch 移植到收尾分支（逐 hunk 比對相同）|
| `zip-infra` | `deployment-guide.md` 補三段「2026-09-12 本人定案不補測／不再瘦身」| 同上 |

原 worktree 的檔案**沒有動**（不覆蓋、不還原）。收尾分支最後以 fast-forward 併進 `main` 並 push（結果見 §2.3）。

### 1.2 修掉的問題（都是「陌生人第一次跑會撞到」的那種）

| # | 問題 | 誰會撞到 | 修法 | 驗證 |
|---|---|---|---|---|
| A | `start.py` 在 **Python 3.9** 直接 `TypeError` | macOS 內建的 `python3` 就是 3.9；README 卻說「只要 Python 3」| `from __future__ import annotations` | 3.9 與 3.11 都實跑 |
| B | **沒有 NVIDIA GPU** 時 Ollama 起不來，`start.py` 就整個中止，app 根本不起 | 所有 Mac、沒獨顯的電腦 | Ollama 改成非致命：印出原因、照常起 app | **故障注入實測**（見 §2.1）|
| C | Ollama 起來不代表視覺模型在；陌生人要到上傳圖才看到 503 | 所有新機器 | `start.py` 比對 `.env` 指定的模型，缺就明說並指路 | 假 Ollama 四種情境全過 |
| D | torch 2.4.1+cu121 **只有 x86_64 的 Linux wheel** ⇒ ARM 主機原生建置失敗 | Apple Silicon、ARM Linux | app 的兩份 compose 加 `platform: linux/amd64`（在 ARM 上走模擬，x86 無差別）| compose 解析 ✅；**ARM 無機器可測** |
| E | Dockerfile 的 `uv sync` 沒鎖 lockfile，lock 過期時會安靜地重新解析 | 未來改相依的人 | `uv sync --locked`（uv 官方 Docker 指南的建議）| 冷建置通過 |
| F | 浮動的 `node:lts-alpine`、`ollama/ollama:latest` 會在沒人維護時自己換版本 | 半年後的所有人 | 釘成實測過的 `node:24-alpine`、`ollama/ollama:0.32.13` | 標籤在 Docker Hub 存在；重建通過 |

另外修了文件錯誤：README 的結果表（還是舊模型的數字）、測試數、已做完卻還寫在「下一步」的 PPO 微調；
部署指南的「4 種 solver」「權重是一個指令的產物」；操作手冊的路徑錯字；`notes/01` 把「加深網路」誤列為已證偽。

### 1.3 新增的文件與工具

- 權重提案與安裝步驟：[`model-weights.md`](../model-weights.md)
- 三份報告（本檔＋RL＋computer-use survey）
- **HTTP 驗收腳本**（[`artifacts/wrap-up-acceptance/`](artifacts/wrap-up-acceptance/)）：`acceptance.py`（全部 solver，答案用獨立裁判判，不信 200）、
  `vision_check.py`（版面、牆、路徑能不能解**標準答案那張盤**）、`gguf_compare.py`（兩個 GGUF 是不是同一個模型）

---

## 2. 驗收：done 條件逐項

### 2.1 全新 clone 的 Docker 實測（2026-09-19，本機 Windows 11 ＋ Docker Desktop 4.32 ＋ RTX 4070 Ti SUPER）

做法：把收尾分支 `git clone` 到一個全新目錄（沒有 `.env`、沒有 `models/`、CRLF 換行，和 Windows 陌生人一樣）。

| 項目 | 結果 |
|---|---|
| 冷建置（`--no-cache --pull`）| **88 秒**，其中 `uv sync --locked` 73.8 秒、`npm install` 8.8 秒、`vite build` 成功；image 5.86 GB |
| 容器內完整測試 | **331 passed, 8 xfailed**（Python 3.11.16、torch 2.4.1+cu121、CUDA 不可用＝符合設計）|
| 頁面 | `/api/echo/health`、`/ui/`、`/svelte-ui/`、`/docs`、`/openapi.json` 全部 200 |
| solver 清單 | 9 種（PSO 依定案不在）|
| **沒有 RL 權重** | 3 種精確解在 4／5／6／7×7 共 5 盤**全解**；RL 在 4–6×6 回 **503**（訊息指出缺哪個 checkpoint）、7×7 回 **400**（說明支援的尺寸）；啟發式在 7×7 有 2 種在 5 秒內放棄（回 200＋「could not find a solution」，符合設計）|
| **放入 14 MB RL 權重（不重啟）** | RL 解出生成的 4／5／6×6；真實題 `puzzle_01`（10 道牆，訓練分布外）這次放棄——與部署指南記錄的「同題 20 次解 10 次」一致 |
| **讀圖**（有權重時）| 全新合成 6 張 **6/6**、held-out 4 張 **4/4**（版面逐格、牆集合、路徑解得開標準答案盤三項全對）；第一張 58 秒（載模型），之後 3–8 秒 |
| 讀圖回歸 | 6 張真實截圖的輸出與操作手冊的預期表**逐列相同**（含刻意保留的失敗案例 `puzzle_03`）|
| **模型缺席** | 讀圖回 **503**，訊息寫明 `model '…' not found` |
| **沒有 GPU**（故障注入：把 GPU driver 改成不存在的名字）| `start.py` 印出 Docker 的錯誤與說明、**沒有中止**、app 照常起來 |
| 開發版 stack | app healthy、`uvicorn --reload` 是 PID 1、vite 5.4.20 在 5173 回 200 |
| `start.py` 在 Python 3.9 | `--help`、`--status`、完整啟動流程都跑過 |

原始數據：[`artifacts/wrap-up-acceptance/`](artifacts/wrap-up-acceptance/)（`acceptance-*.json`、`vision-*.json`）。

### 2.2 ⚠ 這台機器上**沒測到**的部分，以及原因

> **2026-09-19 第二輪更新**：本人定案把 WSL 改回 NAT，這兩項已補驗通過（[第二輪報告](2026-09-19_wrap-up-round-2.md) §0）。另外下文把 microsoft/WSL#41284 當成「mirrored 與 Docker 衝突」的證據是**誤讀**：那個 issue 裡新版 Docker Desktop（4.85）的 localhost 是通的，較準確的根因是「mirrored 搭配本機的 Docker Desktop 4.32」。

實測時發現**本機的 Docker 埠轉發整個失效**：容器內 health 200，但主機連 `127.0.0.1:7440` 失敗；
一個跟本專案無關的最小容器（`python -m http.server`）也一樣。重啟 Docker Desktop 無效。
**根因是本機設定**：`%USERPROFILE%\.wslconfig` 在 2026-09-18 被改成 `networkingMode=mirrored`，
而 WSL 的 mirrored 模式與 Docker Desktop 的埠轉發衝突是已知問題（[microsoft/WSL#10494](https://github.com/microsoft/WSL/issues/10494)、[#41284](https://github.com/Microsoft/wsl/issues/41284)）。

因此：
- 上面所有 HTTP 驗收都是在**容器內**打（`docker exec`）；讀圖是另起一個同 image 的探針容器、直連 Ollama。
  **程式與 image 都驗到了，沒驗到的是「主機 → 容器」這一段埠轉發**（那是 Docker 自己的機制，compose 設定這次沒改）。
- **`start.py` 從主機輪詢 health、Svelte 在瀏覽器裡按解題**這兩項，在本機目前的設定下測不到。
- 這個坑已寫進 README 的 troubleshooting，別的 Windows 使用者也可能踩到。

**我沒有改 `.wslconfig`**（全域設定、昨天才改，可能有別的用途）。要補測的話見 §10。

### 2.3 程式碼品質、git、GitHub 上的實際樣子

| 項目 | 結果 |
|---|---|
| `uv run pytest`（主機）| **330 passed, 1 skipped, 8 xfailed**（skip＝`test_solver_service.py:98` 缺 RL checkpoint，屬預期）|
| `ruff check`、`ruff format --check` | All checks passed、114 files already formatted |
| repo 根 `pre-commit run --all-files` | ruff format ／ ruff check 全過 |
| 文件連結 | 本次改過的 20 份 Markdown、289 個相對連結，0 個失效 |
| 合併與 push | 收尾分支 fast-forward 進 `main`、push（`b72d371..07a78dc`，之後再補一次收尾紀錄）；**本機 `main` ＝ `origin/main`**，無 force push |
| 從 **GitHub** 重新 clone | HEAD 相同；自實測版本 `34f4617` 之後**沒有任何程式或設定變動**（只有文件與驗收產物）；建置命中快取、healthy，容器內驗收與先前一致（5 頁 200、9 種 solver、RL 無權重 503／7×7 400）|
| GitHub **實際渲染**（Chrome headless 讀 github.com 頁面）| README、README_zh-TW、本報告、RL 報告、survey、model-weights、deployment-guide、roadmap、dev_log：**殘留 `**` 0 個**（roadmap 原有 1 處粗體失效，已修）、README 的錨點都存在；README 與本報告另做整頁截圖目視檢查，表格與程式碼區塊正常 |

---

## 3. 四個功能的現況

| 功能 | 入口 | 限制 |
|---|---|---|
| 出題 | Gradio「Generate Puzzle」；`src/core/puzzle_generation/`；RL 資料集產生器 `src/core/rl/generate_dataset_v2.py` | 先畫哈密頓路徑再挖題 ⇒ **一定有解，但不保證唯一解**（6×6 訓練題 87.5% 多解）|
| 編輯盤面讓 AI 解 | Gradio「Puzzle Solver (Interactive)」、Svelte `/svelte-ui/`；API `POST /api/solver/solve` | 啟發式與 RL 會「放棄」（回 200＋could not find），不是盤面無解 |
| 上傳截圖讓 VLM 解 | Gradio「Solve from Screenshot」；API `POST /api/vision/solve` | 需要 NVIDIA GPU ＋ 權重；回應附 `solvable` 與 `warnings`，`solvable: false` ＝ 一定讀錯了 |
| 多種 solver | `GET /api/solver/list`（唯一來源 `src/core/solvers/registry.py`）| 精確解：DFS、A\*、CP-SAT；學習：RL（4–6×6）；啟發式：ACO、GA、SA、Tabu、Monte Carlo（每請求 5 秒、重跑到通過裁判）；PSO 保留程式但不上線 |

---

## 4. 兩個模型

### 4.1 VLM：讀截圖

| | |
|---|---|
| 模型 | Qwen3.5-4B（Apache-2.0）＋ LoRA 微調 → 併回 → GGUF f16 → Ollama `zip-qwen35-4b-p4c:f16` |
| 目標 | 「只要能解我們自己做的圖」（本人定案）|
| 證據 | 合成 held-out **200/200**（四層指標全 1.000，2026-08-22）；收尾時**全新生成**的 6 張（牆 2–12、亮／暗主題、四種格子大小）＋ held-out 4 張 **10/10** |
| 真實截圖 | 6 張：端到端 5/6、牆 F1 0.972（未微調 2/6、0.438）；**n=6 不足以宣稱「看得懂 LinkedIn 截圖」**，也不在目標內 |
| 已知限制 | ① 合成評估集**已飽和**（全 1.000，量不出後續改動的好壞）；② 訓練全是 6×6（兩張 7×7 真實截圖也對了，但樣本少）；③ 權重未公開 |

### 4.2 RL：一步一步走的策略

| | |
|---|---|
| 服務模型 | `bc_multi_456_e6`：一個網路吃 4×4／5×5／6×6，1.17M 參數，行為克隆 6 epochs（GPU 約 5.5 分鐘）|
| 成績（held-out，寬鬆尺）| 4×4 det 0.9410／best-of-32 **0.9953**；6×6 det 0.5430／best-of-32 **0.9465**；5×5 det 0.7701 |
| 最好的實驗 | ExIt 第一輪：6×6 det **0.6455**（+0.125，單 seed，未上線）|
| 誠實結論 | 這題獎勵極稀疏、示範免費、解可驗證、mask 後分支 1.5 ⇒ **RL 的典型優勢都用不到**；最好的模型是監督式訓練的 |

---

## 5. 權重怎麼提供（摘要；完整版 [`model-weights.md`](../model-weights.md)）

| 模型 | 大小 | 建議 | 為什麼不選別的 |
|---|---|---|---|
| RL | 14 MB | **本 repo 的 GitHub Release**（每檔 < 2 GiB、不限頻寬）| 自己重訓不能逐位重現：資料集不進版控，而出題器用牆鐘逾時，重生是另一包題 |
| VLM | 9.1 GB（兩個 GGUF）| **Hugging Face model repo**（兩個 GGUF＋兩行 `FROM` 的 Modelfile＋model card）；Ollama registry 可當鏡像 | 超過 Release 單檔上限；`ollama run hf.co/...` 直拉**不等價**（會另外套 template，而且沒保證 mmproj 會一起抓）|

兩個關鍵事實（2026-09-19 實測）：
- 微調模型在 Ollama 裡**只有兩層、沒有 template 層**——完整 prompt 由 app 自己渲染。所以使用者端必須重建出同樣的兩層，不能讓 Ollama 另外套模板。
- 匯出的文字塔 GGUF 和 Ollama 裡的 blob **雜湊不同**，但逐張量比對 **426/426 位元組相同**、metadata 35/35 相同——Ollama 匯入時只重排了張量順序。⇒ 要發佈的就是匯出檔。

---

## 6. RL 總結（摘要；完整版 [`2026-09-19_rl-where-next.md`](2026-09-19_rl-where-next.md)）

**為什麼到不了 100%**（每一條都有量過的證據）：
1. 6×6 單次要連對約 **14 個真正的選擇**；服務模型每次選對 95.8%，連乘只剩一半。要到 deterministic 0.85 得把單步錯誤率砍 3.7 倍。
2. **致命錯誤犯得早、爆得晚**：完美的一步前瞻只值 +0.03；59% 的關鍵決策點四個方向早就全輸。
3. 網路是反射，**搜尋只在推論期**，想出來的東西沒有回饋到訓練（ExIt 第一輪回饋一次就 +0.125）。
4. 題目多解、標籤只記一條 ⇒ 模仿學習越訓越尖、過擬合。
5. 分布邊界：只訓練 4–6×6、牆 0–5 道；7×7 不支援。
6. 學出來的策略沒有完備性：「100%」只有搜尋能保證，問題是「在多少預算內」。

**「只准走一次」錯了嗎？** 當**訓練**規則是對的（v1 允許重踩時，獎勵與遊戲規則反相關、策略卡迴圈）；
當**評分**規則早就沒在用（一直是 best-of-N）。你提的「**允許倒車、總步數 ≤ 4n²**」是**更好的統一量尺**
（重來、倒車、搜尋都換算成步數），但它**比 best-of-32 更嚴**——4n² 只保證容得下約 4 次完整嘗試：
服務模型 6×6 保證 ≥ 0.7375、估計約 0.80。**盲目倒車沒用**（量過：DFS 式倒車在 6×6 輸給整局重來），要的是知道倒回哪裡。

**AlphaZero 對照，該改的是**：
- **架構**：去掉佔 89.7% 參數的攤平層，改成全卷積殘差網路（8 blocks 約 60 萬參數，全部用在空間推理），與盤面尺寸無關。
- **參數量**：不是瓶頸（三個實測）。
- **訓練量**：BC 在 epoch 2–5 就到頂；缺的是**狀態的多樣性**，不是步數。
- **訓練方法**：缺兩樣——①回答「這局面還解得開嗎」的 value（**精確解可以免費標註**，這是這題比圍棋佔便宜的地方）；②**搜尋進入訓練迴圈**（ExIt／AlphaZero；搜尋算子可選 Gumbel、PHS／LevinTS）。

**還能怎麼走**（依序，都沒做）：收完半截的對照 → 4n² 統一量尺 → 可解性 value 引導搜尋 → 全卷積網路 → ExIt／AlphaZero 迴圈 → pass@k 獎勵的 GRPO。

---

## 7. VLM：目標達成的證據與邊界

目標是「只要能解我們自己做的圖」，所以驗收用的是**自家 renderer 的新圖**，而不是真實截圖：
收尾當天用一個遠離訓練 seed 的 seed（919000000）重新生成 6 張，涵蓋 2–12 道牆、亮／暗主題、72–132 px 格子，
加上 4 張 held-out，**10/10 在版面、牆、解三層全對**。

邊界要講清楚：這證明的是「學會了我們畫的圖」，不是「看得懂任何截圖」（本人 2026-08-22 定案不做真實截圖）。
合成評估集已飽和，下一步若要繼續是**把評估變難**（雜訊、多種渲染風格、更大盤面），不是換更大的模型。

---

## 8. GPT-6 computer use（摘要；完整版 [`2026-09-19_computer-use-agents-and-zip.md`](2026-09-19_computer-use-agents-and-zip.md)）

- **GPT-6 Astra** 在 2026-09-03 發布，官方稱電腦操作最強（OSWorld 2.0 72.6%）。
- 「會用小畫家畫圖」：**社群示範**，不是官方。「會玩 2048」：**查無一手出處**（搜到的是用它做出來的 2048 類遊戲）。
- **沒有人讓它玩過 LinkedIn Zip 的一手紀錄**。推演：它多半會「看截圖 → 寫回溯程式 → 拖滑鼠 → 看畫面修正」——**等於現場重做本專案的管線**；
  最可能失敗在**讀牆**（我們未微調的 4B 模型牆 F1 只有 0.438），而成本是分鐘級加 API 費用，我們本機約 5 秒一張。
- 實證：鉛筆謎題上，通用模型單發作答很弱、加上驗證器迭代才拉高一個量級（Pencil Puzzle Bench：GPT-5.2 20.2% → 56.0%）；
  GPT-6 Astra 玩 Baba Is You，70 關裡只有 7 關是一次、不用 undo 解開的。
- 對本專案的意義：出題器＋精確裁判＋可操作的編輯器，本身就是一個**不怕訓練資料污染**的 computer-use 評測環境（future work）。

---

## 9. 三種讀者的驗收指南

### 9.1 作者

| 要確認的 | 在哪 |
|---|---|
| 所有進度都在 `main`（本機＝remote）| §2.3；`git log origin/main --oneline -12` |
| 散落在 worktree 的工作沒有遺漏 | §1.1 |
| 本機環境被動過什麼 | §11 |
| 需要你動手的事 | §10 的 🔑 項 |

### 9.2 repo 訪客（想看它是什麼、跑起來玩）

1. 讀 [`README.md`](../../README.md)（中文版 [`README_zh-TW.md`](../../README_zh-TW.md)）。
2. `git clone` → `cd ml-workshop/linkedin-zip-challenge` → `python start.py`，開 <http://127.0.0.1:7440/ui>。
3. 預期：出題、編輯、9 種 solver 中 8 種直接能用；RL 與讀圖會說缺權重、指向 README 的「Model weights」一節。

### 9.3 二次開發者

| 想做什麼 | 從哪開始 |
|---|---|
| 了解架構 | [`project_guide.md`](../project_guide.md) |
| 起服務、看容器、排錯 | [`deployment-guide.md`](../deployment-guide.md) |
| 驗證自己的改動沒弄壞服務 | `artifacts/wrap-up-acceptance/acceptance.py`、`vision_check.py` |
| 接手 RL | [`handover-rl-solver.md`](../handover-rl-solver.md) → [RL 子報告](2026-09-19_rl-where-next.md) §5 |
| 接手 VLM | [`handover-vlm-parser.md`](../handover-vlm-parser.md) |
| 接手 solver／API | [`handover-solvers.md`](../handover-solvers.md) |
| 名詞與判讀規則 | [`notes/`](../notes/) |
| 規範（venv、測試、紅線、commit）| [`AGENTS.md`](../../AGENTS.md)、repo 根的 `rules.md` |

---

## 10. 沒做完的事（future work，誠實列）

| | 項目 | 為什麼沒做 | 下一步 |
|---|---|---|---|
| 🔑 | **上傳兩個模型的權重**（2026-09-19 第二輪：RL 已是 draft release、VLM 資料夾備妥待本人登入 HF）| 要本人的 GitHub／Hugging Face 帳號，屬對外發佈 | 照 [`model-weights.md`](../model-weights.md) §4 做，再做一次 §6 的「從零取得」驗收，最後改 README 的「Model weights」一節 |
| ✅ | **主機端埠轉發的兩項驗收**（`start.py` 輪詢、Svelte 瀏覽器按解題）——2026-09-19 第二輪已補驗 | 本機 `.wslconfig` 是 mirrored 模式，Docker 埠轉發失效（§2.2）| 暫時把 `networkingMode=mirrored` 註解掉 → `wsl --shutdown` → 重跑 `python start.py` 與 `ai-collab/reports/artifacts/svelte-solver-list/drive_editor.py` → 改回來 |
| | RL 半截的實驗：嚴格尺 BC vs ExIt、6×6 seed 雜訊、ExIt 第二輪 | 本人要求停止實驗 | [`handover-rl-solver.md`](../handover-rl-solver.md) §0.3（指令與成本都在）|
| | RL 的評分腳本不在版控（`hi-collab/scratch/probe_cross_size.py` 等）| 私人工作區；搬進來要先清理 | 二次開發者目前**無法從 repo 重現 best-of-N 數字**；本報告的 `acceptance.py` 只驗服務行為，不是那把尺 |
| | ARM 主機（Apple Silicon）| 沒有機器 | compose 已釘 `linux/amd64`，理論上走模擬能跑，**未實測** |
| | 沒有 GPU 時的讀圖 | Ollama 容器預約 NVIDIA GPU | CPU 推論未測 |
| | VLM 評估集變難 | 已飽和，但本人定案先不做 | handover-vlm §6 |
| | Svelte 前端的測試不在 pytest | 需要起服務與 Chrome | 端到端腳本已存在（`artifacts/svelte-solver-list/drive_editor.py`）|
| | `start.py --fetch-models`、VLM 轉 Q8_0、RL checkpoint 去 pickle | 新功能，收尾不做 | [`model-weights.md`](../model-weights.md) §8 |

---

## 11. 本機環境變動紀錄（給作者）

| 動作 | 原因 | 還原方式 |
|---|---|---|
| `docker stop` 了 `zip-app-zip-infra-zip-challenge-app-1` | 它佔著 7440（跑的是舊程式 `babc1cb`）| `docker start zip-app-zip-infra-zip-challenge-app-1`（或在 `ml-workshop` 跑 `python start.py` 起最新版）|
| **重啟過一次 Docker Desktop** | 診斷埠轉發 | —（無效，根因是 `.wslconfig`）|
| ⚠ 重啟的副作用：4 個別專案容器被自動拉起 | 它們是 `restart=always`，守護程序重啟時即使原本停著也會起來（`omni_parser` 吃約 50% CPU、`ai_translator_app` 重啟迴圈）| **已 `docker stop` 回原本的停止狀態**（`ai_translator_app`、`pdf2zh_server`、`omni_parser`、`speech_motion_aligner`）|
| `zip_ollama_server` 被重建過（改用釘版本的 `ollama/ollama:0.32.13`，最後一次由 `zip-vlm` 的 compose 建）| 全新 clone 的實測與權重比對 | **收尾時已停止**，與開工時同為停止狀態；模型都在 volume 裡。下次在任何 checkout 跑 `python start.py` 會依該 checkout 的 compose 重建並啟動 |
| 建了測試用的 clone、容器、image | 驗收 | 全部**已停止**：`zip-app-fresh-clone`、`zip-dev-fresh-clone`、`zip-app-gh-clone` 三個 compose 專案與 `zip-vision-probe` 容器；image 與正式版共用 5.86 GB 的層，額外佔用很小。要清掉：`docker compose -p <專案名> down`、`docker rm zip-vision-probe`（我沒有刪任何容器或 image）|
| **沒有動** `.wslconfig`、其他 worktree 的檔案、任何 `models/`／`datasets/` | — | — |

**可以整理的 worktree 與分支**（全部已併進 `main`；要不要刪由你決定，我沒有動）：
`zip-infra`、`zip-rl`、`zip-solvers` 三個 worktree（其中 `zip-infra`、`zip-solvers` 仍留著已移植進 `main` 的未 commit 改動，
`git worktree remove` 會拒絕，需要 `--force`）；本機分支 `dev-hero`、`feat/rl-masked-ppo`、`feat/vlm-parser`、`feat/rl-a2-training`、
`feat/rl-ppo-finetune`、`feat/infra-slim-image`、`feat/expose-heuristic-solvers`、`feat/rl-exit`；remote 上對應的已合併分支。

---

## 12. 出處

本專案：各節連結的報告、[`artifacts/wrap-up-acceptance/`](artifacts/wrap-up-acceptance/)、[收尾進度紀錄](../plans/2026-09-19_project-wrap-up.md)。
外部（查證 2026-09-19）：見三份子文件各自的出處節；本檔另引用
[microsoft/WSL#10494](https://github.com/microsoft/WSL/issues/10494)、[microsoft/WSL#41284](https://github.com/Microsoft/wsl/issues/41284)（mirrored 模式與 Docker 埠轉發）、
[uv Docker 指南](https://docs.astral.sh/uv/guides/integration/docker/)（`uv sync --locked`）、
[ragflow#9573](https://github.com/infiniflow/ragflow/issues/9573)（無 GPU 時的 `could not select device driver` 錯誤）。
