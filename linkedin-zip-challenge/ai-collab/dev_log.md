# Development Log

> Chronological development history of `linkedin-zip-challenge`, **newest first**.
> For the current status and next steps, read [roadmap.md](roadmap.md) instead — this file is the full archive.
> Add one entry per development session, dated `## YYYY-MM-DD`.

## 2026-09-12

### Track B — app image 22.9 GB → 5.86 GB ＋ 兩份 compose 實跑驗收（branch `feat/infra-slim-image`, worktree `zip-infra`）

細節與原始輸出在 [`deployment-guide.md`](deployment-guide.md) §3（身分）／§6（驗收）／§8（image 大小）。

**大在哪先量再改。** `docker history` 顯示舊基底 `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` 裡的
conda（附 torch 2.3.0）7.59 GB、CUDA toolkit／cuDNN／CUDA 函式庫 4.79＋2.45＋2.01 GB——
**app 容器不用 GPU，這些全是死重**；`uv sync` 5.74 GB 才是真正要的。
兩份 Dockerfile 的基底換成 `python:3.11-slim-trixie`（查證：docker-library/official-images 的 `library/python`，
`3.11-slim` 現在指向 `3.11.16-slim-trixie`；標籤寫死 Debian 代號，OS 升級要改檔才會發生）。
**第一次建置就成功**（110 秒），`22.9GB → 5.86GB`；prod／dev 兩個 image 共用 5.858 GB，並存不多佔空間。

**計畫書標「未驗證」的系統函式庫：一個都不用補。** `src/` 沒有 `import cv2`；`matplotlib` 是間接相依，
wheel 自帶函式庫。驗法不是看 health 200（那回答不了這題），而是**在新 image 裡跑完整測試**：
`276 passed, 8 xfailed`，和 host 一模一樣。

**驗收時發現「200 ≠ 解出來」。** RL 對 `puzzle_01`（6×6）回 HTTP 200，body 卻是
`could not find a solution`。連打 20 次：6×6 解出 10/20、4×4 20/20；舊 image 在同一題也是 2 次中 1 次
⇒ 這是題目對 RL 的難度，不是瘦身造成的。驗收腳本因此改成**另外判斷是否解出**，並加一題 4×4 當 RL 的正向證據。

**順手修掉兩個「看起來有起來」的缺陷**（都在 Track B 擁有的檔案內）：
① 兩份 compose 建出**同名 image**（`linkedin-zip-challenge-zip-challenge-app`），建 dev 就蓋掉 prod
（第一版先各加 `image:` 擋住，同日被下面的根治取代——那個寫死的名字換個 worktree 照樣撞）；
② `start.py --dev` 叫你開 `7440/svelte-ui/`，但 dev 模式下那是 **404**（`./src` 蓋掉 image、host 沒有 `dist/`），
vite 的網址又少了 `/svelte-ui/` ⇒ dev 模式改印 `5173/svelte-ui/`（實測 302 → 200）。

**★ 根治 worktree 撞名：app 每個 checkout 一組、ollama 整台機器一個。** 瘦身時從 `zip-infra` 一跑 `up`，
就把 `zip-rl` 起的 stack 安靜地換掉了。撞名的不只一處：compose 專案名（預設＝目錄名，每個 worktree 都叫
`linkedin-zip-challenge`）、寫死的 `container_name`、從 main 複製來的 `.env` 埠號、image 標籤。
第一個念頭是「整台一組、接管前先講一聲」——那是規定，不是根治。**真正的原因是兩種身分被綁在同一個專案裡**：
app 服務的是「這個 checkout 的程式碼」，該每個 checkout 一份；ollama 佔的是 GPU，該整台一份。拆開之後：
`docker-compose.ollama.yml` 自成專案 `zip-ollama`（名字寫死＝刻意的單例），`start.py` **只在它沒跑時才起、
在跑就完全不碰**——`--dry-run` 證實從別的 checkout 對它 `up` 會 **Recreate**（`./models` 掛載路徑不同），
那會把別人正在用的模型卸掉；app 專案由 `start.py` 依 checkout 目錄命名 `zip-app-<checkout>`／`zip-dev-<checkout>`，
所有寫死的容器名與 image 名都拿掉；app 改走 `host.docker.internal:11435` 找 ollama（不同專案、不同網路）。
**驗證**：在 scratchpad 做一個假的第二個 checkout（`APP_PORT=7441`），兩組同時服務——7441 的 RL 回 **503**
（它沒有 `models/`，證明真的是另一份程式與資料）、7440 照常四種都解出；對方起來、關掉的前後，
zip-infra 的 app 與 ollama **容器 ID 與啟動時間完全沒變**。prod↔dev 切換也由 `start.py` 自己停掉另一個。
剩下唯一能撞的是 host 埠，而那會**大聲失敗**（`port is already allocated`），`start.py` 會提示改 `.env` 的 `APP_PORT`。
**最後一塊拼圖（本人授權後補做）**：`Index.svelte:4` 把 API 寫死成 `127.0.0.1:7440`，不在 7440 的 checkout，
其 Svelte 編輯器會打到別人的 app。改成 `import.meta.env.VITE_API_URL ?? ""`——**建置版走同源相對路徑**
（app 在哪個埠就打哪個埠），dev 由 compose 的 `VITE_API_URL` 提供位址（原本填的是容器主機名
`http://zip-challenge-app:7440`，瀏覽器根本解析不到，所以那個變數從來沒生效過）。
⚠ **中途寫錯一次要記下來**：本來想在 `vite.config.ts` 用 `define` 給 dev 一個預設位址，
實測**兩邊都沒生效**——`define` 在 dev 是掛到 globalThis、不替換 `import.meta.env`，而 build 分支我又給了空物件。
已移除，改成「dev 一定要有 `VITE_API_URL`（compose 會給）」並寫進文件。
驗證：正式版 image 的 `dist/` 裡搜不到 `127.0.0.1:7440`、`/svelte-ui/` 200、四種 solver 照常；
dev 的 svelte 容器裡 `VITE_API_URL=http://127.0.0.1:7440`。
⚠ 仍未驗：在瀏覽器裡實際按下解題按鈕——Vite 在 dev 是執行期注入 `import.meta.env`，不開瀏覽器量不到最終值。

**沒做的**：換 CPU 版 torch（剩下 5.66 GB 的大宗是 12 個 `nvidia-*` wheel，但那要動 `pyproject.toml`／`uv.lock`，
不屬於 Docker 層）；hot reload 實際觸發（要改 `src/` 才測得到，只驗到 reloader 起來、盯的是 `/app/src`）。
### Track A — BC → PPO 微調：步子縮小十倍之後，6×6 第一次看到 RL 的加值（branch `feat/rl-ppo-finetune`, worktree `zip-rl`）

完整報告在 [`reports/2026-09-12_rl-bc-ppo-finetune.md`](reports/2026-09-12_rl-bc-ppo-finetune.md)。

**做了什麼**：`train_maskable_ppo.py` 加 `--init-from <run-id>`（只搬 actor＋critic 權重，optimizer 重開、超參仍來自
`train_config`、先比對觀測空間、強制關 curriculum）與 `--learning-rate` 覆寫；測試 276 → **280 passed, 8 xfailed**。
對照組一律是**同一個 `bc_multi_456` 不微調**，評分用產出 BC 基準的同一支 probe（重評 BC 逐位重現 0.940445／0.991714）。

**① 預設超參（lr 3e-4）會把 BC 弄壞。** 4×4 三個 seed 都比 BC 差約 0.025（deterministic −0.0249／−0.0249／−0.0264、
best-of-32 −0.0218／−0.0259／−0.0264）；seed 全距只有 0.0016 ⇒ **確證**。6×6 直接崩：訓練集抽樣 0.785 → 0.07，
held-out deterministic 0.5205 → **0.2415**（900k 手動停，因為對崩掉的模型跑 best-of-32 要 20–40 分鐘卻沒有新資訊）。
診斷：前 100k 步 approx_kl 4×4 **0.14–0.16**、6×6 **0.35**（正常約 0.01–0.02）；6×6 還加上熵暴增約 13 倍。
另外 **BC 的 critic 沒有真的暖好**：第一次更新 explained variance 約 −0.005——它只看過永遠成功的專家軌跡。

**② 只改學習率 3e-4 → 3e-5**：KL 回到 0.009（4×4）／0.019（6×6），訓練曲線不再先掉。
- **4×4（3 個 seed）**：deterministic **+0.0070**（全距 0.0020），但 **best-of-32 −0.0129**（全距 0.0011）
  ⇒ **兩件事都是確證**。機制檢驗：held-out 專家狀態上的熵 **−60%**、每題相異路徑 2.37 → 1.59
  ⇒ **變尖銳、多樣性下降**，和 LLM 做 RL 常見的「pass@1 升、pass@k 降」同形。
  過擬合對照：進步約 2/3 帶到測試集 ⇒ 沒有背題的證據。
- **6×6（3 個 seed）**：deterministic 0.5575／0.5255／0.5400 對 BC 0.5200 ⇒ **3/3 為正、平均 +0.021、全距 0.032**
  ⇒ **方向一致、幅度未定**（順帶第一次量到 6×6 微調的 seed 雜訊 ≈ ±0.016，4×4 是 ±0.001）。
  seed 1 的 **+0.0375** 不能當代表值。測試集的進步比訓練集（+0.025）還大 ⇒ 不是背題。
  **best-of-32（seed 1）：0.7850 對 BC 0.8535 ⇒ −0.0685**，分水嶺在 N=2 與 N=4 之間。
  ⇒ **依本專案的判定基準（best-of-32），微調是淨損失。**

**⚠ 我自己踩到小樣本的坑**：300 題算出的 6×6 best-of-16 是 **+0.013**，完整 2,000 題是 **−0.059**，**方向相反**
（該樣本數的取樣誤差約 ±0.046）。**小樣本只能看機制，不能看勝負。**

**③ 順手量到、會改變優先序的事**：**BC 本身在 6×6 嚴重過擬合**——訓練集 deterministic 0.9005 vs 測試集 0.5200（落差 0.38；
4×4 只有 0.042），BC 的 val 選擇準確率在 epoch 6 就到頂、之後只剩訓練 loss 在降。
handover §2 關掉「加資料」的依據（落差 +0.009）量的是 **PPO** 模型——對 BC 這條線**從來沒被測過**。

**④ ★★ 追加實驗：BC 早停，買到的比整條微調線多一個數量級。** 同 goal／同資料／**同 seed**，只把 `--epochs` 10 改成 6
（前六行訓練紀錄與 10 epochs 那次**逐位相同** ⇒ 配對比較、不含訓練 seed 雜訊）：
**6×6 best-of-32 從 0.8535 變成 0.9465（+0.093），每題嘗試次數 7.76 → 5.24**；4×4 從 0.9917 變成 0.9953。
但 **best-of-1 反而降**（6×6 0.4810 → 0.4265）⇒ **和微調是同一條軸的兩端**：訓練越久越尖、單次越準、多樣性越少。
⇒ **判定基準是 best-of-32 的話，正確方向是「少訓練」不是「多訓練」**；6×6 從踩線變成遠離門檻。
⚠ 單一 seed、**沒有掃 epoch**（最佳點可能更早）、選 epoch 6 的依據是 val 選擇準確率（pass@1 型訊號，本實驗正好說明它與 best-of-N 會分歧）。
**服務中的仍是 10 epochs 版，要不要換由本人決定**（`solver_service.py` 不屬於本 track）。

**⑤ 收尾：服務模型換成早停版**（本人當次授權）。`solver_service.py` 的 `RUN_ID_BY_SIZE` 三個尺寸改指
`bc_multi_456_e6`，並更新 `DEFAULT_ATTEMPTS` 註解裡過期的數字（0.8500／8.05 次 → 0.9465／5.24 次）。
測試 280 passed 全綠，**且實跑驗證**：一題 6×6 第一次嘗試就解開，log 印 `RL solver (bc_multi_456_e6) solved a 6x6`。
⚠ **這跨了檔案所有權**（計畫書把 `solver_service.py` 排除在 Track A 外）。**`start.py:42-49` 與 `deployment-guide.md`
仍寫舊 run id，那兩個檔是 Track B 的，刻意沒動**——合併前要讓 B 補上，否則 `start.py --status` 會對著舊路徑報告。

**踩到的坑**：probe 用 run id 命名產物，重評已發表的 checkpoint 會**覆蓋**原檔 ⇒ 包一層 `hi-collab/scratch/probe_to.py` 改輸出目錄。
兩個 seed 的 deterministic 逐位相同（0.915588）——驗過是巧合（權重 36/36 不同、各有 106 題只有自己解對）。

### RL Track — 收尾：一個模型吃三個尺寸、掛上 API、Docker 起得來，外加把三份 solver 清單收成一份（branch `feat/rl-a2-training`, worktree `zip-rl`）

完整報告在 [`reports/2026-09-12_rl-wrap-up.md`](reports/2026-09-12_rl-wrap-up.md)；
名詞與判讀規則抽到新的 [`notes/`](notes/)；服務怎麼起看
[`deployment-guide.md`](deployment-guide.md)。

**先回答擱著的那題：6×6 加預算過不過得了門檻。** `bc_6x6` × 2,000 題 held-out ×
`--max-attempts 64`：best-of-16 **0.8105**、best-of-32 **0.8500**、best-of-64 **0.8835**
（5.35／8.05／12.25 次嘗試）。**best-of-32 剛好等於門檻**。
事前用 hazard rate 預測 N=32 ≈ 0.887 — **又外推錯了，這是第三次**，規則因此升級成
「hazard 表只能用來排除，不能用來預測」。而且同一個模型同一份測試集，只因抽樣 rng 串流不同
（上一輪跑到 16 就停、這輪跑到 64），**N=16 的分數就從 0.8000 變成 0.8105** ⇒
**best-of-N 的評估雜訊約 ±0.01，0.8500 是踩線不是穩定達標**。本人定案判定用 best-of-32。

**跨尺寸泛化第一次被量到，而且是單向的。** 觀測本來就 padding 到 8×8、純量帶 `height/8`、
`width/8`，`_load_sample()` 每局重讀高寬 ⇒ **任何 checkpoint 直接就能吃任何 ≤8×8 的盤面**，
不用改一行程式。deterministic 矩陣：`bc_6x6` 在**沒看過的 4×4** 拿 **0.5456**（greedy 0.1170），
`bc_4x4` 在 6×6 只有 **0.0105**（greedy 0.0041）。**大盤面往下相容，小盤面往上完全不行。**
⚠ 這同時**更正了 handover 寫反的一句話**：舊版說「`Linear(4104→256)` 綁死 8×8 padding
所以跨尺寸測不到」——padding 正是**讓**它可測的原因。

**於是做了多尺寸模型，而它推翻了我自己寫在計畫書裡的風險預測。** 新生一份 4／5／6 的資料集
（`seed20300000_n20000_456`，60,000 題，11m16s；4×4 train 15,439／5×5 15,996／6×6 16,000），
`Goal.size: int` 改成 `sizes: tuple[int, ...]`（全專案只有 3 個呼叫點，**env 完全不用改**），
訓一個混合模型 ＋ 三個單尺寸對照，**同資料同測試集**：

| 盤面 | 多尺寸 det | 對照 det | 差 | 多尺寸 bo32 | 對照 bo32 | 多尺寸嘗試數 | 對照嘗試數 |
|---|---|---|---|---|---|---|---|
| 4×4 | **0.9404** | 0.9042 | +0.0362 | 0.9917 | **0.9938** | **1.55** | 1.61 |
| 5×5 | **0.7496** | 0.7131 | +0.0365 | **0.9495** | 0.9330 | **3.59** | 4.33 |
| 6×6 | **0.5205** | 0.4890 | +0.0315 | **0.8535** | 0.8520 | **7.76** | 8.02 |

計畫書裡我寫「混合尺寸**沒有機制理由**能改善 6×6」——**錯了**，6×6 從 0.4890 變 0.5205。
**但 best-of-32 改變了結論的形狀**：加上推論預算後優勢幾乎被吃掉（4×4 **−0.0021**、6×6 **+0.0015**，
都遠小於 ±0.01 的評估雜訊），只有 5×5 還留 +0.0165。**優勢沒消失，只是換成「比較便宜」**——
同一個 N 下三個盤面都用更少嘗試次數。⇒ 實務結論仍是**一個模型取代三個**：deterministic 更好、
best-of-32 相當、推論更便宜、訓練成本略低（465.3s vs 478.1s），而且 **5×5 從此有模型**
（先前只能借 6×6 模型的 0.2941）。⚠ 6×6 的 seed 雜訊仍然沒量過。

**⓪ 順手解掉擋在 PPO 微調前面的問題。** 同 goal 同 seed 同資料，只差 `--value-coef`：
`0.5` 的 val_choice 0.8770／solve 0.9042，`0` 的 0.8777／**0.9125**，差 **−0.0083**
⇒ **在雜訊內，沒有證據支持 value 回歸傷到策略**，2026-09-05 煙霧測試看到的下降沒有重現。
**微調的前置封鎖解除。**

**A5 完成：RL solver 上線。** 新增 `src/core/rl/solver_service.py`——`src/core/rl/` 裡唯一被實驗腳本
以外的東西使用的檔案。它**重用評估用的同一條路徑**（同 env、同遮罩、同 rollout 迴圈），
所以報告的數字就是端點交付的數字。**推論不需要解答**：`PuzzleSample.solution_path` 在
`reverse_curriculum_k=None` 下只被讀第一個元素（起點），而起點就是編號 1 的位置——
有測試釘住這個不變量。三種回應分清楚：**缺 checkpoint → 503、尺寸沒模型 → 400、找不到解 → 200 ＋ 說明**。

**Refactor：三份 solver 清單收成一份。** `app/routers/solver.py`、`app/routers/vision.py`、
`ui/gradio_app.py` **各自維護一份一樣的 `SOLVERS`**，加一種 solver 只會出現在改到的那一處。
RL solver 是壓垮它的案例（第一個既不精確、也不保證可用的 solver）⇒ 新增
`src/core/solvers/registry.py` 當唯一正本（含 `kind` 與說明欄），三處改成 import，
並用測試釘住「三個入口指向同一個物件」。

**Docker：四個「看起來有起來、其實沒有」的缺陷。**
① 正式 image **沒有 `CMD`** ⇒ `docker compose up` 建好 image、容器 Up，**裡面沒有 server**；
② 開發 image 停在 `tail -f /dev/null`，靠 `run_docker_dev.py` 把 server exec 進去，
但那行**沒有 `-d`**、`subprocess.run` 會一直等 ⇒ 後面的 healthcheck 永遠跑不到；
③ **`models/` 6.6 GB 被送進 build context**；④ 兩個服務都沒有 healthcheck。
全部修掉，`models/` 改成唯讀 volume，並新增 **`start.py`** 取代 `run_docker_dev.py`
（`--dev`／`--down`／`--status`，會等到 API 真的回應、並講明少了哪個選配）。
**實測**：兩個容器 `Up (healthy)`、health／`/ui`／`/svelte-ui`／`/docs` 全 200、
四種 solver 全 200、7×7 回 400、`/api/vision/solve` 200。

**順帶抓到設定漂移。** 這個 worktree 的 `.env` 停在 2026-08-15 的
`OLLAMA_MODEL_NAME=openbmb/minicpm-o2.6`，卻配著 `finetune` 的 prompt ⇒ 視覺端點回 200
但吐出**全空的 10×10 盤面**。同步成 `.env.example` 的 `zip-qwen35-4b-p4c:f16` 後，
同一張圖讀成 **7×7 並抓到編號**。`.env` 不進版控、每個 worktree 一份，**所以它會各自過期**。

**文件按「交付品」重寫。** `README.md` 與 `README_zh-TW.md` 重寫：專案是什麼、
**從 clone 到服務跑起來的一行指令**、提供哪些能力（出題／讀圖／解題）、**10 種 solver 清單**、
**兩個模型的選型與訓練與結果**、專案結構、文件地圖。新增 `ai-collab/notes/`（做中學筆記：
[`01-rl-methods-explained.md`](notes/01-rl-methods-explained.md)、
[`02-reading-the-numbers.md`](notes/02-reading-the-numbers.md)、
[`03-inference-and-serving.md`](notes/03-inference-and-serving.md)、決策紀錄）與
`deployment-guide.md`。過期的 `docker-compose.yml.vl_version` 與被取代的 `run_docker_dev.py`
移入 `soft-delete/20260912-005132/`。

**驗證**：`276 passed, 8 xfailed`（原 261；新增 5 個多尺寸、6 個 solver service、4 個 registry）、
`ruff check` 綠、repo 級 `pre-commit run --all-files` 綠、28 份文件的相對連結全數存在。

**⚠ 還沒做、記下來**：app image **22.9 GB**（基底是 CUDA devel 但 app 根本不用 GPU，
GPU 在 ollama 那邊），換 slim 基底可大幅縮小但**未驗證系統相依**；
`start.py --dev` 的完整啟動**沒有實跑過**（只驗了 `--status` 這條路徑）；6×6 的 seed 雜訊沒量過。

## 2026-09-05

### RL Track — the labels were on disk all along: behaviour cloning beats PPO at a ninth of the cost, and it is not RL (branch `feat/rl-a2-training`, worktree `zip-rl`)

Full write-up in [`reports/2026-09-05_rl-behaviour-cloning.md`](reports/2026-09-05_rl-behaviour-cloning.md);
the part worth reading is §4, the conceptual one, not the numbers.

**The gap that was sitting there.** Every puzzle in the dataset ships with its
`solution_path`, and a grep confirms it has been used for exactly two things: choosing the
reverse curriculum's start cell, and replaying legal walks in the A0 diagnosis. **It has
never been a training target.** PPO has been rediscovering, from a reward that only fires
after 35 correct steps, answers already on disk — about 560,000 labelled (state, action)
pairs on 6×6 against a 1.17M-parameter network. Checked the plan, the restart report and
its A0–A6 route table first: supervised warm start is not a decided-against option, it was
never considered.

**It also matches the measured bottleneck's shape.** The oracle report reads the solve rate
as a per-decision accuracy: 6×6 is right 93.9% of the time and needs 98.9%. That is a
classification problem. And BC has two properties PPO's setup does not — it sees full-length
states from the first step (the curriculum needed 5.5M steps to reach k=30/36, so states
near the true start are the least trained), and its signal is one label per step rather than
one reward per episode.

**Result, single seed, same evaluation path, `greedy` reproducing bit-identically on both
boards (0.1170 / 0.0041).**

| | deterministic | best-of-4 | best-of-16 | training |
|---|---|---|---|---|
| 4×4 PPO (1M steps) | 0.8771 | **0.9238** | 0.9549 | 245s |
| 4×4 **BC** (10 epochs) | **0.8947** | **0.9533** | **0.9850** | **95s** |
| 6×6 PPO (8M steps) | 0.4095 | 0.5330 | 0.6490 | ~2,000s |
| 6×6 **BC** (10 epochs) | **0.4620** | **0.6425** | **0.8000** | **232s** |

BC is at least as good at every inference setting on both boards, for a third and a ninth
of the training cost, and on 6×6 the gap *widens* with N (+0.053 deterministic → **+0.151**
at best-of-16). **6×6 best-of-16 is 0.800 against a 0.85 bar — the closest this track has
been.** What holds up is the *cost* claim; 4×4's +0.018 is inside the ±0.04 noise floor, and
6×6's +0.053 is a single seed on a board whose seed noise has never been measured, so it is
"moved, not confirmed". Whether best-of-32/64 clears 0.85 is extrapolation, and this track
has been wrong by 3× extrapolating twice.

**An accident worth keeping: `choice_accuracy` understates the policy, because the puzzles
have multiple solutions.** BC agrees with the recorded solution on 88.11% of real choices
but solves 89.47% of held-out boards — an effective per-choice accuracy of 97.65%. The
missing 9 points are not errors; they are *other legal solutions*. The generator draws a
Hamiltonian path and then carves the puzzle, so most boards admit more than one, and the
dataset records one. **BC won while training against label noise**, and the accuracy metric
is a lower bound that must not be exponentiated into a predicted solve rate.

**Asked directly whether this is still RL. It is not, and that matters.** BC is one member
of the imitation-learning family — the simplest, the one that never touches the environment
(DAgger, IRL and GAIL are the others). Algorithmically it is plain supervised learning:
cross-entropy on a fixed `(X, y)`, no reward, no exploration, no credit assignment. The one
property ordinary classification lacks is that its test distribution is *self-inflicted* —
leave the expert's path and there is no training signal, which is exactly what DAgger exists
to fix. So: the pipeline stays RL (env, masking, evaluation protocol, and the intended next
step is PPO fine-tuning from these weights), and "supervised first, RL second" is what the
original AlphaGo did before Zero dropped the supervised half. **But the honest reading is
that this problem may simply not be one RL should be used on**: the reward is extremely
sparse, perfect demonstrations are free, solutions are verifiable, and after masking the
mean branching factor is 1.5 — so exploration, RL's main advantage over supervision, is
barely needed here. PPO spent 8M steps and ~2,000 seconds to lose to 232 seconds of
`model.fit`. **Knowing when not to reach for RL is the most solid thing this track has
learned**, and it is a "learning by doing" result, not a failure.

**Dataset integrity, re-verified rather than cited.** Asked how many puzzles train and test
hold and whether they can have mixed. Identity here is *derived, not stored*:
`sample_fingerprint()` canonicalises board, walls, blocked cells, number positions and the
solution path into sorted JSON — deliberately not a hash of the pickle, which would hash the
encoding rather than the content. `PuzzleSample` itself has no id field. `--verify` recomputed
all three split digests: ok.

| dataset | train | val | test | internal dups | train∩val | train∩test | val∩test |
|---|---|---|---|---|---|---|---|
| `seed20300000_n20000_4-6` (current) | 31,419 | 3,927 | 3,928 | **0/0/0** | **0** | **0** | **0** |
| `main_n1700_456` (A2's) | 4,080 | 510 | 510 | **4**/0/0 | **1** | **0** | 0 |

Per size, which is what the runs actually use: 4×4 train 15,419 / val 1,927 / test 1,928 and
6×6 16,000 / 2,000 / 2,000, **train∩test = 0 in both**. So the current pack is clean and no
reported test number is affected. **The old pack has two defects nobody had recorded**: four
duplicates inside train, and one puzzle in both train and val. The handover verified and
claimed only `train∩test = 0`, which today's recomputation confirms — but **"verified
train∩test" is not "verified all three pairs are disjoint"**, and that distinction is now
written down.

**Can PPO succeed at all?** On 4×4 it already has, under the inference rule settled today:
best-of-4 is 0.9238 against a 0.90 bar, though deterministic 0.854 ± 0.038 does not clear it.
On 6×6 there is no evidence it can reach 0.85 and several signs the route is expensive: the
curriculum is stuck at k=30/36 having burned 2.4M steps at that level, promotion cost roughly
doubles each rung (812,944 → 1,307,280 → 2,867,776), and the extrapolation to full length
(15–30M steps, 1–2 hours) is explicitly untrustworthy. Reaching full length would also only
mean it can *train* there, not that it scores 0.85 — at 8M steps it is 0.4095 deterministic.
The one PPO experiment nobody has run is exactly that, and it is hour-scale, so it needs
authorisation.

**Also landed: the handover was restructured** after the developer asked whether the session
had diverged. It had — the restart-DFS arm was run to completion after a pilot had already
rejected it. The goal was written at line 520 behind a wall of thirteen bullet points, so the
document now opens with the goal, the judging rule, current standing and the map of closed
lines, and drops from **864 to 215 lines**; the accumulated 28 verified facts, 30 traps,
settled design decisions and experiment chronicle moved verbatim to
[`rl-traps-and-facts.md`](rl-traps-and-facts.md). A self-administered "ask yourself three
questions before running anything" checklist was drafted and then **cut**: the agent that
diverged had read and quoted the goal, so a reflective prompt was never going to bind — the
load-bearing artifact is the factual dead-end map, and the two repeatable analytical errors
went into the traps list where such things already live.

**Verification.** `259 passed, 8 xfailed` (was 251; the eight are the new BC tests), `ruff`
clean, `ruff format --check` clean. The new tests pin three things that would fail silently:
a one-step drift between observation and label, `choice_accuracy` counting forced moves (69%
of 4×4 decisions have one legal move, so an untrained net already scores ~0.69), and BC being
a *third* env construction point after training and evaluation — trap #24 cost three
experiment arms when a flag reached only one of two.

### RL Track — what the learned prior is worth, measured on a budget axis; and half my own recommendation falsified (branch `feat/rl-a2-training`, worktree `zip-rl`)

The previous entry ended by naming policy-ordered DFS as the next candidate, on the strength
of best-of-16 buying +0.240 with the dumbest possible search. Did it the same day, and it
produced one strong positive result and one refutation of the recommendation itself.

**Why a different axis.** Every number this track has recorded is a solve rate, and a solve
rate blends "how hard the puzzle is" with "how good the policy is". best-of-16's +0.240 also
spent 7.5× the inference budget. So this ran everything on one axis — **node expansions** —
and compared five arms at matched budgets. `dfs_random` is the control that matters: if
random branch ordering matches the policy's, the search is doing the work and the network is
decoration. And the point was never to beat `dfs.py`: these puzzles are solvable by
construction and DFS is complete, so at a large enough budget even random ordering reaches
100%. The question is how much search the prior saves.

**The prior is worth a great deal — the first direct measurement of it on this track.**

| board | budget = one path, no backtracking | to reach a target |
|---|---|---|
| 4×4 (n=1,928) | policy **0.8771** vs random **0.0902** | 0.90: policy 36 nodes, random 175 → **4.9× cheaper** |
| 6×6 (n=2,000) | policy **0.4205** vs random **0.0005** | 0.40: policy 36 nodes; **random never gets there inside 500** |

Median nodes to a solution is **15.0 on 4×4 and 35 on 6×6 — exactly the path length**, so more
than half the puzzles are walked straight through with no backtracking at all. Random ordering
needs 51.0 and 251.5.

**But "structured search beats resampling" is false, and it is board-dependent.** On 4×4
`dfs_policy` wins at every budget and reaches 1.0000 by 500 nodes; on 6×6 the crossover sits
between 50 and 100 nodes and `best_of_n` leads from there (0.6260 vs 0.5760 at 175, 0.7005 vs
0.6745 at 500). The mechanism is the one the oracle probe measured earlier the same day: **the
fatal mistake happens early**. Backtracking re-decides the *last* moves; resampling re-rolls
the whole path, early decisions included. 4×4's tree is 10^2.0, so 500 nodes is a real
fraction of it and backtracking recovers; 6×6's is 10^6.2, where 500 nodes is nothing.

**The obvious confound was measured, not argued away.** `dfs_policy` is deterministic and
`best_of_n` samples, so comparing them conflates "backtrack vs restart" with "argmax vs
sample". Added `dfs_policy_sampled`, which keeps the DFS structure and draws the branch order
from the policy's probabilities instead of sorting it. On both boards the sampled arm is
slightly *worse* (−0.014 on 4×4, −0.030 on 6×6), not better, so the resampling arm's advantage
is not its stochasticity.

**And the 1.0000 does not mean what it looks like.** `dfs_random` reaches 0.9990 at the same
budget. The honest sentence is "the prior makes the same search about 5× cheaper", not "the
policy solves every puzzle" — using that 1.0000 as a score would be this track's recurring
mistake in a new costume.

**Recording that the recommendation was half wrong.** The previous entry reasoned "independent
resampling is dumb, so something structured will be better". On 6×6 — the goal with the actual
gap — that was wrong: policy-DFS loses to the dumb thing it was meant to replace. The error was
proposing a method without first asking *which failure mode it assumes*, and checking that
against the failure mode already measured. The evidence that would have predicted this was
collected earlier the same day: 59% of prunable 6×6 decisions are positions where all four
moves already lose. **The refutation was sitting in hand and went unused.**

**Practical read for A5**, if these ever ship as a solver: the inference strategy should differ
by board — policy-DFS at ~100 nodes for 4×4 (0.9907), best-of-N at ~175 nodes for 6×6 (0.6260).
Neither is shippable yet, and the out-of-distribution wall problem is untouched.

**Next candidate, not authorised**: policy-DFS *with restarts* — keep the prior ordering but
restart from the top every K nodes with a resampled order. It is low-budget policy-DFS and
high-budget resampling in one, and both halves are already measured rather than assumed.
About ten minutes.

**Cost and resources.** 6×6 four arms 1,235s, its sampled control 687s, 4×4 five arms 364s.
System CPU **sampled at 18% of 24 cores** while two of these ran concurrently, against the 75%
ceiling — measured, not computed. One operational note: Python buffers stdout when it is
redirected to a file, so the logs stay empty mid-run; **liveness has to be read off accumulated
CPU time, not output** — the same test as trap #16's deadlock, used in the opposite direction
(that one was every worker idle, this one is sustained load).

### RL Track — the one-step lookahead has no headroom, and search is the only order-of-magnitude lever (branch `feat/rl-a2-training`, worktree `zip-rl`)

Baseline first: **251 passed, 8 xfailed in 14.42s**, `ruff` clean. Nine more than the entry
below, which is the tests the connectivity commit added; the count is branch-dependent and
only means something within a session.

**Why an oracle instead of the experiment.** Handover §6.5 named the action-conditioned
connectivity feature as the one direction on the topological line not yet falsified, and P0
had just spent forty minutes of training to learn that a feature the policy never uses is
worth nothing. So this measured the ceiling first, with the weights that already exist: run
inference but delete, before the argmax, every move a one-step lookahead can already prove
is a loss. The oracle *enforces* what the feature would merely *disclose*, so it bounds the
mechanism the feature was proposed for. **The decision rule was written into the report
before any arm ran** (`reports/2026-09-05_rl-lookahead-oracle.md` §3).

Two prunes, because split is not the only loss visible one step out: the unvisited region
splitting, and the agent having no legal move left with cells still unvisited. Those are the
*only* two, so `oracle_split_trap` is the ceiling for the whole family — articulation points
and every other one-step topological feature included, not just the connectivity version.

**Result: R1 fires.** Held-out test, `deterministic=True`, both sentinels reproducing the
recorded numbers bit-identically (4×4 0.877075, 6×6 0.409500).

| arm | 4×4 (n=1,928) | 6×6 (n=2,000) |
|---|---|---|
| baseline | 0.877075 | 0.409500 |
| oracle_split | 0.890560 (+0.0135) | 0.427500 (+0.0180) |
| oracle_split_trap | 0.892635 (**+0.0156**) | 0.438000 (+0.0285) |

+0.0156 against a ±0.04 noise floor, so by the pre-registered rule the action-conditioned
feature is not worth building and **the topological-feature line is closed**, GNN's main
argument with it.

**Why the ceiling is that low — the counter that explains it.** Of the 6,467 6×6 decisions
where pruning changed the mask, **3,828 were positions where all four moves lose**. The
oracle can delete losing options; it cannot help when every option is losing. That is P0's
diagnosis seen from the other side: P0 found the policy does not avoid the fatal move when
told about it, and this run finds that forcing it to avoid the move rescues almost nothing,
because the mistake was made several steps earlier. **The failure is long-range planning,
not one-step perception.**

**best-of-N, filling the half of §7.20 that 6×6 never had.** A Zip solution is verifiable,
so this is budget, not guessing; attempts stop at the first solve, so the episode count is
the deployment cost.

| | deterministic | best-of-2 | best-of-4 | best-of-16 |
|---|---|---|---|---|
| 4×4 | 0.877075 | 0.897303 | **0.923755** (1.33 ep) | 0.954876 (2.02 ep) |
| 6×6 | 0.409500 | 0.459000 | 0.533000 | **0.649000** (7.50 ep) |

**4×4 clears its 0.90 bar at best-of-4 for 1.33 episodes a puzzle.** 6×6 gains **+0.240**,
which is larger than every training-side change this track has measured put together, and it
lands on the goal with the biggest gap. The 4×4 column also re-measures trap #19's 300-puzzle
numbers on the full 1,928 and they hold, slightly optimistic.

**One more control, because the obvious objection deserved an answer.** Does the same
lookahead become valuable when handed to *search* rather than to the observation? Measured on
the full 6×6 split: the oracle adds +0.02 to +0.03 at every N, the same size as its
deterministic gain, with no amplification. **The information is cheap either way** — take it
as a free correctness prune, but it changes nothing.

**Asked mid-session whether the network is big enough, compared against AlphaGo.** Written up
in the report §7 with sources; the short version is that model size is not the binding
constraint and there is direct evidence rather than an argument. The policy is 1,170,949
parameters, but **89.7% of them are one `Linear(4104→256)` that flattens the board** — the
convolutional trunk doing the spatial reasoning is 78,528, against roughly 22.4 M for AlphaGo
Zero's. Yet: P0 handed the network the topological answer for free and it did not help; this
run enforced perfect one-step lookahead and bought +0.016; and the task itself measures far
smaller than the parameter gap suggests — replaying ground-truth solutions, **59.4% of 6×6
decisions have exactly one legal move**, leaving **14.2 real choices per puzzle** and a search
tree of about 10^6.2 against Go's 10^170. Reading the solve rate as a per-choice accuracy,
6×6 is already **93.9% correct per decision** and loses by needing 14 of them in a row;
reaching 0.85 means cutting that error 5.4×. Capacity is not what is missing.

**Next.** The topological line is closed. The open one is inference-time search, which
design notes §5.4 already pointed at: use the policy to order the branches of the existing
`dfs.py` rather than resampling independently. Not authorised yet.

### RL Track P0 — the connectivity feature is a null, and the interesting part is why (branch `feat/rl-a2-training`, worktree `zip-rl`)

Baseline first: **243 passed, 8 xfailed in 12.99s**, `ruff` clean. That is one more than the
242 recorded earlier the same day, and the extra test is
`test_shaping_lambda_zero_is_an_override_not_an_omission` from the previous entry — the count
is branch-dependent and only means something as a within-session comparison.

**What was added.** `PuzzleEnvV2` can now append two scalars derived from one flood fill over
the unwalked region: a flag for "the region is split" and the component count saturated at 4.
Two rather than one, because the mechanism is binary — a split already loses the puzzle — and
handing the network a pre-thresholded bit removes the risk that a null result is really a
too-weak encoding, which matters when the whole point of the experiment is to be a falsification
test for the GNN route. The count is kept because it is what handover §6.5 literally asks for
and it is free once the flood fill has run.

**It is opt-in and off by default.** Turning it on takes the scalar vector from 8 to 10, and the
policy's first layer is `Linear(4104→256)`, so any checkpoint trained without it can no longer
be resumed with it on — including the 8M-step `goal2_6x6_bigdata` model. `Goal.connectivity_features`
defaults to `False` and the CLI uses `argparse.BooleanOptionalAction` with `default=None`, so
`resolve_goal` can keep the `is not None` test: `False` is falsy in exactly the way `0.0` was for
`--shaping-lambda`, and the control arm's setting has to be expressible.

**The headroom was measured before spending the forty minutes.** Running the existing control
model over 500 held-out 4×4 puzzles while computing the component count each step: all 58
failures are dead ends, **36 of them (62.1%) had the region split before termination**, median
2 steps of warning, and the flag never once fired on a solved episode. So warned failures are
7.2% of all puzzles, the ceiling is 0.884 → 0.956, and **clearing ±0.04 requires converting more
than 55% of them** — which was written into the report before any arm was run.

**Result: 3 seeds × 2 arms, 4×4, 1M steps, held-out test (n=1,928).**

| arm | seeds | mean | within-arm spread | dead ends |
|---|---|---|---|---|
| off | 0.8771 / 0.8392 / 0.8449 | **0.8537** | 0.0379 | 0.1463 |
| on | 0.8242 / 0.8247 / 0.8335 | **0.8275** | 0.0093 | **0.1725** |

The difference of means is **−0.0263**, inside the ±0.04 noise floor, so by the pre-registered
rule this is a **null**. It is not the clean null `shaping_lambda` gave, though: there the paired
per-seed differences changed sign, here all three are negative (−0.0529, −0.0145, −0.0114), the
dead-end rate rises in all three and mean coverage falls in all three. The honest statement is
"no evidence it helps, and the point estimate is negative", not "no difference".

**Both sentinels came back clean.** The control arm reproduced the historical numbers
bit-identically across all three seeds (+0.000000, whole result dicts equal), so the change did
not leak into the off arm; and per seed the greedy and masked-random baselines are identical
between arms, which is direct evidence that the option is observation-only and changes no
environment dynamics.

**The mechanism diagnosis is the real output.** Re-running the same 500-puzzle probe with the
trained *treated* model: the share of failures preceded by a visible split is **62.1% → 64.6%,
essentially unchanged**, while warned failures rise from 7.2% to 12.4% of all puzzles because the
policy is simply worse. A policy that can see the signal walks into splits at the same relative
rate as one that cannot. **So the null is not "the encoding was too weak to detect"; it is "putting
the answer in the observation does not make the policy avoid the move."** The explanation — an
explanation, not a measurement — is that the signal is one step late: at decision time the current
state is not yet split, the split only appears in the *next* observation, so avoiding it still
requires the same one-step lookahead the policy needed without the feature. The feature can sharpen
the value function; it hands the policy nothing actionable. The version that would be actionable is
action-conditioned — for each of the four moves, would *this* move split the region — and it fits
in the same flood-fill budget. That is the one direction on this line that has not been falsified.

**One alternative explanation is not excluded.** The treated arm's curriculum reaches full length
later in all three seeds (340,288 → 388,619 steps on average, +14.2%), so it trains about 7.4%
less at full length. "The feature makes the final policy worse" and "the feature slows the
curriculum, so full length gets less training" cannot be separated from these runs. Topping the
treated arm up by the missing ~50k steps would separate them in about four minutes; not done.

**Defect: the flag reached training but not evaluation (handover trap #24).** All three treated
runs trained perfectly — 1M steps, curriculum to full length, every checkpoint written — and then
**crashed during scoring**, with a traceback that ends deep inside SB3 (`policies.py:258
obs_to_tensor` → `utils.py:489 is_vectorized_observation`) and looks nothing like "your flag is
half-wired". `baselines.evaluate()` builds its own env and knew nothing about the option, so a
policy trained on 10 scalars was scored against an 8-scalar env. Nothing was lost — `model_final.zip`
is saved *before* `score()` — so the scoring was replayed from the checkpoints. Two things were
wrong in kind, not just in detail: the environment has **two** construction sites and only one was
wired, which is the same shape as the resource-budget defect that missed `generate_dataset_v2`'s
pool; and the verification done that morning covered CLI → goal → training env and felt like
"verified by effect", when the invariant that actually needed pinning is one sentence —
**the training env and the evaluation env must agree on the observation space** — which covers both
paths at once. Evaluation now builds envs only through `baselines.make_eval_env()`, and
`test_training_and_evaluation_envs_agree_on_the_observation_space` asserts the invariant for both
arms. The fix was then verified by effect: re-scoring the control run under the patched code
returns a result dict identical to the one the original run wrote.

**Cost.** Treated runs took 261.7s against the control's 244.7s, **+7.0%**, against the +4.6%
measured for an equivalent extra flood fill on 2026-09-05. The control arm's own three runs span
12.9s (5.3%), so the two numbers agree on magnitude rather than contradicting each other.

**Where this leaves the priority list.** P0 is closed and does not go to 6×6. The GNN stays
deferred, and its reason is now stronger than "not yet falsified": a GNN computes graph properties
of the *current* state too, so it inherits the same limitation this experiment exposed. What it
still cannot speak to is the one advantage a GNN would really have, generalisation across board
sizes, which the current `Linear(4104→256)` forecloses. Report:
`reports/2026-09-05_rl-connectivity-feature.md`.

### RL Track — the budget hypothesis holds, and what a run this small does to a 16 GB card (branch `feat/rl-a2-training`, worktree `zip-rl`)

Baseline re-measured before touching anything: **242 passed, 8 xfailed in 20.44s**, `ruff`
clean. The number is larger than the 214 the handover records for `main` because this branch
carries the A2 tests on top of it; the handover already warns that the count is branch-dependent
and only useful as a within-session comparison.

**Only one thing was changed.** The handover's next-step list puts "add budget" first and
`shaping_lambda=0` second, explicitly not together, so this run resumed `goal2_6x6_bigdata`
from 5,005,312 steps / k=27 for 3,000,000 more and left every other knob alone.

`--resume` overwrites `model_final.zip`, `train_state.json` and `eval_test.json` under the same
run id, so the previous round's evidence was copied to `*_at5005312.*` first. `progress.jsonl`
appends, so the k=27 curve survived on its own. The resume itself was verified from the log
rather than from the code, because defect #10 of the last round is precisely a resume that
silently restarts the curriculum: `Resuming goal2_6x6_bigdata from model_final.zip at 5005312
steps, k=27`.

**Result: `Trained 3,000,000 steps in 749.5s (4002 fps), k=30, promotions=9`.**

| | previous (5M) | this run (8M) |
|---|---|---|
| held-out test solve | 0.344 | **0.409** |
| dead end | 0.656 | 0.591 |
| coverage | 0.626 | 0.678 |
| curriculum k | 27/36 | **30/36** |

Both controls are unmoved (greedy 0.004, masked random 0.000), the same values as 2026-08-15 and
2026-08-29, so the evaluation protocol is aligned across three rounds and the numbers compare
directly. The 0.85 target is still far away, but the question this run had to answer was whether
more budget buys anything, and it does.

**The promotion that had stalled took 2,867,776 steps.** The cost per curriculum level is now
416 → 3,744 → 47,376 → 76,624 → 168,976 → 289,696 → 812,944 → 1,307,280 → **2,867,776**, i.e.
roughly ×2 per level over the last three. k=30 has already consumed 2,436,944 steps without
promoting, which is consistent with that shape rather than evidence against it: its rollout
success rate is still climbing monotonically (0.679 → 0.712 → 0.731 → 0.745 → **0.764** across
five equal slices of its 298 progress rows, threshold 0.90) — the same shape k=27 had before it
promoted. Extrapolating puts full length at roughly 15–30M more steps, one to two hours, **but
that extrapolation is explicitly untrustworthy**: the same reasoning was off by three times last
round, and it is recorded here as an order of magnitude to argue about, not a prediction. The
alternative explanation — a capacity ceiling that makes the cost diverge rather than double —
has not been ruled out, and the cheapest test is simply what 30→33 ends up costing.

**A 16 GB card is asleep during this.** Sampled during training the GPU sits at 36–40%
utilisation; sampled again after it exits, **0–5%**, so the ~38 points really are the run and not
the desktop that shares the card. But utilisation counts *time with a kernel resident*, not
occupancy: a 1.17M-parameter policy launching small kernels at high frequency looks busy and
computes almost nothing, and both rounds landed at the same throughput (3,968 vs 4,002 fps),
which is what a CPU-bound loop looks like. Rebuilding the extractor and measuring directly:
weights 4.31 MiB, 30.21 MiB allocated after a forward/backward/Adam step, **115.58 MiB peak** at
batch 512 — against 13.0 GiB for a 7B model's fp16 weights alone. Nearly all of the parameters
sit in the single `Linear(4104→256)` (~1.05M of 1.17M); the three convolutions hold ~77k. The
rollout buffer never reaches the card at all — SB3 keeps it as `np.zeros` on the host
(`buffers.py:392`) and moves one minibatch at a time. The 12,281 MiB cap in `ResourceSettings`
is therefore a guard rail, not a constraint anything currently approaches.

**And the loop is not bound by the environment, which contradicts what this log has been
saying.** The handover records "the bottleneck is the single-threaded Python env step, not the
GPU", inferred from two correct observations — one core busy, 84 MiB of VRAM — but the inference
does not survive measurement. `env.step()` including masks and observation assembly has a median
cost of **10.40 us**, while 4,002 fps gives each step a **250 us** wall-clock budget: the step is
about **4%** of the loop. An A/B settles it without arithmetic. Running the real training loop
for 40,000 steps per arm, identical except that every step also runs a connectivity BFS whose
result is thrown away, costs **4.6%** (3,825 → 3,657 fps) against **0.3%** run-to-run noise on
the baseline. Nearly tripling the cost of a step cannot cost 4.6% of a loop that step dominates.
What fits all three observations instead is that the one busy core is spending its time
*launching small CUDA kernels*, which is also what 36-40% utilisation at 84 MiB looks like. Only
the negative is established here — the remaining 95% has not been profiled, so whether it sits in
torch or in SB3's Python plumbing is still open. If it is launch overhead, the way to go faster
is bigger and fewer kernels (`n_envs`, `n_steps`, `batch_size`), which is the opposite of what
this log previously pointed at.

**Design notes were written down, on purpose.** The owner restated on this date that the point of
this side project is learning by doing, not the metric, so the session's design questions — why
the memory footprint is that small, what the 8 channels are, why `MaskablePPO` was chosen, how
all of it maps onto Go engines — went into
`reports/2026-09-05_rl-budget-and-design-notes.md` §5 instead of evaporating with the
conversation. The short version: the 8 planes *are* AlphaGo-style feature planes (AlphaGo used
48, AlphaGo Zero 17, this uses 8, and ours are closer to Zero's in character since none of them
encode heuristics); `MaskablePPO` was picked for masking and toolchain availability, not for any
Go lineage, though restart plan §9 already schedules an `AlphaZero-lite` comparison as A6; and
the measured best-of-N result (deterministic 0.870 → best-of-16 0.967) is the same
"policy prunes the search" idea that AlphaGo rests on, reachable here far more cheaply by
ordering the existing DFS with the policy than by writing an MCTS.

**A GNN was proposed and declined for now, on the same measurement.** Swapping the CNN for a
graph network is meant to buy topological awareness, but connected-component count and
articulation points *are* the topological answer, and computing them in the environment costs
5.5-18 us — the 4.6% already measured. If handing the policy the answer directly changes nothing,
a network that must spend several message-passing layers deriving that same answer will not do
better, so the scalar feature is a falsification test for the GNN at two orders of magnitude less
work. It is a necessary condition rather than a sufficient one: the one thing a GNN could still
win is generalisation across board sizes, since message passing is not tied to the 8×8 padding
that `Linear(4104→256)` is built around. Two claims made for the GNN do not hold here either.
This board's graph topology is fixed for the whole episode — walls and blocked cells never move,
only the visited mask does — so `edge_index` is built once at reset and shapes stay static;
edges only churn if one chooses to delete visited nodes, which is a design choice. And PyTorch's
caching allocator does not return freed memory to the OS, which is why `empty_cache()` exists, so
varying shapes cost fragmentation and re-planning rather than a syscall per step. The feature is
worth trying on its own merits regardless: 59.1% of held-out failures are dead ends, and the
environment only terminates when all four directions are blocked, which is the *local* dead end —
the puzzle is already unsolvable the moment the unvisited region splits in two.

**One documentation trap worth naming.** `handover-rl-solver.md` exists in every worktree, and
the copy in a *different* worktree is whatever that branch last committed — the `zip-vlm` copy
still describes A2 as unstarted. Read the handover from the worktree whose branch owns the
track, or read a nine-month-old plan by mistake.

### RL Track — the shaping control returns nothing, and hands over a noise floor instead

The handover's next step was to run the environment with `shaping_lambda=0`, because the restart
plan specifies zero for the one-stroke phase while the implementation has always run 0.2 — 14% of
an episode's return, never once tested. Doing it needed a `--shaping-lambda` override, added
alongside `--timesteps` and `--dataset` rather than by editing the goal, so the default is not
silently changed for every later run. `resolve_goal` tests `is not None`: 0.0 is falsy and is
exactly the value the control needs, so a truth test would have made the experiment a no-op with
perfectly healthy curves. `test_shaping_lambda_zero_is_an_override_not_an_omission` pins that.

One seed said shaping was worth 0.038 of held-out solve rate. Three seeds per arm said otherwise.

| arm | 20260815 | 31415926 | 27182818 | mean | spread |
|---|---|---|---|---|---|
| λ=0.2 | 0.877 | 0.839 | 0.845 | 0.8537 | **0.0380** |
| λ=0 | 0.839 | 0.833 | 0.863 | 0.8450 | 0.0300 |

The arms differ by 0.0087 in the mean while λ=0.2 spans 0.0380 within itself, and the paired
per-seed differences change sign (+0.038, +0.006, −0.018). Lambda is not measurable here. The
spec deviation closes with 0.2 kept and nothing needing to be redone.

**The by-product is worth more than the experiment.** This is the first measurement of what a
single seed is worth on this track: **±0.02–0.04 at 4×4**. The greedy baseline across all six
runs spans 0.002, so essentially none of that is evaluation noise — it is training seed, which is
consistent with trap #19 finding deterministic replay bit-identical. That reaches backwards. The
0.877 in the handover table is the luckiest of three seeds and the honest figure is 0.854 ±
0.038, which puts 4×4 further from its 0.90 bar than the table implied; the 6×6 resume's +0.065
sits close to the noise and should be read as moved-but-unconfirmed; the +0.089 from enlarging
the dataset is comfortably above it and still stands.

**Priorities were re-ordered against the original goal on the owner's instruction.** The track
exists to train a policy that solves unseen puzzles — 0.90 at 4×4, 0.85 at 6×6, eventually a
tenth solver behind the API — and a session spent on measurement had drifted from that. Handover
§6.5 now states one P0 and an explicit do-not-do list. P0 is the connectivity feature: 59.1% of
held-out failures are dead ends while the environment only terminates once all four directions
are blocked, which is the local dead end and far too late, since the puzzle is already lost the
moment the unvisited region splits. It costs the 4.6% already measured, it is the only candidate
with a mechanism rather than a knob, and it doubles as the falsification test for the GNN.
Pushing 6×6 to full length, the GNN, PPO hyperparameters and a shaping sweep are all listed as
deliberately deferred, with reasons, so they are not reopened. The rule that decides all of it:
an effect smaller than ±0.04 is not an improvement.

## 2026-08-29

### RL Track A2 — the first training run, and three defects that never raise an error (branch `feat/rl-a2-training`, worktree `zip-rl`)

Branched from `main`, which already carries the merged `feat/rl-masked-ppo` work, so the A1
baseline was re-measured rather than assumed: **214 passed, 8 xfailed**, `ruff` clean,
`MaskablePPO` imports, CUDA available on an RTX 4070 Ti SUPER. `tensorboard` was already
installed, so A2 needed no new dependency.

Added `src/core/rl/train_config.py` — the single place a goal is defined (board size, wall
policy, step budget, done condition, and the PPO / network / curriculum / resource settings) —
plus `src/core/rl/train_maskable_ppo.py` and `src/core/tests/rl/test_train_maskable_ppo.py`.
`rl_env_v2.py` gained a public `observation()`, and `baselines.evaluate()` now also accepts a
callable, so a trained model and the two controls are scored through **one** accounting path
instead of two copies of it.

**Three defects that produce no error message.** All three were found by running, not by reading.

-   **SB3 was flattening the board.** `MultiInputPolicy` chooses between `NatureCNN` and a plain
    `Flatten` via `is_image_space`, which requires `uint8` in 0-255; our grid is `float32` in
    0-1, so `CombinedExtractor` reduced the 8×8×8 observation to a 512-vector and trained an MLP
    on it. No warning, no error — the board geometry was simply gone. Forcing the image path
    (`normalized_image=True`) then *does* fail, with `Calculated padded input size per channel:
    (1 x 1). Kernel size: (4 x 4)`, because `NatureCNN` opens with an 8×8 stride-4 convolution.
    The handover predicted the crash but not the silent path, and the silent one is the
    expensive one. `GridScalarExtractor` (three padded 3×3 convolutions, no pooling, 1,170,949
    policy parameters) replaces both. Pinned by `test_sb3_would_flatten_the_grid`.
-   **Gymnasium 1.x removed wrapper attribute pass-through.** `Monitor(env).action_masks()`
    raises `AttributeError`; `VecEnv.env_method` survives only because SB3 routes it through
    `get_wrapper_attr`. The probe written to de-risk exactly this missed it, because it
    exercised construction but never called the mask lambda — a probe that does not touch the
    hot path proves less than it looks like it proves. The smoke run caught it in seconds.
    Masks now go through `read_action_masks`, which unwraps.
-   **A resume restores the model but not the curriculum.** The SB3 zip carries policy and
    optimiser state and nothing about `reverse_curriculum_k`, so resuming from the model alone
    silently restarts the curriculum at `k_start` while every curve still looks healthy.
    `train_state.json` is now written beside each checkpoint, and the round trip was verified
    by doing it: saved at 8,192 steps / k=6, resumed, continued to 12,288 steps / k=9,
    promotions 1→2, episodes 1,858→2,469.

**Measured costs (100k-step pilots).**

| run | wall clock | fps | GPU peak |
|---|---|---|---|
| 4×4, `DummyVecEnv` | 25.8s | 3,880 | 83.8 MiB |
| 4×4, `SubprocVecEnv` | 31.0s | 3,221 | 83.8 MiB |
| 6×6, `DummyVecEnv` | 27.4s | 3,647 | 83.8 MiB |

`SubprocVecEnv` is **slower** here, and both produce identical promotion steps and identical
solve rates. The environment step is cheap single-threaded Python — the training process uses
about one core of 24 — so Windows process IPC costs more than the parallelism returns. The
restart plan's §4.8 `n_envs = 16 (SubprocVecEnv)` is therefore wrong for this environment;
`vec_env="dummy"` now lives in `ResourceSettings` with the measurement as its justification.

**Resource limits.** 75% headroom was requested so the machine stays usable.
`apply_resource_limits` scales PyTorch's *own* default intra-op thread count (torch sets it to
the physical core count, 12 of 24 logical here) and caps GPU memory: **9 threads, 12,281 MiB of
16,384**. Actual use is nowhere near either — about one core, 1.42 GiB RSS of 63.1, and **83.8
MiB of GPU** — and capping made it marginally *faster* (4,036 vs 3,880 fps), confirming the
workload is not thread-bound. psutil was deliberately not used: it is present only as a
transitive dependency of another track, and this repo requires dependencies to be declared.

**Checkpoints are never pruned** — the VLM track lost an entire experiment to
`save_total_limit=2` — so disk is controlled by the *interval* instead: one checkpoint is 14 MB
and `checkpoint_every` is set per goal to land 25-50 of them.

**Evaluation protocol.** The model plays each puzzle once (`deterministic=True` is an argmax
from a fixed start, so repeats would be identical) while the sampling baselines play 20 episodes
per puzzle, which is how the 2026-08-15 baseline table was produced. The controls reproduced
those numbers exactly on the 4×4 test split — masked random **0.088**, greedy **0.102** against
the recorded 8.8% and 10.2% — which is the strongest available evidence that the two protocols
are comparable.

**Datasets: shared generator, different wall distributions.** Both tracks call
`puzzle_generation.generate_puzzle`, so boards and numbers are drawn the same way, but the RL
set takes the generator's own walls (absent, or the hard-coded 2-5 of
`puzzle_generator.py:10-11,123`) while the VLM builder asks for a wall-free board and samples
**0-12** itself, because real 6×6 screenshots carry 0, 4, 4 and 10. A5 will feed the RL solver
boards read by the VLM parser, so those boards are out of distribution for it. Recorded now,
not fixed.

**goal1 (4×4, 1,000,000 steps): the curriculum finished, the done condition did not.**
Full length (k=None) was reached at **152,064 steps** after 5 promotions, and the run took
**248.0s at 4,032 fps**. Held-out test solve rate is **0.788** against a target of 0.90 — 7.7×
masked random and 7.5× greedy, but short of the bar. Promotion cost grew steadily
(416 → 13,680 → 20,144 → 29,312 → ~88,500 steps, roughly ×1.45 per step of k), which is the
number to watch on larger boards. The gap worth recording is internal: at full length the
training-split rollout success ran at **0.94-0.95** while held-out deterministic play scored
**0.788**. That is precisely the "training-time scores do not count" failure the track plan
warns about, and the first concrete evidence for handover §9's open question about dataset size
(1,360 training puzzles at 4×4).

**goal2 (6×6, 5,000,000 steps): the curriculum ran out of budget three cells short of the start.**
1,203.1s at 4,156 fps, 10 promotions, ending at **k=33 of 36** — the final transition, to the
true start cell, never happened. Held-out test solve **0.253** against a target of 0.85, which is
34× greedy (0.0074) and 281× masked random (0.0009); both controls reproduced the 2026-08-15
table (recorded 0.8% and 0.0%). Promotion cost per step of k was
416 → 960 → 41,584 → 53,024 → 57,312 → 130,432 → 314,208 → 445,168 → 690,736 → 662,064.

**It is not stalled, it is under-budgeted.** At k=33 the rollout success rate rose monotonically
for the remaining 2.6M steps — 0.518 → 0.605 → 0.684 → 0.728 → 0.788 → **0.800** at the cutoff,
against a 0.90 promotion threshold — with dead ends falling 0.482 → 0.200 in step. The last
stretch gains about +0.02 per 250k steps, so roughly another 1.2M steps would plausibly clear the
promotion, and `--resume` carries the curriculum so that is a continuation rather than a restart.
Two readings made while the run was in flight were wrong and are recorded as such: an
extrapolation of "×1.29 promotion-cost growth ⇒ full length by ~1.3M steps" was off by a factor
of three, and "it has stalled at k=27" was simply false — it promoted at 1.73M.
**Curriculum progress cannot be extrapolated from its own early segments**, which is the same
shape of error the VLM track made extrapolating s/step from a short run.

**The finding that matters: both goals are memorising.** Scoring the final policies
deterministically on the *training* split as well as the held-out split separates the two
candidate explanations for goal1's 0.95-versus-0.788 gap:

| goal | deterministic, train split | deterministic, test split |
|---|---|---|
| goal1 4×4 | **0.947** | 0.788 |
| goal2 6×6 | **0.553** | 0.253 |

Deterministic play on the training split matches the training rollout curve, so the gap is **not**
argmax-versus-sampling. The other candidate explanation — that the test split is simply harder —
was ruled out by running both controls, which have seen neither split, over the same two sets:

| policy | 4×4 train / test | 6×6 train / test |
|---|---|---|
| masked random | 0.0753 / 0.0876 (z −1.86) | 0.0000 / 0.0009 (z −1.73) |
| greedy | 0.1038 / 0.1018 (z +0.28) | 0.0041 / 0.0074 (z −1.77) |

Every one of the four says the test split is as hard or marginally *easier*, which pushes against
the confound rather than towards it, while the model's own gaps are large (z +4.32 and +5.64 on
170 puzzles a side). So it is generalization.

⚠ **What this does not yet prove.** The training subsample was 170 of 1,360 puzzles and only one
seed was trained, so "not enough data" is the best-supported explanation, not the only possible
one — network capacity and the absence of any regularisation are untested. The actual proof is
that the gap narrows when the dataset grows, which is why that is the done condition in
handover §6 rather than a claim made here. 1,360 training puzzles per size is not enough.
That settles the open question recorded in handover §9 ("if A2 shows obvious overfit, go back and
enlarge the dataset"): it does, so the next move is **a larger dataset before a larger step
budget** — training longer on 1,360 puzzles only memorises them harder. Generation is cheap
(5,100 puzzles in 45 seconds), so this is minutes of work.

**Decisions taken this session, so they are not re-litigated.**

-   **Two goals, both reported.** 4×4 was first proposed as a throwaway smoke test; the
    developer made it `goal1` in its own right, with 6×6 as `goal2`. Both numbers are
    published, not just the 6×6 ones.
-   **6×6 trains on all 1,360 puzzles, walled ones included**, rather than the 668 wall-free
    ones. The observation already carries `wall_right` / `wall_down` channels — excluding
    walls would leave them permanently zero — and episode difficulty is set by `k`, not by
    walls, so keeping them does not impede the early curriculum.
-   **`GRID_PAD` stays 8.** It is a padded canvas, not a board size: a 6×6 board sits in the
    top-left and `valid_mask` says which cells are real. A fixed input shape is exactly what
    lets goal1's weights carry into goal2 (plan A3 requires continuing, not reinitialising)
    and 7×7 later (A4). Shrinking it would break both and force a rewrite of the 21 env tests
    for a saving of 84 MiB of GPU that is not under any pressure.
-   **`DummyVecEnv` over `SubprocVecEnv`**, on measurement rather than on the restart plan's
    recommendation.
-   **No psutil in production code.** It would have made the resource accounting easier, but
    it is present only as a transitive dependency of another track, and this repo requires
    dependencies to be declared and installed by the developer.
-   **Datasets are identified by digest, and a new pack gets a distant base seed.** Reusing
    `base_seed=20260815` for the larger set would have made its 4×4 half a shifted copy of the
    existing pack — the same trap the VLM track hit with seeds one apart — so the new pack
    starts at 20300000, past the old range entirely.

**Lessons, each paid for in this session.**

1.  **A resource budget has to cover every process the work starts, not just the one you had
    in mind.** The developer's "keep it under 75% so I can still use the machine" was wired
    into training — torch threads and GPU memory — while `generate_dataset_v2` kept
    `Pool(processes=None)`, which is `os.cpu_count()`. Dataset generation then ran 24 workers
    and pegged the machine at 99-100%, which the developer noticed before I did. The budget
    now lives once in `train_config.DEFAULT_CPU_FRACTION` and both call sites read it.
    *Next time:* when a limit is agreed, grep for everything that spawns work — `Pool`,
    `set_num_threads`, `n_envs`, subprocesses — before reporting that the limit is in place.
2.  **"75% of the cores" is not "75% CPU".** The first fix budgeted `int(24 × 0.75)` = 18
    workers and still measured 74-82% system CPU, because the parent process feeds tasks and
    collects results while the OS takes its own cut. Two workers are now reserved for that,
    and 16 workers measured a mean of 68% with a single 77% spike.
    *Next time:* a resource limit is not in place until it has been **sampled while the work
    is running**. Computing it is not measuring it.
3.  **A train/test gap is not evidence of overfitting until the two splits are shown to be
    equally hard.** The memorisation claim went out before that control existed, and the
    developer challenged it. Running both baselines — which have learned nothing from either
    split — over the same two sets showed the held-out set is as hard or marginally easier,
    which is what makes the model's own gap mean something.
    *Next time:* pair every "the model generalises worse than it fits" claim with a policy
    that could not have generalised or fitted.
4.  **A generator that gives up on wall-clock time is not reproducible from its arguments.**
    `generate_dataset_v2`'s docstring claimed "the same arguments reproduce the same dataset";
    they do not, because `generate_puzzle` abandons its search on elapsed time and a clipped
    attempt is retried under a different derived seed. The VLM track had already measured this
    on the shared generator. The claim is corrected, and the manifest now carries a SHA-256
    over the canonical *content* of each split — not over the pickle bytes, which would hash
    the encoding rather than the puzzles — with `--verify` to recheck it.
    *Next time:* a reproducibility claim in a docstring is a claim like any other, and needs
    the same evidence as one in a report.
5.  **A probe that exercises construction but not the hot path proves less than it looks
    like.** The probe written specifically to de-risk the wrapper stack never called the mask
    lambda, so it missed that Gymnasium 1.x had removed attribute pass-through; the smoke run
    found it immediately.
6.  **Neither curriculum progress nor cost per step survives extrapolation from its own early
    segments.** Both mid-run readings were wrong — "×1.29 growth ⇒ full length by 1.3M steps"
    missed by three times, and "stalled at k=27" was contradicted at 1.73M steps. The
    numbers in a report should come from the finished run, not from a log being watched.

7.  **Many worker processes writing to one stderr pipe deadlock, and the symptom looks like
    slowness.** Building the larger dataset hung twice. The state was diagnostic once
    looked at properly: all 16 workers idle with near-identical CPU time (415.0-417.1s),
    the parent idle at 10.1s, every thread in `UserRequest` wait, and nothing written for
    fifteen minutes — that is everyone blocked on one thing, not work in progress. The
    accumulated 6,640 CPU-seconds also already exceeded the ~5,500 the whole job needs, so
    the computation was finished and only the output was stuck. `generate_puzzle` logs a
    line per attempt and a 40k build emits **19.8 MB and ~90k lines** from 16 processes onto
    one inherited stderr; when that stderr is a pipe, the writers deadlock. Three controls
    isolate it: piped with worker logging **hangs past 120s**; the same pool writing to a
    file finishes in **2.2s**; the same pipeline with worker logging disabled finishes in
    **2.1s**.
    *Two hypotheses were disproven on the way, both mine:* that `grep`/`tail` stall on a
    carriage-return stream with no newline (a single writer pushes 200k updates through in
    461 ms) and that `grep` degrades as the unterminated line grows (it is linear:
    131/305/461 ms at 20k/100k/200k). The variable was the **number of concurrent writers**,
    not the format of the output.
    *And the first fix did not work.* `logger.disable(...)` was added to `build_dataset`,
    which runs in the parent — under `spawn` a child never executes it, and the log stayed
    at 19,787,873 bytes, unchanged. Moving it into `_generate_one` took an equivalent build
    to 1,868 bytes and zero worker lines, and a 10,000-puzzle build through the exact
    pipeline that had deadlocked now finishes in 109s.
    `vl_models/dataset_builder.py:291` already disables the same logger inside its worker,
    with the same comment, and its docstring records the same Windows hang. **The shared
    generator's logging has now cost two tracks; the RL builder simply never got the
    treatment the VLM builder wrote down.**

8.  **Small boards run out of distinct puzzles, and the duplicates leak across splits.**
    Asked for 20,000 per size, 4×4 returned 97.2% unique and put **111 puzzles in both train
    and test — 5.57% of the held-out split** — which would have flattered exactly the
    generalization measurement the larger dataset exists to make. 6×6 was 100% unique. The
    builder now deduplicates by content fingerprint before splitting, so the splits are
    disjoint by construction; the new pack drops 726 duplicates at 4×4 and none at 6×6, and
    all pairwise split intersections are zero. The old pack was re-checked and had **no**
    train/test overlap at 1,700 per size, so the A2 conclusion is unaffected.
    *This was the developer's question, not my check.* Asking for more data without asking
    how much of it is new is how a dataset gets bigger without getting more informative.

9.  **A fix is not in place until its effect is measured.** Both of this session's resource
    and logging fixes looked correct in the diff and were wrong: the CPU cap computed 18
    workers and measured 74-82%, and the logger was disabled in a process that does not
    spawn the writers. In both cases a single measurement — sample the CPU, look at the log
    size — settled it in seconds.

**Retrying a puzzle only helps if the inference mode changes, and then it helps a lot.**
`deterministic=True` is an argmax from a fixed start, so a replay is bit-identical: across 300
held-out 4×4 puzzles, two evaluations seeded differently produced **the same action sequence
300/300 times**. Repeating a deterministic evaluation is therefore pure waste, which is why the
model plays each puzzle once while the sampling baselines play twenty. Sampling
(`deterministic=False`) walks differently each time, and because a Zip solution is *checkable* -
run it against the rules, no judge model required - best-of-N is a legitimate inference-time
strategy rather than guesswork. The same goal1 policy on the same 300 puzzles:

| inference | solve |
|---|---|
| deterministic | 0.870 |
| sampled, 1 attempt | 0.863 |
| sampled, best-of-2 | **0.903** |
| sampled, best-of-4 | 0.930 |
| sampled, best-of-8 | 0.947 |
| sampled, best-of-16 | **0.967** |

One sample scores *below* argmax, as expected, but two attempts already beat it because the two
failures differ. The median attempt count when a puzzle is eventually solved is **1**, so the
extra budget is spent almost entirely on the hard tail.

**This exposes a problem with the done condition rather than solving one.** The same policy
misses the 0.90 bar deterministically and clears it at best-of-2, so "did it pass" is currently a
function of an inference setting the bar never specified. The 0.90 has no derivation anywhere in
the plan or the restart report: it is inherited from the superseded three-phase design, where it
was a *soft* success rate used as a **phase-promotion** threshold, and it now collides with
`CurriculumSettings.promote_threshold`, a different 0.9 measuring stochastic rollouts on the
*training* split. Two numbers should be reported from here on - deterministic solve rate for
comparing training configurations, since it has no inference-budget variable in it, and
best-of-N with N and the mean attempt count for what the solver is actually worth in use. The
plan's "evaluate deterministically" rule was written to stop training-time scores being quoted as
results; it is not an argument against reporting a sampled inference budget as well.

⚠ These numbers are on 300 of the 1,928 test puzzles and are **not** comparable with the 0.788 /
0.877 headline figures, which used the full split. The 0.870 here and the 0.877 there differ by
less than the subsample error, and no claim is made that they differ at all.

The dataset the next run trains on is `seed20300000_n20000_4-6`: 15,419 / 1,927 / 1,928 at 4×4
and 16,000 / 2,000 / 2,000 at 6×6, roughly 11x the training rows A2 had, digests verified.
`DEFAULT_DATASET` now points at it, and `--dataset main_n1700_456` reproduces an A2 number.

Suite after A2: **242 passed, 8 xfailed**, `ruff check` clean. Nothing outside `src/core/rl/` and
`src/core/tests/rl/` was modified.

### The enlarged dataset settles the diagnosis, and hands 6×6 a different bottleneck

Both goals were retrained on `seed20300000_n20000_4-6` - about 11x the training rows - with the
same budgets, the same protocol, and the same evaluation split conventions as A2.

| | A2 (1,360 / size) | enlarged (15,419 / 16,000) |
|---|---|---|
| 4×4 held-out test | 0.788 | **0.877** |
| 6×6 held-out test | 0.253 | **0.344** |
| 4×4 gap (train − test) | +0.162 | **+0.050** |
| 6×6 gap (train − test) | +0.250 | **+0.009** |

The gap is the result, not the solve rate. It collapsed on both boards, and it collapsed in the
shape that means generalization rather than luck: **held-out went up while the training split went
down** (4×4 0.950 → 0.927, 6×6 0.502 → 0.352). A policy that stopped memorising is exactly what
that looks like. The done condition handover §6 set for this experiment is met.

**The consequence is that 6×6 now has a different bottleneck.** At a gap of +0.009 there is
essentially nothing left to overfit, so more data will not move it. What will: the curriculum
reached only **k=27 of 36** inside 5M steps - shallower than A2's k=33, which is expected when the
same budget has to cover eleven times the variety - and at k=27 the rollout success rate was still
climbing monotonically when the budget ran out: 0.704 → 0.732 → 0.762 → 0.790 → **0.820** over the
final 2.3M steps, against a 0.90 promotion threshold. So the next lever is budget, and `--resume`
continues the curriculum from k=27 rather than restarting it. Neither goal met its target (0.877
against 0.90; 0.344 against 0.85) - though see above for what that bar is actually worth.

**Tracing where the reward came from surfaced a divergence nobody had recorded.** The restart
plan's phase table specifies the shaping weight per phase: λ = 0.5, then 0.5 → 0.2, then **0 for
the one-stroke phase** ("只剩 +1 與 γ"). The 2026-08-15 decision made every episode one-stroke,
which is permanently that phase, so by the design's own spec shaping should be off. `PuzzleEnvV2`
ships `shaping_lambda=0.2` and has never been run without it; measured, that is 14% of an episode's
return (4×4 +0.171 of +1.171; 6×6 +0.158 of +1.158). Handover §9 recorded the value as untuned but
not that the design called for zero. Untested in either direction - the env already accepts 0.0, so
this is one run to settle.

For the record, since the question came up: the ice-lake reward (success +1, everything else 0,
speed expressed through γ) and the switch of the shaping potential from distance-to-next-number to
coverage were both proposed in the restart plan report, which credits the developer with the
"allow backtracking" and "visit counts in the observation" ideas that the same report was written
around - and both of those were later overturned by the developer's own one-stroke decision. Git
authorship settles none of this: every commit in this repo carries the same identity.

**Lesson 9 claimed a third victim in the same session.** The first run of the train-vs-heldout
diagnostic produced numbers that contradicted the training run's own evaluation - 0.754 where the
run had reported 0.877, on the same model and the same split. Rather than report a number that
could not be explained, reloading the checkpoint and re-running the trainer's own `score()`
reproduced **0.8771 exactly**, which located the fault in the diagnostic rather than the run: its
checkpoint path had been edited by a string replacement that matched single quotes against source
using double quotes, so the edit did nothing and the script kept scoring the A2 model - while
printing a line announcing that it had been repointed. It now takes the run id as a parameter and
prints the path it actually loaded.

### VLM Track: the model is out of Drive and into the product, and it reads real screenshots better than expected

Reports: [reports/2026-08-29_vl-p4d-export-and-integration.md](reports/2026-08-29_vl-p4d-export-and-integration.md)
(export, the two dead ends, real-screenshot numbers, wiring) and
[reports/2026-08-29_vl-training-reproducible.md](reports/2026-08-29_vl-training-reproducible.md)
(how the training was actually done, reproducibly). User-facing how-to:
[vlm-operating-guide.md](vlm-operating-guide.md).

Baseline first: `uv sync` clean, `uv run pytest` **167 passed, 8 xfailed**, `ruff` clean.

**The adapter was never lost, just misfiled.** A full-disk search found no copy, and the Drive
MCP had no scope; the operator then downloaded it. Google Drive splits a large folder into
several zips **and puts the big file in a different one from its config files** —
`adapter_model.safetensors` was in `-002`, everything else in `-001`. Both have to be expanded
into one tree. Verified it is the right artifact: the training tar and the 200 predictions both
match the SHA-256 recorded in the P4c report, and the adapter carries **688 tensors, visual 96
pairs / language 248 pairs** — exactly the `96/96` and `248/248` the P4c notebook printed.

**One correction to the record**: the handover said checkpoints 200/400/600/800/975 were on
Drive. `save_total_limit = 2` had deleted all but **800 and 975**, so the "would checkpoint-200
also score 200/200, i.e. was 4/5 of the training wasted" experiment **can no longer be run**.
`save_total_limit` decides which questions stay askable, not just disk usage.

**Merging without peft.** The adapter's base needs `transformers` 5.x while this project pins
`<5`; upgrading the whole stack to add two matrices is not worth it, and the arithmetic does not
need it. `src/core/vl_models/merge_lora.py` reads and writes the tensors directly, implements
only the plain case, and **refuses** every variant it does not implement rather than merging it
wrongly. 344 tensors in 14 seconds; the per-tensor audit shows `visual 96/96` moved with
`max|delta| 2.24e-2` against `language 248/248` at `6.03e-3` and **zero non-target tensors
changed** — the vision stack moved more, matching the `max|B|` 0.272 vs 0.166 measured during
training by an entirely different route.

A trap worth naming: the obvious rule "prefer the adapter's copies of the config, since that is
what training used" is **wrong here**. The trainer re-serialised `tokenizer_config.json` with
transformers 5.2.0, naming a tokenizer class no 4.x tool can load, which breaks GGUF conversion.
Measured: `chat_template.jinja`, `tokenizer.json` and `processor_config.json` are **byte-identical**
between base and adapter, so nothing needed preserving. The merger now uses the base's copies and
**verifies those three byte for byte, aborting on a mismatch** — that mismatch would be the
train/inference rendering trap that has cost this track two rounds already.

**Two export dead ends, recorded so nobody repeats them.** `ADAPTER` pointing at the safetensors
adapter is refused because the base must also be a safetensors directory. And
`ollama create --experimental` on the merged directory **imports successfully** — 738 tensors,
747 layers, `ollama show` even reports `vision` — then dies on first use with
`mlx runner failed: MLX not available`. Unquantized safetensors are served by the **MLX runner**,
which does not exist on Linux/NVIDIA. GGUF is the only route. llama.cpp already registers
`Qwen3_5ForConditionalGeneration` on both sides, so the conversion produced a 8.42 GB text tower
and a 672 MB mmproj; **both `FROM` lines are required** or the model cannot see.

**The export lost nothing.** Same 200 held-out samples, local GGUF against the Colab LoRA:
**200/200 byte-identical output**, and 197 distinct timings so it is really generating. Scoring
reproduces P4c exactly — every metric 1.000, exact match 200/200, `solvable_but_wrong` **0**.
And it is **6.5x faster** (34.5 s -> 5.3 s per image), which is the batch-1-with-unmerged-LoRA
cost from P4c section 5.3 being paid back.

**The part that was not expected.** Six real LinkedIn screenshots, `finetune` prompt (no few-shot,
so no `puzzle_01..03` leakage): cell accuracy **1.000**, waypoint recall **1.000**, wall F1
**0.972**, end-to-end **5/6** — against 0.947 / 0.917 / **0.438** / 2/6 untuned. The single failure
is `puzzle_03`, where recall is 1.00 and one **extra** wall was hallucinated, over-constraining the
board into unsolvability — the *visible* failure mode, caught by the `solvable` flag. **Silent
failures: 0.** And both **7x7** screenshots came out perfect, including the 21-waypoint one, which
**contradicts** the handover's expectation that a model trained on 100% 6x6 would answer 7x7 as
6x6. n=6 is far too small to claim "verified on real screenshots", and these six have been the
development set since P0 — but the risk that synthetic training would collapse on real images did
not materialise.

**Wiring.** `POST /api/vision/solve` (multipart; `warnings`, the `solvable` confidence flag, the
solver path and the rendered answer) and a `Solve from Screenshot` Gradio tab that hands the read
board back as Python literals so a misread can be corrected in the other tabs. 503 / 422 / 415 are
deliberately distinct: unreachable model, unusable answer, unsupported file type mean different
things to a caller.

**A bug the unit tests could not catch.** `pydantic-ai`'s `run_sync` cannot be called from inside a
running event loop, and the endpoint was written `async def`, so the first live request returned
`503 ... RuntimeError: This event loop is already running` while all 11 endpoint tests were green —
the stub backend never touched the loop. Fixed by making the handler a sync `def` (FastAPI then
runs the whole blocking chain in a threadpool, which it should have been anyway), and the stub now
**asserts no loop is running**, so the constraint is held by a test. Verified the guard bites:
reverting the handler to `async def` fails 8 of 12.

**Every request is now kept, in a shape chosen on purpose.** `request_log.py` writes the
uploaded image and the model's own words into `logs/vision/` (git-ignored) as
`images/` + `metadata.jsonl` — the same field names `dataset_builder` uses. What
`score_predictions` needs is `label` and `raw_output`, carrying `file_name` and
`generation_seconds`; all of those are written except `label`, which nobody can know for
a picture a user just uploaded. **So hand-writing one `label` turns a line of real usage
into a scoreable evaluation sample** — the cheapest route to the real-screenshot set that
section 6.3 of the report names as the only thing standing between "it did well on six"
and "verified". Demonstrated end to end: two logged lines plus hand-written labels scored
`EXACT MATCH 1/2`, with the 0.889 wall F1 landing on `puzzle_03`, as it should.
The **422** case is logged too (`usable: false` plus `parse_error`), which is why
`ModelOutputError` now carries the text that failed — an unusable answer is the case no
evaluation set holds and no retry fixes, so locking it inside the exception threw away
the best evidence there is. Logging never raises, and the endpoint tests redirect it to a
temp folder so the suite cannot quietly fill the very folder meant to become a dataset.

Tests **167 -> 214 passed, 8 xfailed**; `ruff` clean. `.env` and `.env.example` now carry the
fine-tuned tag plus `VISION_PROMPT_VARIANT`, and `docker-compose.dev.yml` mounts `./models` into
the ollama container read-only and overrides `OLLAMA_PROVIDER_URL` for the app container so both
"run on the host" and "run in compose" stay correct without editing `.env`.

## 2026-08-22

### VLM Track: P4c ran, reading is done, and the benchmark it was measured on is now saturated

Full results and the raw numbers: [reports/2026-08-22_vl-p4c-results.md](reports/2026-08-22_vl-p4c-results.md).
Execution record with every cell's output: `notebooks/p4c_finetune_8000.ipynb`.

Baseline first: `uv sync` (resolved 131 / audited 116), `uv run pytest` **136 passed, 8 xfailed**,
`ruff` clean — matching the handover, so nothing had drifted.

**The result.** 7,800 samples, one epoch, 975 steps on a paid Colab L4. On 200 held-out
samples that never entered training, every metric came out at **1.000** — JSON 200/200,
grid size 200/200, cell accuracy 1.000, waypoint recall 1.000, wall F1 1.000 over the 189
walled boards, micro wall precision and recall 1.000, end-to-end **200/200**. The
per-wall-count breakdown is 1.00 in all thirteen buckets, including the 24 boards carrying
twelve walls. The untuned baseline this had to beat scored wall F1 **0.438** and 2/6
end-to-end on real screenshots.

**A perfect score is a bug report until proven otherwise**, so it was attacked before it was
believed:

| check | result |
|---|---|
| is `raw_output` secretly the label? | identical 200/200 — but that is *expected*, the training target is exactly that serialisation |
| is the timing real? | `generation_seconds` has **200 distinct values**, 17.4–50.8 s — measured per call, not a constant |
| is the split what the notebook claimed? | file names match the tail slice in order; **intersection with training: 0** |
| content-level leakage? | duplicate labels **0**, duplicate render recipes **0**, **duplicate image bytes 0** |
| recompute without the project's scorer | independent structural comparison **200/200 agree** |

It is real. Training measured 975 steps / **1.56 h** / **5.77 s/step** / peak VRAM **20.90 of
22.03 GiB (94.9%)** / ~2.4 CU, with the vision stack moving more than the language stack
(`max|B|` 0.272 against 0.166), consistent with the whole premise that the bottleneck was visual.

**And that saturates the benchmark.** Every metric at 1.000 means the ruler can no longer
measure anything: a vision-layer ablation, a CoD variant, less training data, a different
batch size would all score 1.000 on this set. Regaining discrimination needs *harder synthetic
data* — visual noise, several renderer styles, simulated screenshot compression and rescaling,
larger boards. **The operator decided the same day not to collect real screenshots** and to
stay on self-generated data, upgrading the earlier "deferred" to "not doing". The cost of that
is stated everywhere it matters: these numbers prove the model learned *our renderer*, not
that it can read a LinkedIn screenshot.

Note also that the win is not shipped yet: the adapter lives on Drive, and
`puzzle_parser.parse_puzzle_image()` still calls the untuned Ollama model. P4d (export) is
what closes that gap, and is the next step.

**Built this session:**

-   **`src/core/vl_models/score_predictions.py`** — scores a held-out `predictions.jsonl`
    offline, through `puzzle_parser.parse_model_output` (so hallucinated walls are dropped
    exactly as the endpoint will) and `benchmark.score_layout` / `score_walls` (so the layers
    match the published baseline). No metric is reimplemented anywhere, which is the same
    discipline that fixed the transport drift. Beyond the existing layers it adds micro wall
    precision/recall, a breakdown by wall count, and — after the operator articulated the
    pipeline intent — `path_is_legal()` plus **`solution_valid_on_truth`**: solve the board the
    model *read*, then check that answer against the board that was really there. That is the
    product's actual criterion and it is deliberately weaker than `exact_match`, because
    misreading a wall the route never touches costs the user nothing. On fabricated test data
    the two differ by nearly a factor of two (0.250 vs 0.475), so reporting only `exact_match`
    would badly understate the pipeline.
-   **`notebooks/p4c_finetune_8000.ipynb`** — the real run: a lazy `datasets.Dataset` +
    `set_transform` instead of P4a's materialised PIL list, a **learning-rate-0** five-step dry
    run that exercises the whole path while provably not moving the weights, Drive checkpoints
    with a `RESUME` switch, and no metrics computed in the notebook at all.
-   Tests 136 → **167 passed, 8 xfailed**; `ruff` clean.

**The pipeline asymmetry worth remembering.** Because an exact solver consumes the parse, the
two wall errors fail differently. Predicting *extra* walls can only over-constrain, so any
solution found is still legal on the true board — the risk is a visible "unsolvable". *Missing*
a wall lets the solver walk straight through one and nothing complains. So recall is the
safety-critical side, and the solver doubles as a free verifier: the generator builds every
board from a Hamiltonian path, so an unsolvable prediction is *known* to be a misread without
any ground truth. `solvable_but_wrong` counts the silent failures; P6 should surface this.

**Four defects in code written this session, all found the expensive way:**

1.  **The held-out set nearly leaked.** The obvious candidate was the existing 120-sample
    `smoke_6x6` pack. Measured against the training pack: **render recipes identical 120/120,
    labels identical 82/120**. `draw_recipe` seeds each item with `random.Random(seed + index)`,
    and the two packs were built one seed apart, so one is the other shifted by an index. Only
    the wall-clock non-determinism in `generate_puzzle` kept the other 38 apart. Fixed by
    carving the held-out set from the tail of the same archive, which also costs nothing to
    upload. (P4a is unaffected: it trained on `smoke[4:]` and evaluated `smoke[:4]`.)
2.  **The dry run's timing projection was useless.** It measured **37.59 s/step** and printed
    "projected 10.18 h" against an actual 5.77 s/step and 1.56 h — a 6.5× error, because the
    first of five steps absorbs all the compilation and autotuning. Its other jobs (plumbing,
    memory, proving the weights do not move) were sound; only the extrapolation must be
    discarded-first-step or labelled an upper bound.
3.  **★ Evaluating with batch-1 sequential generation cost more than the training run.**
    Inference took 1.92 h / ~3.0 CU against training's 1.56 h / ~2.4 CU. At batch 1 each
    generated token re-reads all 9.16 GB of weights, so the L4's 300 GB/s sets a 30.5 ms/token
    floor — about 9.2 s per sample — yet it took 34.5 s, i.e. **27% of roofline**. The rest is
    per-token fixed cost: 344 unmerged LoRA adapters adding 688 kernel launches per token, and
    the Python overhead of an uncompiled `generate` loop. The GPU was mostly idle. Batch the
    generation and merge the adapter next time.
4.  **"Line-by-line flush to Drive survives a disconnect" was false.** Colab's Drive FUSE only
    uploads on *close*; `flush()` reaches the FUSE layer and no further. The file was invisible
    on drive.google.com for the entire two-hour run and a disconnect would have lost all of it.
    Write to `/content` and `shutil.copy` to Drive per batch instead.

A fifth, smaller correction: the P4a notebook's claim that training renders no `<think>` block
is wrong — the measured prompt ends `...assistant\n<think>\n\n</think>\n\n`. That is the second
time the prose description of the training rendering has been wrong while
`build_inference_prompt` produced the right prefix anyway, because it derives from the training
path rather than from anyone's description of it.

**Hardware, measured rather than quoted**, since it decides where future runs go. The same
matmul benchmark on both: local RTX 4070 Ti SUPER **90.06 TFLOP/s** bf16 and **588 GB/s**
(measured copy) against the L4's **64.11 TFLOP/s** and 300 GB/s spec — the L4 is not faster, it
just has more VRAM, and at a 20.90 GiB peak the local 15.99 GiB card cannot hold this
configuration at batch 2. Training ran at roughly **42% MFU** with weight traffic accounting for
about 4% of the step, so it is compute- and launch-bound, not bandwidth-bound. Raising the batch
size is a trap here: images vary from 472 to 998 px, and a micro-batch pads to its longest
member, so batch 2 already wastes **18.9%** of its compute on padding and batches of 4 and 8
would waste 30.6% and 37.2%.

Cost for the session: ~2.4 CU training + ~3.0 CU inference + ~0.6 CU setup ≈ **6.0 CU** against
a 3.2 CU plan, with the entire overrun in defect 3.

### VLM Track: P5 skeleton, and the transport bug that made the shipped path 12x slower (branch `feat/vlm-parser`, worktree `zip-vlm`)

Session goal was "refactor the two parsers"; the refactor turned up a defect that would have
invalidated the whole P4 comparison, so that is the headline.

-   **The thinking switch was never wired into the transport the app ships.** `benchmark.py` passed
    `think` only to the native `/api/chat` path; `_request_pydantic_ai()` did not even take the
    argument, so `--no-think` was a silent no-op there. Measured on `qwen3.5:4b-q8_0`, `--prompt
    sized`, `puzzle_01` + `puzzle_05`: **`pydantic-ai` 66.5s/JSON 0/2 against `native` 4.1s/JSON
    2/2**. Every good number in the P1 report was measured on `native`, while the production parser
    was built on the other one.
-   **Root cause is not "the flag was forgotten" — the two surfaces take different knobs.** Probed
    `/v1/chat/completions` directly: a top-level `think: false` is **ignored** (9.7s, 1392 reasoning
    characters, identical output to the control), while `reasoning_effort: "none"` works (0.9s, 0
    reasoning characters). Confirmed the same through `pydantic-ai` with both
    `openai_reasoning_effort="none"` and `extra_body={"reasoning_effort": "none"}` (6.2s → 0.7s).
-   **New `src/core/vl_models/backends.py`** — the single source of truth for transports, holding
    that translation. `benchmark.py` and the new parser both build backends from it, so they cannot
    drift apart again. After the fix, the same two images on `openai-compat` give **5.5s / JSON 2/2 /
    cell accuracy 0.944 / shape 2/2**, matching `native` (6.4s / 2/2 / 0.944 / 2/2).
-   **New `src/core/vl_models/puzzle_parser.py`** — the supported `image → ParseResult` entry point.
    Reads the model tag and URL from `src.settings` instead of hard-coding `openbmb/minicpm-o2.6` and
    port 11434 (the project publishes Ollama on **11435**, so the scratchpad had been pointing at
    another project's daemon). Logs through `loguru`, raises `VisionBackendError` /
    `ModelOutputError` instead of returning `None`, and **drops hallucinated walls** (out of bounds,
    or between non-adjacent cells) into `ParseResult.warnings` — P1 showed false-positive walls are
    as fatal as missed ones, so they must be visible rather than silently accepted.
-   **New `src/core/vl_models/prompt_baseline.py`** — the frozen few-shot prompt lifted out of the
    scratchpad, verified byte-identical (SHA-256 `b8e75a8c…cfaa`) and now pinned by a hash test.
    `final_puzzle_parser.py` is marked SCRATCHPAD and re-exports it; nothing was deleted.
-   **New `src/core/tests/vl_models/`** — 39 tests, all mocked. Suite goes 76 → **115 passed, 8
    xfailed**; `ruff` clean.

### P4a: the smoke test passed, and caught a defect that would have invalidated P4c

Ran `notebooks/p4a_finetune_smoke.ipynb` on a paid Colab **L4** (capability 8.9, 22.03 GiB,
native bf16). 120 synthetic samples, 50 steps, adapted from the official Unsloth
`Qwen3_5_(4B)_Vision.ipynb`.

**Measured:**

| | |
|---|---|
| speed | **7.54 s/step** (50 steps in 377s), effective batch 8 |
| extrapolated | 1 epoch over 8,000 = 1,000 steps = **2.09 h**; 3 epochs = 6.28 h |
| peak VRAM | 16.568 / 22.034 GiB (75.2%) -- little room to raise batch size |
| loss | 1.011 -> 0.0019 over 50 steps |
| adapter | 168 MB, saved to Drive |
| cost | L4 bills **1.54 compute units/hour**; 1 epoch = 3.2 CU, i.e. 3.3% of a 98.99 CU balance |

Loss collapsing to ~0.002 is memorisation of 116 samples over 4 epochs, which is expected
here and is *not* a result. It is worth remembering as a tripwire for the real run: on
8,000 samples the loss should not do that.

**Unsloth confirmed the intended setup** with `QLoRA and full finetuning all not selected.
Switching to 16bit LoRA.` -- bf16, unquantised, which is exactly why this route needs an
L4 rather than the free T4.

### The vision tower really is being trained

`get_peft_model` emitted a warning that it could not register an input-embedding hook on
`model.base_model.model.model.visual` and was falling back to a pre-forward hook. Since the
entire premise of this track is that the failure is *visual*, a silently untrained vision
tower would have wasted the whole run. Checked directly rather than trusted:

```
trainable params : visual=6,291,456   language=32,464,896
visual   : 96/96  lora_B non-zero, max|B| = 1.137e-01
language : 248/248 lora_B non-zero, max|B| = 5.866e-02
```

`lora_B` initialises to zeros, so non-zero after training proves gradients flowed. The
warning is noise. Note the vision adapters moved *further* than the language ones, which is
consistent with the premise.

### The defect: training and inference rendered the prompt differently

P4a's post-training output was **correct content in the wrong shape** -- every wall it named
matched ground truth, but it arrived as prose inside a thinking block instead of as JSON.
Two mismatches, either of which alone is enough:

| | training | inference (as first written) |
|---|---|---|
| thinking | `<think>

</think>

` then the answer | `<think>
`, opened and never closed |
| content order | `[text, image]` | `[image, text]` |

`build_inference_prompt` fixes both *by construction*: it renders a conversation through the
same `apply_chat_template` call training uses and cuts at the answer, so the prefix matches
by definition. Measured on the same adapter, same 4 held-out images, changing nothing but
the prompt:

| prompt | JSON parsed | layout exact | wall F1 |
|---|---|---|---|
| **fixed** | **4/4** | **4/4** | 0.833, 1.000, 1.000, 1.000 (mean **0.958**) |
| broken (P4a control) | 0/3 before it crashed | - | - |

Three of the four had **every wall exactly right**, from an adapter trained on 116 samples
for 50 steps. That establishes the task is learnable from this renderer.

⚠ Not comparable to the 0.470 wall F1 quoted elsewhere: that was measured on the six *real*
screenshots, this on *synthetic held-out*. It says "it learned our renderer", not "it can
read a real screenshot".

**Correction to an earlier claim in this log's 2026-08-22 entry:** the training rendering was
described as having no `<think>` at all. It does -- an empty closed block. The mistake came
from inspecting only the tail of the rendered string, where the JSON ends. The diagnosis was
right and the fix works, but it works because it derives from the training path rather than
from my description of it; `enable_thinking=False` would in fact also have been correct.

### E1: what image resolution costs

7.54 s/step is slow for a 4B model on an L4, and the dataset renders at 472-975 px. Counted
tokens per sample instead of timing runs (instant, and it does not train the adapter further):

| longest side | tokens | vs base | est s/step |
|---|---|---|---|
| original (656) | 529 | 1.00x | 7.54s |
| 768 | 705 | 1.33x | 10.05s |
| 640 | 529 | 1.00x | 7.54s |
| 512 | 385 | 0.73x | 5.49s |
| 448 | 325 | 0.61x | 4.63s |
| 384 | 273 | 0.52x | 3.89s |

Two things worth keeping: **640 buys nothing** (the patch grid rounds to the same count as
656, so small downscales are wasted effort), and **768 costs 33% more** -- never upscale for
the sake of a uniform size.

**Not applying it to P4c.** Halving the step time saves an hour on a run that costs 3.2 CU
out of 99, while real screenshots are ~920x1018 and training smaller would reintroduce a
train/inference mismatch of exactly the kind that just cost a run. The number is banked as a
lever for later, once there is a real validation set to check that walls survive the
downscale.

### Colab access settled: the VS Code kernel is enough, and the runtime is an L4

Ran the smoke test through `mcp__ide__executeCode` against the notebook's Colab kernel, i.e.
the assistant executing directly on the remote runtime from a VS Code session.

```
python   3.12.13          cwd /content          colab True
NVIDIA L4, 23034 MiB, driver 580.82.07, compute_cap 8.9
torch 2.11.0+cu128    capability (8, 9)    VRAM 22.03 GiB
bf16 NATIVE  True
```

-   **The WSL2 + `google-colab-cli` route is unnecessary.** The official Colab CLI is
    Linux/macOS only (`fcntl`, issue #12), which made WSL2 look mandatory on Windows. It is
    not: the VS Code Colab extension exposes the runtime as an ordinary Jupyter kernel, and
    the IDE tool drives it. One human click to connect, then no further local setup.
-   **L4 gives 22.03 GiB, more headroom than the local 16 GiB card.** Qwen3.5-4B bf16 LoRA
    (~10GB per the survey) fits comfortably. Whether 9B fits is *borderline* -- the survey
    puts it at 22GB against 22.03 GiB available, which leaves nothing for activations, so it
    should not be planned for without measuring.
-   **`torch.cuda.is_bf16_supported()` is a trap and the smoke test had fallen into it.**
    Its signature is `(including_emulation: bool = True)`, so on the free **T4** it returned
    `True` despite Turing having no bf16 hardware at all; asking for
    `including_emulation=False` returned `False`. Reporting the default would have supported
    the conclusion "the free tier can train Qwen3.5-4B in bf16" -- it runs, unaccelerated,
    and would have been a slow and expensive way to discover the mistake. The notebook now
    prints both and states which model the answer implies.
-   **bf16 is fully accelerated on L4** (4096x4096 matmul, 30 iterations):

    | dtype | time | throughput |
    |---|---|---|
    | fp32 | 11.05 ms | 12.43 TFLOP/s |
    | fp16 | 2.26 ms | 60.74 TFLOP/s |
    | bf16 | **2.14 ms** | **64.11 TFLOP/s** |

    5.2x over fp32 and marginally ahead of fp16, which is what native tensor-core bf16 looks
    like. **So the model choice resolves to Qwen3.5-4B + bf16 LoRA**, the preferred branch.

### Two workflow traps worth knowing

-   **Selecting L4 does not move a running session.** The first connection had already
    created a T4 VM; changing the runtime type only affects the *next* one. It takes
    `Runtime -> Disconnect and delete runtime`, then a fresh connect.
-   **vscode-jupyter #17094**: after a server is removed and another created, the notebook
    controller stays disposed and cell execution fails silently (`Cannot call start again`).
    Closed, fixed in #17097/#17362. Workaround if it appears: reload the notebook file.
-   `mcp__ide__executeCode` needs the notebook to be the **active editor**; typing in the
    assistant panel steals focus, so it fails with `No active notebook editor found` more
    often than not. Running the cells by hand and reading the saved outputs back out of the
    `.ipynb` is the reliable path.

⚠ **Pay-as-you-go: an idle runtime burns compute units.** Disconnect and delete when not
training. Google's own Colab agent skill leads with this rule.

### P2 first half: a renderer that teaches the right cue, and a label that is the answer

Scope narrowed to **6x6 only** at the user's request; the renderer and builder both take a
size list, so widening it later costs a flag, not a rewrite.

-   **New `src/core/vl_models/render_puzzle.py`.** Rebuilt against the real screenshots
    rather than the old white-grid style. The decisive change is contrast: the 2025-10
    renderer drew cell borders with `outline="black"` *and* walls in black, so a wall was
    "a slightly thicker black line among black lines" -- while the real UI draws a light
    grey grid with heavy black bars. Training on the old images teaches a discriminating
    cue that does not exist at inference, which is a plausible structural reason wall F1
    sits at 0.31. Also switched waypoints to a filled disc with the number knocked out in
    white, rounded the board, and added optional Undo/Hint chrome and a cursor artefact.
-   **Fonts are now portable.** `ImageFont.truetype("arial.ttf")` with a silent
    `load_default()` fallback meant the same code rendered differently on Linux.
    `ImageFont.load_default(size=N)` returns a scalable **Aileron** face bundled inside
    Pillow (11.3.0 here), so there is no system-font dependency, no vendored file and no
    licence question.
-   **New `src/core/vl_models/schema.py`** -- the root fix for the label mismatch. One
    Pydantic model now defines both the training target and what the parser validates, and
    `to_prompt_json` reproduces the few-shot layout exactly: 22 lines for a 6x6 board,
    rows on one line each. Plain `json.dumps(indent=2)` was the first attempt and it
    exploded a board to ~40 lines, which is both a shape the prompt never demonstrates and
    roughly ten times the tokens on every example.
-   **New `src/core/vl_models/dataset_builder.py`.** Walls are sampled by this module, not
    by the generator: `puzzle_generator.py:123` hard-codes 2-5 with no override, and the
    real 6x6 screenshots carry 0, 4, 4 and 10. It asks for a wall-free puzzle, recomputes
    the safe edges from the returned solution path and samples 0-12 itself, so
    `src/core/puzzle_generation/` stays untouched. Augmentation covers light/dark theme,
    five cell sizes, optional buttons and cursor, +-2 degree rotation and JPEG 60-95.

### Two things that were tried and rejected, with measurements

-   **`multiprocessing.Pool` hangs on Windows here.** 32 items produced no output in 10
    minutes, with or without the CP-SAT step, because `spawn` re-imports this module in
    every child. Removed rather than shipped behind a flag. Sequential throughput measured
    at 200 samples in 72s, so 8,000 is roughly 45-50 minutes -- a one-off cost.
-   **Bit-identical regeneration is not achievable without touching the shared generator,
    so the claim was dropped.** `generate_puzzle` aborts its randomized backtracking on
    *wall-clock* time, and a clipped attempt consumes a different amount of the global
    `random` stream than a completed one. Same seed, two runs: **8 of 30** samples differed
    at a 0.5s budget. Raising the budget to 5s only moved it to **3 of 30** and made the
    build 3x slower, because 6x6 search times are heavy-tailed -- measured over 30 seeded
    searches: median 0.07s, 8 of 30 above 0.5s, maximum 5.4s. Fixing it properly means
    bounding the search on *work* rather than time, which lives in the module this track
    may not modify.
    Instead the **artifact** is now the unit of reproducibility: the manifest carries
    SHA-256 digests of `metadata.jsonl` and of the image bytes, and
    `--check <dir>` re-verifies them. That is what confirms the copy on Colab is the copy
    inspected locally -- which is the property actually needed.

Suite: 115 -> **136 passed, 8 xfailed**; `ruff` clean.

### Open question this session raised: the Q4 verdict may be confounded

P1 concluded "qwen needs Q8" because `qwen3.5:4b` at Q4 never emitted JSON. That measurement was
taken **with thinking on** — the failure mode described was 16,505 reasoning characters and 6,215
output tokens hitting the ceiling, which is the *thinking* budget running out, not the quantisation
losing information.

Re-measured 2026-08-22 with reasoning off (`--no-think`, `--prompt sized`, `openai-compat`,
`puzzle_01` + `puzzle_05`):

| model | JSON | cell acc | shape | wall F1 (walled) | mean latency |
|---|---|---|---|---|---|
| `qwen3.5:4b-q8_0` | 2/2 | 0.944 | 2/2 | 0.314 | 5.5s |
| `qwen3.5:4b` (Q4) | 2/2 | 0.944 | 2/2 | 0.305 | 6.1s |

**Indistinguishable.** Not enough to overturn the decision table — two images, one run, and the
report's own warning that `seed` plus `temperature=0` is not deterministic still applies — but enough
that the "Q8 is required" line should be re-measured over all six images with repeats before P4
budgets anything on it. The GPU peak from that Q4 run (12,142 MiB) is **not usable**: the Q8 model
was still resident, the same contamination the handover already flags.

Practical consequence either way: peak VRAM for the Q8 run is **7,992 MiB of 16,376**, so a 4B model
fits at F16 (~8GB of weights) on this card. Quantisation is a variable this project can simply
remove at deployment time rather than tune.

### Environment findings

-   **`/svelte-ui` 404 in any fresh worktree**: `frontend/dist/` is not in version control and
    `main.py:63` only prints a warning that gets buried. `npm install && npm run build` fixes it.
    Added to the plan's "does not come with the worktree" table.
-   **Chasing that 404 cost more than it should have.** Repeated `python -m src.app.main` spawns left
    several LISTENING entries on port 7440, and Windows kept routing requests to the *oldest* live
    process — the one started before `dist/` existed. `TestClient` returned 200 while `curl` returned
    404. Verify on a clean port (`APP_PORT=7441`) when a change appears not to take effect.
-   **`.env.example` was three variables behind `.env`** — no `OLLAMA_HOST_PORT`,
    `OLLAMA_PROVIDER_URL` or `OLLAMA_MODEL_NAME`, although `src/settings.py` reads the last two.
-   `feat/vlm-parser` fast-forwarded to `main` (it was a strict ancestor, so nothing was rewritten).

### Inventory: what already exists for P2

-   **A renderer already exists.** `src/core/puzzle_generation/generate_cod_dataset.py` generates
    puzzles, renders PNGs and writes Chain-of-Draft labels in parallel — this is what produced
    `zip_puzzles/cod_dataset_*`. Its wall drawing (a `width=5` black line on the grid line) is
    already the right idea; its white-grid/black-outline style, its lack of a seed, its hard-coded
    2–5 wall count and its `arial.ttf` dependency are not. P2 should modify it, not start over.
-   **The 8,000 synthetic images do not exist.** Local totals: `zip_puzzles/` 10 PNGs,
    `illustrations/` 14, everything else zero. `datasets/` is empty.
-   **A fine-tuning run really happened in 2025-10**, and its artifacts survive **only on Google
    Drive** (`colab_finetune/`: `cod_dataset_20251024_170006.zip` 14.3MB, two
    `finetune_dataset_20251024_*.zip`, plus `all_trained_runs/` and `trained_models/`). A full-disk
    search found no local copy — only a stale `generate_finetune_dataset.cpython-311.pyc`; the source
    was deleted in `422643d` (2025-10-28). The 2026-08-15 baseline report never evaluated any
    fine-tuned model, so what is in `trained_models/` is unknown.

## 2026-08-15

### VLM Track P0 + P1: Deployment Smoke Test and Untuned Baseline (branch `feat/vlm-parser`, worktree `zip-vlm`)

Executed stages P0 and P1 of [`plans/2026-08-15_track-vlm-parser.md`](plans/2026-08-15_track-vlm-parser.md).
Full numbers, method and caveats: **[`reports/2026-08-15_vl-p0-p1-baseline.html`](reports/2026-08-15_vl-p0-p1-baseline.html)**.

-   **New: `src/core/vl_models/benchmark.py`** — the measurement harness. Imports the *existing*
    `final_puzzle_parser.build_puzzle_prompt()` rather than copying it, so the baseline cannot drift
    from the prompt it claims to measure. Scores four layers (JSON parse rate; per-cell accuracy,
    waypoint recall and wall P/R/F1 against `src/core/tests/conftest.py`; end-to-end via CP-SAT;
    latency plus `nvidia-smi` peak). Two transports behind `--client`: Ollama's native `/api/chat`
    (full timing counters) and `pydantic-ai` (the path the shipped parser will use). Every call is
    persisted to `ai-collab/reports/artifacts/` with its seed and raw output.
-   **Environment**: `docker compose pull ollama` took the container from **0.16.1 → 0.32.13**; the
    pinned volume kept the 15GB of 2025-10 models. Pulled `gemma4:e4b`, `gemma4:e4b-it-q8_0`,
    `qwen3.5:4b`, `qwen3.5:4b-q8_0` from the official library. All four load **100% on GPU** — a 16GB
    card is not the bottleneck for 4B-class Q8 (worst case 9582 MiB peak).
-   **Quantisation matters far more than expected, and in opposite directions per family.**
    `qwen3.5:4b` at Q4 never emits JSON at all — it thinks for 16,505 characters and burns 6,215
    output tokens hitting the ceiling. The same model at Q8 reads the grid perfectly in 6.2s.
    `gemma4:e4b` is the reverse: Q8 misreads a 6×6 grid as 7×6 while Q4 gets it right.
-   **Disabling thinking is a free, large win — for one family only.** With `think: false`,
    `qwen3.5:4b-q8_0` goes from 3/6 to **6/6** parseable, gets **6/6 grid sizes right** (including the
    two 7×7 puzzles it previously could not answer at all), and runs **5.8× faster** (44.5s → 7.7s).
    The same switch makes `gemma4:e4b` *worse* on every structural metric. Measure it per model;
    never carry the setting over.
-   **The remaining problem is almost purely walls.** Best untuned configuration
    (`qwen3.5:4b-q8_0` + no thinking) scores cell accuracy 0.924 and waypoint recall 0.910 across the
    six screenshots, but wall F1 only 0.410 — and end-to-end is **1/6**. `gemma4:e4b` is 0/6.
    Wall *false positives* are as fatal as misses: on `puzzle_03` gemma4 found all 4 real walls yet
    the puzzle was unsolvable because it hallucinated 2 more.
-   **Two metric traps found and fixed.** Wall-free puzzles score a free F1 of 1.0, which inflated
    gemma4's wall mean from 0.268 to 0.512 — added `mean_wall_f1_walled_only`. And `seed` plus
    `temperature=0` does **not** guarantee determinism: `gemma4:e4b` Q4 produced different answers on
    cold vs warm runs, while the Q8 models were stable. Comparisons need repeats.
-   **Two zero-training interventions, both measured.** Beyond disabling thinking, a `sized` prompt
    variant adds an explicit grid-counting step and a synthetic 7×7 example (generator seed 20260815,
    CP-SAT verified; deliberately *not* puzzle_04/06, which are evaluation data). For
    `qwen3.5:4b-q8_0` this lifts cell accuracy 0.924 → **0.961**, wall F1 0.410 → 0.438, end-to-end
    matches 1/6 → **2/6**, and latency 7.7s → **5.4s**. Read that 2/6 carefully: puzzle_01–03 have
    their answers inside the few-shot prompt, so the previously-correct puzzle_03 was leaked — the
    newly correct one is **puzzle_05, which is not in the prompt**, so the genuinely generalising
    count went 0 → 1.
-   **`gemma4:e4b` got worse under both interventions.** The sized prompt drops it 4/6 → 3/6 on grid
    size, 0.444 → 0.315 on cells and 0.268 → 0.023 on wall F1; told that grids are often not 6×6, it
    over-corrects and reads the genuinely-6×6 puzzle_01 as 7×7. Two independent interventions now
    point the same way: **Qwen absorbs instructions, Gemma is unstable under prompt perturbation.**
    Best untuned configuration is `qwen3.5:4b-q8_0` + no thinking + `--prompt sized`, and that is the
    bar fine-tuning has to clear.
-   **Fine-tuning order revised.** Both families ship an official Unsloth vision notebook at the size
    we need, so that criterion ties. Recommendation is now **Qwen3.5-4B first if paid Colab is
    acceptable** — it only has to learn walls, whereas Gemma must learn size, digits and walls — and
    **Gemma 4 E4B if the free tier is a hard constraint**, since Unsloth explicitly advises against
    QLoRA for Qwen3.5 ("no matter MoE or dense, due to higher than normal quantization differences")
    and a free T4 has no bf16. Also verified first-hand: Qwen3.6 is 27B minimum, Qwen3.7 has no open
    weights, Qwen3.8 is 27B/2.4T — **Qwen3.5 is the only generation with sizes that fit a 16GB card**,
    and Gemma 4 is symmetric (official vision fine-tuning covers E2B/E4B only). Full generation table,
    release dates and a re-verification recipe are in §9 of the report.
-   **Dependencies took three rounds to settle**; see the `build(zip)` commit for the full reasoning.
    `pydantic-ai` was pinned to `==1.107.5`, but the meta-package forces `huggingface-hub>=1.3.4`,
    which is incompatible with `transformers<5`, and `transformers` 5.x silently disables its PyTorch
    backend against the pinned `torch 2.4.1` — that would have left the planned transformers VL
    backend unable to load a model. Settled on **`pydantic-ai-slim[openai]==1.107.5`** (what the
    official Ollama docs recommend, and all this project uses), which dropped **104 packages** —
    anthropic, boto3, cohere, groq, mistralai, google-genai, xai-sdk, temporalio, logfire, mcp and
    the whole opentelemetry stack — and freed `transformers` to stay at 4.57.6.
    ⚠ Separately, **`uv add` could not resolve at all** in this project: the cu121 index is declared
    before PyPI and `index-strategy` lives under `[tool.uv.pip]`, which does not apply to
    `uv add`/`uv lock`/`uv sync`. Fixed structurally by marking that index `explicit` and routing the
    torch trio through `[tool.uv.sources]`; as a bonus the lock now pins them solely to the cu121
    build. **The RL track would have hit the same wall when changing torch.** Two casualties of the
    re-resolution were repaired: `ruff` (only present transitively, so it was pruned and broke the
    documented `uv run ruff check .` — now an explicit dev dependency) and `griffe` (pydantic-ai now
    depends on the renamed `griffelib`, and uninstalling old `griffe` took the shared module files
    with it). Verified after all of it: `pytest` 46 passed, `ruff check` clean,
    `transformers.utils.is_torch_available()` True, torch still `2.4.1+cu121`, and the Gradio UI
    rendered in Chrome — also compared against the main worktree's 5.49.1 to confirm the
    `gradio` 5 → 6 jump caused no regression.
### RL Track A1 — one-stroke env v2, dataset, and the baselines A2 must beat

Curriculum decision changed by the developer before A1 started: **one-stroke all the way,
with reverse curriculum instead of the "allow backtracking, tighten later" phases**
(rationale recorded in the track plan §4). Consequences: revisits are masked from step one,
so the v1 2-cycle is impossible by construction and the `visit_count` / `visit_recency`
channels were dropped; every training success is now a legal Zip solution.

-   **`src/core/rl/rl_env_v2.py`** — `PuzzleEnvV2`: Dict observation (8 channels padded to
    8×8 + 8 scalars), `action_masks()` covering bounds / blocked / walls / visited /
    out-of-order numbers, dead-end termination before an all-False mask can reach the
    sampler, sparse reward (+1 success, 0 otherwise) with optional potential-based coverage
    shaping. Legality mirrors `dfs.py:96-105`, and reset collects number 1 exactly like
    `dfs.py:72-77` — the detail v1 got wrong. 21 unit tests, all passing, including the
    ground-truth replay v1 failed 0/7.
-   **`src/core/rl/generate_dataset_v2.py`** — deterministic dataset builder that *keeps the
    solution path* (the old `generate_rl_dataset.py:59` discards it, which reverse curriculum
    cannot afford) and splits train/val/test per size. The old script is untouched.
-   **Generation cost fixed, 18x**: the generator's default `timeout_per_attempt=20s` is spent
    proving that wrong-parity start cells are impossible. Measured on 7×7: successful searches
    finish in ≤0.415s at a 0.5s cutoff and ≤1.606s at 2s. Dropping the cutoff to 0.5s took the
    5,100-puzzle build from a projected ~23 hours to **45 seconds**, and 100 7×7 puzzles from
    ~14 minutes to **35 seconds**. This is a call-site parameter; the shared generator was not
    modified.
-   **Baselines on 510 held-out puzzles × 20 episodes** (`logs/rl_baselines/`):

    | policy | 4×4 | 5×5 | 6×6 |
    |--------|-----|-----|-----|
    | masked random | 8.8% | 0.9% | 0.0% |
    | greedy (distance to next number) | 10.2% | 3.7% | 0.8% |

    Dead ends account for 90–100% of failures, confirming that under one-stroke rules the
    dominant failure mode is getting trapped, not running out of budget. Greedy is the ceiling
    of what distance-based shaping can teach, and it collapses by 6×6 despite the highest
    coverage (0.595) — **an experimental confirmation of restart-plan §2.2**, which until now
    was a static argument.
-   **Dependency settled**: `uv add sb3-contrib==2.7.1 --index-strategy unsafe-best-match`
    (the project's `index-strategy` lives under `[tool.uv.pip]`, which `uv add` ignores).
    Verified afterwards: `torch 2.4.1+cu121` and `stable-baselines3 2.7.0` unchanged,
    `MaskablePPO` imports, suite still green. sb3-contrib is the official SB3 contrib package
    (Antonin Raffin / DLR, MIT); 2.7.1 was released 2025-12-05.

Suite after A1: **76 passed, 8 xfailed**, `ruff check` clean. Next is A2 (Phase 1 training on
4×4 with reverse curriculum), which is the first stage that actually trains anything.

### RL Track A0 — env v1 is not merely hard to learn, it is unsolvable (branch `feat/rl-masked-ppo`)

The A0 sanity stage of [plans/2026-08-15_track-rl-solver.md](plans/2026-08-15_track-rl-solver.md) ran in the
`zip-rl` worktree. Baseline first: `uv sync`, `uv run pytest` → **46 passed**, `ruff` clean, matching the
2026-08-08 record. Then six probes were run against **unmodified** `src/core/rl/rl_env.py`. Full write-up:
[reports/2026-08-15_a0-env-v1-findings.md](reports/2026-08-15_a0-env-v1-findings.md).

-   **Replaying a ground-truth solution never terminates — 0/7 puzzles.** `reset()` puts the agent on
    waypoint 1 but leaves `_next_waypoint_idx` at 0, and the collection check only runs *after* a move
    (`rl_env.py:143-146`, `:199-208`). A legal one-stroke path never re-enters the start cell, so the
    waypoint index is pinned at 0 and `terminated` is unreachable. Every fixture path covered all cells
    (36/36, 49/49) and still scored about −35 to −48.
-   **The success bonus is reserved for illegal paths.** Prefixing the same solution with a single
    step off and back onto the start cell terminates **6/6** fixtures with **+999.01** (episode totals
    +2359 to +4946). `all_cells_visited` uses `len(set(path_taken))`, so revisits are not penalised at
    the terminal check. env v1's reward is therefore anti-correlated with the rules of Zip: the best
    scoring strategy it can teach is a cheat. This amends §2.4 of the restart plan — the probability of
    a *legal* positive sample was not "close to zero", it was exactly zero.
-   **The 2-cycle hypothesis is confirmed, so the v2 design stands.** Oscillating between two visited
    cells yields exactly 2 distinct observation hashes over 8 steps, and a deterministic 2-state policy
    built from them ran 69 steps to truncation without ever escaping, touching only those 2 cells.
-   **Illegal moves do not consume the step budget**: 82 boundary bumps against a budget of 72 produced
    1 distinct observation and never reported truncation (`:176-180` hard-codes `truncated=False`).
-   **Side finding for A1**: `generate_puzzle` fails on odd open grids by parity — a 5×5 start-cell sweep
    gave 13/13 success on `(r+c)` even and 0/12 on odd, so ~2.5% of seeds exhaust all retries and return
    `None`. Dataset generation must retry with a new seed. The generator itself was left untouched
    (shared module, read-only per the track plan).

Added: `src/core/rl/diagnose_env_v1.py` (six probes, JSON evidence to the git-ignored
`logs/rl_diagnostics/`), `src/core/tests/rl/test_rl_env_v1_diagnosis.py`, and `src/core/rl/action_space.py`
(shared path→action encoding). The unsolvable replays are pinned with `xfail(strict=True)` so the suite
stays green while failing loudly if v1 is ever changed. After the additions:
**55 passed, 8 xfailed**, `ruff check` clean. `rl_env.py` and the old checkpoints were not touched.

### Planning Reports for the VLM and RL Tracks (research only, no code changed)

Two design reports were written to unblock roadmap items #2 (VL integration) and #3 (RL restart).
No source code was modified in this session; the work was code reading plus external verification.

-   **`ai-collab/reports/2026-08-15_vlm-model-survey.html`** — model selection and fine-tuning plan
    for `image -> puzzle JSON`:
    -   Recommends **Qwen3.5-4B** (natively multimodal, Apache-2.0) with Unsloth **bf16 LoRA**
        (Unsloth explicitly advises against QLoRA for Qwen3.5); `Qwen3-VL-4B/8B` as the fallback.
    -   Training data comes from the existing puzzle generator: labels are free and exact, but a new
        LinkedIn-style renderer is required — `save_solution_as_image()` draws a *solution* in a
        different visual style than the real screenshots (black circles, thick wall bars, UI chrome).
    -   Two deployment landmines were found and documented: unsloth#3899 (garbled GGUF after vision
        fine-tuning) and ollama#14730 (imported GGUF + mmproj fails on some architectures). Hence the
        plan starts with a **deployment smoke test before any training**.
    -   Colab: the official Colab CLI (2026-06-05) is **Linux/macOS only**, so on Windows either use
        WSL2 or the VS Code Colab kernel extension.
    -   Also evaluates the "one-shot" variant the developer asked about, splitting it into an
        *agent-orchestrated* one-shot (parse → existing solver, cheap and reliable) and a
        *model end-to-end* one-shot (research-grade, hands off to the RL report).
-   **`ai-collab/reports/2026-08-15_rl-restart-plan.html`** — the RL restart plan required by
    roadmap item #3, covering two routes:
    -   **Route A (recommended first)**: a dedicated agent with **action masking**. Reading
        `rl_env.py` produced a sharper root cause for the 2025-10 failure than the original
        diagnosis: because `ch_path` is binary and the step counter is absent from the observation,
        an agent oscillating between two already-visited cells produces an observation sequence
        `o_A, o_B, o_A, ...` — a genuine 2-cycle, so a deterministic policy is *provably* trapped.
        Illegal moves are an even more degenerate single-state loop. Masking removes both by
        construction. Other findings: the potential used for shaping (distance to the next waypoint)
        is not isomorphic to the real objective (full coverage), and the observation exposes only the
        *next* waypoint, making long-horizon planning impossible in principle.
    -   **Route B**: GRPO/GSPO post-training of a language model, using the existing solver and
        `calculate_fitness_score()` as a verifier (RLVR). Recommends text-only input and 4x4 grids
        first, so vision and reasoning are not debugged simultaneously.

### Both reports revised to v2 after review

The developer reviewed both reports and pushed back on five points; both were rewritten the same day.

-   **VLM report v2**:
    -   The v1 survey was **out of date** and is now corrected: **Gemma 4** shipped 2026-03/04 with
        five image-capable sizes (E2B/E4B/12B/26B-A4B/31B) under a plain **Apache-2.0** license, while
        **Qwen 3.7/3.8 went closed (API-only)** — the newest *small open-weight* Qwen VL is still the
        Qwen3.5 series, and Qwen3.6 has nothing under 10B.
    -   New section on **what this machine can actually run**: 7–9B inference is *not* a problem
        (~6GB at Q4), the ceiling is training — Qwen3.5-9B bf16 LoRA needs 22GB and does not fit 16GB.
    -   Recommendation changed from a single model to a **split by Colab tier**: the free T4 is Turing
        and has no bf16, and Unsloth advises against QLoRA for Qwen3.5, so the free path is
        **Gemma 4 E4B QLoRA (10GB)** while the paid L4/A100 path is **Qwen3.5-4B bf16 LoRA**. Plan now
        trains both families on the same data and compares.
    -   New section on **OCR-specialist models** (PaddleOCR-VL 0.9B, DeepSeek-OCR 2, dots.ocr): the
        task looks like OCR but the bottleneck — "this bar separates cell (2,3) from (3,3)" — is a
        relation-extraction problem outside their pretraining. Verdict: cheap enough to run as a
        parallel B-arm, not the main line.
    -   New section on the **Unsloth notebook catalogue** (250+ notebooks; 30+ vision, 40+ GRPO/RL,
        OCR incl. DeepSeek-OCR and Paddle OCR). Notably a **Gemma 4 E2B Sudoku GRPO notebook** exists,
        which is the closest available template for the RL route B reward design.
-   **RL report v2** — route A's experiment was **redesigned from scratch** at the developer's request:
    -   Adopts the developer's two proposals: **allow backtracking** (easier to train than forcing a
        one-stroke path from the start) and **put visit counts in the observation**. The second one
        directly dissolves the 2-cycle diagnosed in v1; the report adds that a **strictly monotonic
        `steps_used / budget` scalar** is also needed, because a clipped visit counter can saturate.
    -   Observation is now 9 channels (valid mask, two wall planes, **visit count**, visit recency,
        agent, next/future/done waypoints) plus 6 global scalars; the `wp_future` plane fixes the v1
        finding that the agent could not see waypoints beyond the next one.
    -   Reward is now **FrozenLake-style**: +1 on success, 0 otherwise, with "finish faster" expressed
        by the discount factor rather than a per-step penalty (the old -1/step accumulated to -72 and
        drowned the +1000 terminal signal). Shaping potential switched from *distance to next waypoint*
        to **coverage ratio**, which is isomorphic to the real objective.
    -   Three-phase curriculum on constraint strictness: free backtracking → priced backtracking →
        hard-masked one-stroke. Explicitly notes that only the last phase produces a *legal* Zip
        solution, so "soft success rate" and "legal one-stroke rate" must be reported separately.
    -   **Dependency conflict found**: `MaskablePPO` lives in `sb3-contrib`, whose latest (2.9.0)
        requires `stable-baselines3>=2.9.0`, which in turn requires `torch>=2.8` — but this project
        pins `torch==2.4.1+cu121` (a deliberate decision recorded in `../AGENTS.md §5`). Three options
        documented; the choice needs the developer's call since packages are installed manually.
    -   Also adds a build-your-own assessment: ~620–860 lines using sb3-contrib, ~970–1310 lines fully
        hand-rolled (masking itself is ~40 lines in the CleanRL style).

### Ollama brought back as a Docker service (dev stack)

An earlier claim in this session — "Ollama is not installed" — was **wrong**: it had only been checked
as a native install. It runs in Docker here, and the 2025-10 assets were all still intact.

-   **Found**: image `ollama/ollama` present; volume `linkedin-zip-challenge_ollama_data` still holds
    ~15GB of blobs with three models (`openbmb/minicpm-o2.6`, `qwen2.5vl:7b`,
    `bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`). `docker-compose.yml.vl_version` already contained
    a working `ollama` service definition, but it was never merged into the active dev stack.
-   **Changed**:
    -   `docker-compose.dev.yml` — added the `ollama` service with the NVIDIA device reservation, a
        healthcheck, and a `volumes:` block pinning `ollama_data` to the pre-existing
        `linkedin-zip-challenge_ollama_data` so the 15GB is reused rather than re-downloaded.
        Container is named `zip_ollama_server` and the host port is `${OLLAMA_HOST_PORT:-11435}`:
        the name `ollama_server` and port 11434 are already taken on this machine by an unrelated
        project (verified via the container's compose labels).
    -   `.env` (not versioned) — re-enabled `OLLAMA_MODEL_NAME` / `OLLAMA_PROVIDER_URL` (both had been
        commented out). In-network URL is `http://ollama:11434/v1`; the host-side alternative is noted
        in a comment.
    -   `run_docker_dev.py` — added a non-fatal `check_ollama_ready()` that polls `/api/tags` and
        prints which models are available, plus the Ollama endpoint in the final summary.
-   **Verified** (2026-08-15): `docker compose -f docker-compose.dev.yml config` OK;
    `docker compose up -d ollama` starts; inside the container `nvidia-smi` reports
    `NVIDIA GeForce RTX 4070 Ti SUPER, 16376 MiB` (GPU passthrough works) and `ollama --version` is
    0.16.1; `ollama list` shows all three old models; host `GET :11435/api/tags` returns 200;
    `uv run ruff check run_docker_dev.py` passes.
-   **Not verified**: the app container reaching `http://ollama:11434/v1` (would require building the
    app image); and whether ollama 0.16.1 — a ~6-month-old cached image — can serve Qwen3.5 / Gemma 4.
    `docker compose pull ollama` is required before P0.

### Task plans written for two parallel worktree tracks

The two reports explain *why*; these new plans say *what to do*, and are written for a fresh agent
landing in an empty worktree.

-   **New directory `ai-collab/plans/`** (documented in both `AGENTS.md` files):
    -   `2026-08-15_track-vlm-parser.md` — P0 deployment smoke test → P1 baseline → P2 data pipeline
        → P3 real eval set → P4 SFT → P5 integration → P6 one-shot endpoint.
    -   `2026-08-15_track-rl-solver.md` — A0 env sanity → A1 env v2 (9 channels + 6 scalars, masking,
        FrozenLake-style reward) → A2/A3/A4 three-phase curriculum → A5 ship as the 10th solver.
-   **Worktree gotchas documented up front**, because a fresh worktree only gets version-controlled
    files: `.env` must be copied by hand (its absence broke startup back in 2025-10), each worktree
    needs its own `uv sync`, `datasets/rl_datasets/` is empty even in the main tree, and `models/`
    holds only the failed 2025-10 DQN checkpoints (do not resume from them). `illustrations/
    puzzle_01..06.png` *are* tracked, so the VLM track has its eval material from the start.
-   **Cross-track coordination rules**: code barely overlaps (`vl_models/` vs `rl/`), but
    `src/core/utils.py` and `src/core/puzzle_generation/` are read-only for both;
    `pyproject.toml`/`uv.lock` changes are serialised through the developer; `roadmap.md` and
    `dev_log.md` edits stay in each track's own section; and the Docker stack must not be started
    from two worktrees at once (container name and host port are machine-unique).

## 2026-08-08

### AI Collaboration Scaffold and Environment Recovery

The project had been dormant since 2025-10-30 (~9 months). This session rebuilt the collaboration
documentation so that any agent (or future self) can pick the project up without re-deriving context,
and verified that the toolchain still works.

-   **New documentation structure** (mirroring the conventions already used by `board-game-rl` and
    `deep-learning-karpathy` in this monorepo):
    -   `AGENTS.md` — the single source of truth for how to work on this project: startup routine,
        document ownership, five-step task workflow, environment/verification requirements, a
        task→file map, code conventions, and hard limits.
    -   `CLAUDE.md` — a one-line `@AGENTS.md` import, so opening Claude Code directly in this
        directory loads the same rules.
    -   `ai-collab/roadmap.md` — **the new first stop**: current status, prioritised next steps with
        explicit done-criteria, and a "settled decisions, do not reopen" table distilled from the
        660 lines of history in this file.
    -   `ai-collab/project_guide.md` — architecture (the three-layer Core/App/UI split), module
        responsibilities, the nine solvers, the API contract, and all three ways to run the app.
    -   `ai-collab/commands.txt` — reusable prompts and a command cheatsheet.
    -   `ai-collab/reports/` — directory for future task reports.

-   **File moves and link updates**:
    -   `dev_log.md` moved from the project root into `ai-collab/` via `git mv`, matching the other
        two sub-projects. Links in `README.md`, `README_zh-TW.md` and `gemini_readme_raw.md` updated.
    -   The duplicated `# Development Log` heading at the top of this file was removed.
    -   `gemini_readme_raw.md` marked as a superseded historical artefact, pointing to `AGENTS.md`.

-   **Environment determinism**:
    -   Added `.python-version` pinning **3.11**, so `uv sync` no longer has to guess a version that
        satisfies `requires-python = ">=3.11,<3.12"`.
    -   Documented explicitly (here and in the repo-level `AGENTS.md`) that this project is
        **deliberately kept out of the root `uv` workspace** — it pins `torch==2.4.1` with a cu121
        index against Python 3.11, which conflicts with the repo-root 3.9 devtools environment.

-   **Recovery verification** (the point of the exercise — the baseline is good):
    -   `uv sync` → resolved 190 packages, audited 172, no changes required.
    -   `uv run python -c "import sys; print(sys.version)"` → `3.11.13`.
    -   `uv run pytest` → **46 passed in 8.10s**.
    -   `uv run pre-commit run --all-files` → `ruff-format` and `ruff` both passed.

-   **End-to-end verification against a live server** (started with the documented
    `uv run python -m src.app.main`, not `TestClient`):
    -   `GET /` → 200; `GET /api/echo/health` → `{"status": "ok"}`; `POST /api/echo/` → `Echo: zip`; `GET /docs` → 200.
    -   `POST /api/solver/solve` for **DFS**, **A\* (heapq)** and **CP-SAT** against `puzzle_01`: all three
        returned a 36-step path **identical cell-by-cell to `solution_01` in `conftest.py`**, plus a
        ~74 KB animated GIF and a ~7.6 KB final PNG each (byte-identical across solvers, i.e. they
        converge on the same solution). The rendered PNG was inspected: waypoints, walls and the green
        step-order overlay all match the returned path (start `(1,1)` = step 1, `(0,0)` = step 5).
    -   Error paths: malformed layout → 400; unknown solver → 404.
    -   Chrome headless (`--dump-dom` + `--screenshot`) on `/ui`, `/svelte-ui` and `/docs`: the Gradio
        console renders all four tabs, the Svelte editor hydrates and draws its 320×320 canvas grid,
        and Swagger lists every endpoint.

### Small Defects Found During Verification (recorded, not fixed)

-   **Swagger shows the Echo endpoints twice.** `src/app/main.py` includes the router with
    `tags=["Echo"]` while `src/app/routers/echo.py` already declares `tags=["echo"]`; FastAPI merges
    both, producing two identical groups in `/docs`.
-   **The Svelte "Instructions" panel prints raw Markdown** — `**middle**` and `**border**` render as
    literal asterisks because that text is not passed through a Markdown renderer.
-   **The 3-of-9 solver gap is confirmed on both front-ends**, not just the API: the compiled Svelte
    bundle only contains the strings `DFS`, `A* (heapq)` and `CP-SAT`.

-   **Documentation fix**: the "Running Tests" snippet in `README.md` rendered as a broken two-line
    command (`.` followed by `un_tests.bat`) because the backslash-r was consumed; corrected to
    `.\run_tests.bat` and preceded by the `uv sync` / `uv run pytest` workflow.

### Known Gaps Recorded (not fixed in this session)

-   `src/app/routers/solver.py` only exposes **3 of the 9 implemented solvers** (`DFS`, `A* (heapq)`,
    `CP-SAT`). The six metaheuristic solvers are implemented and tested but unreachable from the API,
    the Gradio dropdown, or the Svelte dropdown. This is now the top item in `roadmap.md`.
-   `src/custom_components/puzzle_editor/frontend/Dockerfile` still exists, although the 2025-10-28
    entry below records it as removed during the Docker overhaul.

## 2025-10-30

### Documentation Refinement and Environment Verification

Conducted a comprehensive review and update of project documentation (`README.md`, `README_zh-TW.md`), alongside a thorough verification of local and Dockerized development environments. This phase focused on improving clarity, consistency, and ensuring the project's operational readiness.

-   **Documentation Enhancement**:
    -   Updated Svelte UI descriptions to accurately reflect its Canvas-based WYSIWYG editing capabilities.
    -   Clarified service access instructions, emphasizing unified access via `APP_PORT` and segregating developer-specific hot-reloading details.
    -   Added "Highlights" and "Technologies Used" sections to `README.md` for a comprehensive project overview.
    -   Integrated a note directing users to the `illustrations/` directory for visual aids and UI screenshots.

-   **Environment Operationalization & Debugging**:
    -   **Unified Settings Management**: Migrated `SVELTE_PORT` to `src/settings.py` for centralized configuration. Its reliance on the `.env` file for `docker-compose.dev.yml` was removed by hardcoding the value in the compose file.
    -   **`run_docker_dev.py` Debugging**:
        -   Resolved initial `SVELTE_PORT` not set errors (addressed by ensuring `.env` was correctly configured).
        -   Diagnosed and fixed FastAPI application startup failures within Docker containers.
        -   Identified that `Dockerfile.dev` initially lacked a `CMD`, causing containers to exit prematurely (addressed by adding `CMD ["tail", "-f", "/dev/null"]`).
        -   Discovered `docker compose exec -d` suppressed FastAPI startup logs (addressed by removing the `-d` flag).
        -   Pinpointed `pydantic.ValidationError` for `ollama_model_name` and `ollama_provider_url` (due to `.env` not being copied into the container).
        -   Corrected `Dockerfile.dev` to copy the `.env` file into the container, ensuring environment variables are properly loaded by `pydantic-settings`.
        -   (Note: Temporarily set default empty strings for `ollama_model_name` and `ollama_provider_url` in `src/settings.py` as a workaround for startup.)

-   **Environment Verification**:
    -   Confirmed the local environment setup instructions are accurate.
    -   Confirmed the Docker development environment (`run_docker_dev.py`) is operational after resolving startup issues.
    -   Confirmed the Docker production environment (`docker-compose.yml`) instructions are accurate.

## 2025-10-28

### Project Production-Ready Refactoring

Conducted a major refactoring initiative to improve code quality, streamline the user interface, and professionalize the deployment workflow. This effort touched upon configuration management, code duplication, error handling, and the entire Docker setup.

#### Phase 1: Code Quality and Consistency

-   **Settings Centralization**:
    -   Standardized all core application settings in `src/settings.py`.
    -   Centralized `app_port` and `app_host` to remove hardcoded values in the UI and utility scripts.
    -   Formalized `ollama_model_name` and `ollama_provider_url` to use Python's `snake_case` convention for internal consistency.
    -   Identified and removed the obsolete `svelte_port` setting after the frontend integration.

-   **DRY Principle Refactoring**:
    -   Identified significant code duplication in the setup phase of various solvers.
    -   Created a new `prepare_solver_input` utility function in `src/core/utils.py` to consolidate common logic for puzzle parameter extraction and validation.
    -   Refactored the `dfs.py` and `a_star.py` solvers to use the new utility function, significantly reducing their boilerplate code.
    -   Extended the refactoring to `generate_random_path` in `utils.py`, benefiting all metaheuristic solvers that depend on it.

-   **Error Handling and Logging**:
    -   Reviewed the API endpoint (`src/app/routers/solver.py`) and the Gradio UI (`src/ui/gradio_app.py`).
    -   Enhanced exception logging by replacing `logger.error(f"...")` with `logger.exception("...")` in the main solver API, ensuring full stack traces are captured for unexpected errors.
    -   Added error logging to the Gradio UI's API calling functions, which previously failed silently in the server logs.

#### Phase 2: UI Enhancements and Frontend Integration

-   **Svelte UI Integration**:
    -   Successfully integrated the standalone Svelte frontend into the main FastAPI application.
    -   Modified `vite.config.ts` to set the `base` path to `/svelte-ui/`, fixing asset loading issues.
    -   The FastAPI application in `src/app/main.py` now serves the built static files (`dist` directory) from the `/svelte-ui` path.

-   **New "Generate Puzzle" Feature**:
    -   Added a new "Generate Puzzle" tab to the Gradio UI.
    -   Implemented the UI with a dropdown to select the number of blocked cells (0, 1, or 2) and a button to trigger generation.
    -   The UI displays a preview image of the generated puzzle and provides the layout/walls in a copy-paste friendly format.
    -   Added a new unit test (`test_generate_puzzle_ui_success`) for this feature, using mocking to ensure its reliability.

#### Phase 3: Docker Workflow Overhaul

-   **Dual-Environment Strategy**:
    -   To balance development convenience with production-readiness, a dual-environment Docker setup was implemented.
    -   **Development (`docker-compose.dev.yml`)**: A new configuration was created to restore the two-container (backend + Svelte dev server) setup, enabling full hot-reloading for both frontend and backend development.
    -   **Production (`docker-compose.yml`)**: The main compose file was streamlined to define a single, self-contained service for production.

-   **Multi-Stage Production Dockerfile**:
    -   The main `.devcontainer/Dockerfile` was converted into a multi-stage build file.
    -   A `node:lts-alpine` stage is now used to build the production-optimized Svelte frontend (`npm run build`).
    -   The final Python stage copies the application code and the compiled frontend `dist` directory, creating a single, efficient, and immutable production image.

-   **Workflow Automation**:
    -   The `run_docker_dev.py` script was updated to default to using the new `docker-compose.dev.yml`, ensuring the best out-of-the-box experience for developers.
    -   Obsolete files (`frontend/Dockerfile`) and settings (`svelte_port`) were identified and removed to maintain project cleanliness.


## 2025-10-24 (Second Entry)

### Environment Deep Dive: Resolving Fine-Tuning Dependencies

With the decision made to proceed with fine-tuning, the next phase involved setting up the development environment to handle the complex dependencies required by Unsloth. This process revealed several layers of platform and package incompatibilities.

-   **Initial `xformers` Failure:** An attempt to install `unsloth` on the host Windows machine failed due to the `xformers` package lacking compatible wheels for Windows. This validated the necessity of using the project's Dockerized Linux environment for all fine-tuning tasks.

-   **Dependency Resolution in Docker:** Moving into the Docker container revealed a series of deeper dependency conflicts when trying to install `unsloth` into the project's existing environment:
    1.  A `numpy` version conflict arose due to `uv`'s default index strategy, which was resolved by using the `--index-strategy unsafe-best-match` flag.
    2.  A subsequent, more complex conflict was discovered between the project's pinned versions of `torch` and `transformers`, and the different versions required by `unsloth`.

-   **Root Cause Analysis: Build-time vs. Runtime Environment:** The final installation attempt failed while trying to build the `flash-attn` package. The error `OSError: CUDA_HOME environment variable is not set` and the warning `nvcc was not found` led to the root cause: the service's base Docker image (`python:3.11-slim`) was a **runtime** image, lacking the NVIDIA CUDA development toolkit required to **compile** custom CUDA extensions.

-   **Solution: Environment Isolation and `devel` Image:**
    1.  The `.devcontainer/Dockerfile` was modified to use `pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel` as its base image, which includes the full CUDA toolkit.
    2.  A new workflow was established: create a separate, isolated virtual environment (`unsloth_env`) inside the rebuilt Docker container to prevent any conflicts with the main project's dependencies.
    3.  A robust, multi-step `pip install` process was defined to first install `torch` from its specific index, followed by installing `unsloth` and its dependencies using the `--no-build-isolation` flag to ensure the build process could find the pre-installed `torch`.

-   **Next Step:** With a correctly configured and isolated environment, the next step is to execute the SFT training script (`train_puzzle_sft.py`) inside the new container setup.


## 2025-10-24

### Final VL Model Validation & Success of the Hybrid Strategy

Following the previous entry, the initial plan to pivot to Strategy A was revised to conduct a final, conclusive test of Strategy B (`pydantic-ai`).

-   **Final Capability Test of Model 1 (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`)**
    -   The test script was modified to request a structured Pydantic object (`AnimalInfo`) as the `output_type`.
    -   **Finding:** The model successfully returned a **structurally correct but empty** Pydantic object.
    -   **Conclusion:** This definitively proved that the `bsahane` model **supports tool-calling**, but its core **vision module is defective**, preventing it from providing any content.

-   **Capability Test of Model 2 (`openbmb/minicpm-o2.6`)**
    -   After replacing the model with `openbmb/minicpm-o2.6`, the same structured output test was performed.
    -   **Finding:** Received a definitive `400 Bad Request` error from the Ollama server with the message: `...does not support tools`.
    -   **Conclusion:** This proved that the `minicpm` model **does not support** the tool-calling API required by `pydantic-ai`.

-   **The Hybrid Strategy: Proposal and Success**
    -   Faced with a dilemma where one model had tool support but broken vision, and the other had working vision but no tool support, a new "hybrid strategy" was adopted. This approach continues to use `pydantic-ai` for its convenient API, but sets the `output_type` to `str` and leverages **Prompt Engineering** to instruct the model to generate a JSON-formatted string in its raw text response.
    -   The `experiment_minicpm_json_prompt.py` script was created to validate this strategy.
    -   **Result:** **Complete success.** The `minicpm` model correctly identified the image content (cat, bird) and returned a perfectly formatted JSON string, which was then successfully parsed in Python.

-   **Final Conclusion**
    -   A complete and viable technical pipeline has been established. The combination of a **vision-capable model (`minicpm`)** with the **`pydantic-ai` + Prompt Engineering** software pattern will serve as the foundation for the actual puzzle parser development.


## 2025-10-22 (Fourth Entry)

### VL Model Strategy Refinement & Tool-Calling Explained

Building on the experimental plan from the "Third Entry," this entry refines the VL model validation strategy and provides a deep dive into "Tool-Calling" to clarify why it is core to `pydantic_ai`'s structured output.

-   **Important Clarification: Experimental Phase**
    -   All current Vision-Language (VL) model integration work is in an **experimental phase**.
    -   All code within the `src/core/vl_models/` directory (including `vl_extractor.py`, `hf_parser.py`, and the new PoC scripts) should be considered a **"Scratchpad"**.
    -   The purpose of these scripts is to rapidly validate model capabilities and integration feasibility. They should not be considered final production code until the features are proven and standardized.

-   **Phase 1: Vision Sanity Check (Refined)**
    -   **Objective:** To validate the basic visual understanding of the new model (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`).
    -   **Test Assets:** The test images will be updated to `cat.jpg` and `bird.jpg`.
    -   **Methodology:** Continue using the `vision_sanity_check.py` script, asking the model a question (e.g., "Please describe the animal in the image and its primary color"), and expecting a reasonable natural language string response.

-   **Phase 2: Tool-Calling Proof of Concept (PoC)**
    -   **Objective:** To strictly verify if the model supports the "Tool-Calling" feature required by `pydantic_ai` for structured data output.
    -   **Methodology:** Use the `tool_calling_poc.py` script, which defines an `IdentifiedAnimal` Pydantic model and configures the `pydantic_ai` Agent with `output_type=IdentifiedAnimal`. This will directly test if the model can return a JSON object compliant with the Pydantic model, rather than just a `str`.

### Technical Deep Dive: "Tool-Calling" & `pydantic_ai`

-   **What is "Tool Support" (Tool-Calling)?**
    -   This is a key capability of an LLM (or VLM). **It does not mean the model "executes" code itself**.
    -   Instead, it means the model is trained to understand the "tool" definitions (i.e., a function's schema, including its name, parameters, and parameter types).
    -   When the model believes it needs to use a tool to answer a query (e.g., user asks "What's the weather in Miami?"), it **outputs a structured JSON request**, such as: `{"name": "get_weather", "arguments": {"city": "Miami"}}`.
    -   Our application (e.g., the Python script) receives this JSON and *then* the application *itself* executes the corresponding `get_weather("Miami")` function.
    -   This capability allows the model to interact with external APIs, databases, or local functions to retrieve real-time information or perform actions.

-   **How does `pydantic_ai` use Tool-Calling for `output_type`?**
    -   `pydantic_ai` cleverly abstracts this "Tool-Calling" mechanism.
    -   When we set `output_type=IdentifiedAnimal` in a `pydantic_ai` Agent:
        1.  `pydantic_ai` automatically reads the structure of the `IdentifiedAnimal` Pydantic model.
        2.  It converts this Pydantic structure into a "Tool" schema that the LLM can understand (something like: `{"name": "IdentifiedAnimal", "parameters": {"animal_name": "string", "color": "string", ...}}`).
        3.  `pydantic_ai` sends this schema, along with our prompt, to the VL model.
        4.  **If** the model supports tool-calling (like the `...-Instruct` version), it will recognize that we want it to "call" the `IdentifiedAnimal` tool and will generate a JSON string matching that schema.
        5.  **If** the model does not support it (like our previous `qwen2.5vl:7b`), it will ignore the schema and just return whatever natural language `str` it wants, causing `pydantic_ai` to fail parsing.
    -   This is precisely why the `bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh` model is critical; it claims to support this feature, which is the core hypothesis the Phase 2 PoC is designed to test.

### References

-   [1] IBM (2025). *What Is Tool Calling?*. Retrieved 2025-10-22, from: `https://www.ibm.com/think/topics/tool-calling`
-   [2] Analytics Vidhya (2025). *Guide to Tool Calling in LLMs*. Retrieved 2025-10-22, from: `https://www.analyticsvidhya.com/blog/2024/08/tool-calling-in-llms/`
-   [3] Medium (2025). *Understanding LLM Tool Calling*. Retrieved 2025-10-22, from: `https://medium.com/garantibbva-teknoloji/understanding-llm-tool-calling-traditional-vs-embedded-approaches-fc7e576d05de`
-   [4] Medium (2024). *Tool Calling for LLMs: A Detailed Tutorial*. Retrieved 2025-10-22, from: `https://medium.com/@yasir_siddique/tool-calling-for-llms-a-detailed-tutorial-a2b4d78633e2`
-   [5] PromptLayer Blog (2024). *Tool Calling with LLMs: How and when to use it?*. Retrieved 2025-10-22, from: `https://blog.promptlayer.com/tool-calling-with-llms-how-and-when-to-use-it/`
-   [6] LangChain Docs (2025). *Tool calling*. Retrieved 2025-10-22, from: `https://python.langchain.com/docs/concepts/tool_calling/`

### To-Do / Next Steps

1.  **[User]** Prepare `cat.jpg` and `bird.jpg` image files and place them in the `illustrations/` directory.
2.  **[Dev]** Ensure the `src/core/vl_models/vision_sanity_check.py` script is updated to use `cat.jpg` and `bird.jpg` for testing.
3.  **[User]** Execute the Phase 1 test: `python src/core/vl_models/vision_sanity_check.py` and report the results.
4.  **[User]** If Phase 1 is successful, execute the Phase 2 test: `python src/core/vl_models/tool_calling_poc.py` and report the results.
5.  **[Dev]** Based on the results of Phase 1 and Phase 2, jointly decide on the next implementation strategy for Puzzle extraction.

## 2025-10-22 (Third Entry)

### VL Model Experimental Plan

Finalized a two-phase experimental plan to validate the capabilities of the newly selected VL model (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`) before integrating it into the main puzzle-solving workflow. This approach defers the decision on the final implementation (manual JSON parsing vs. direct tool-calling) until the model's capabilities are confirmed.

-   **User-Provided Research:** The new model was selected based on user research indicating that it is an instruction-tuned vision model that explicitly supports the "tool-calling" feature, which was the blocker for the previous model.

-   **Phase 1: Vision Sanity Check**
    -   **Objective:** To perform a basic test of the model's core visual understanding.
    -   **Implementation:** A new script, `src/core/vl_models/vision_sanity_check.py`, was created.
    -   **Methodology:** This script uses the existing `VLExtractor` (which expects a `str` output) to ask the model to identify the animal and its primary color from `cat.jpg` and `dog.jpg`. This tests the model's ability to follow simple instructions and describe an image without complex formatting requirements.

-   **Phase 2: Tool-Calling Proof of Concept (PoC)**
    -   **Objective:** To verify if the new model truly supports the `pydantic_ai` tool-calling feature for structured data output.
    -   **Implementation:** A second new script, `src/core/vl_models/tool_calling_poc.py`, was created.
    -   **Methodology:** This script defines a simple `IdentifiedAnimal` Pydantic model with `animal_name`, `color`, and `confidence` fields. It then configures a `pydantic_ai` Agent with `output_type=IdentifiedAnimal`, directly testing if the model can return a structured Pydantic object instead of a raw string.

-   **To-Do / Next Steps:**
    1.  The user will prepare the `cat.jpg` and `dog.jpg` image files in the `illustrations` directory.
    2.  The user will execute the Phase 1 test: `python src/core/vl_models/vision_sanity_check.py`.
    3.  If Phase 1 is successful, the user will execute the Phase 2 test: `python src.core/vl_models/tool_calling_poc.py`.
    4.  The results of these experiments will determine the final implementation strategy for the puzzle extraction feature.



## 2025-10-22 (Second Entry)

### VL Model Debugging and Strategy Pivot

Conducted a deep debugging session on the Ollama-based Vision-Language model integration (Strategy B) and established a new, phased experimental plan.

-   **Initial State:** The test script (`run_pydantic_ai_test.py`) was failing with various errors, preventing successful communication with the VL model.

-   **Debugging Journey & Discoveries:**
    1.  **`ImportError` Resolution:** A series of `ImportError` and `NameError` issues were traced back to version differences in the `pydantic_ai` library. By inspecting the locally installed package files, the correct import paths and class names (`OpenAIChatModel`, `OllamaProvider`) were identified and fixed.
    2.  **Networking `404` Error:** A `404 Not Found` error was diagnosed as a mismatch between the Docker-internal hostname (`ollama_server`) defined in the `.env` file and the required `localhost` for scripts run from the host machine. The test script was updated to explicitly use `http://localhost:11434/v1`.
    3.  **Pydantic Validation Error:** A `ValidationError` for `extra_forbidden` was resolved by configuring the `Settings` class in `src/settings.py` to ignore extra fields from the `.env` file (e.g., `svelte_port`).
    4.  **`does not support tools` Error:** The final and most critical error was a `400 Bad Request` from the Ollama server, explicitly stating that the model (`qwen2.5vl:7b`) does not support the "tool-calling" feature. This is the core mechanism `pydantic_ai` uses for structured JSON output.

-   **Analysis of External Resources:** Based on user-provided research, it was confirmed that:
    *   Instruction-tuned model variants (e.g., `...-Instruct`) are critical for complex tasks.
    *   A community-provided model on Ollama Hub (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`) explicitly claims to support tool-calling.

-   **Revised Strategy & Next Steps:**
    1.  **Pause on "Tool-Calling":** Per user instruction, the current "manual JSON parsing" implementation in `vl_extractor.py` will be kept as a baseline. The more advanced tool-calling implementation is deferred.
    2.  **New Model Preparation:** The immediate next step is for the user to prepare the new, more capable model (`bsahane/Qwen2.5-VL-7B-Instruct:Q4_K_M_benxh`) in their Ollama instance.
    3.  **Sanity Check:** A new test script (`vision_sanity_check.py`) will be created to perform a basic vision test (e.g., identifying a cat/dog) using the new model. This validates the model's core visual processing before attempting complex extraction.
    4.  **Proof of Concept:** A separate script (`tool_calling_poc.py`) will be created to demonstrate and validate the "tool-calling" capability of the new model in isolation.


## 2025-10-22

### Vision-Language Model Integration Strategy

Analyzed the new requirement to parse puzzles from uploaded images using a Vision-Language (VL) model. Two parallel implementation strategies were identified in the existing codebase (`src/core/vl_models/`).

-   **Strategy A: Integrated Hugging Face Transformers (`hf_parser.py`)**
    -   **Architecture:** Loads and runs a VL model (e.g., `Qwen/Qwen3-VL-4B-Thinking`) directly within the main application process using the `transformers` library.
    -   **Pros:** Self-contained, simplifies the end-to-end testing of the core extraction logic. The existing script appears more mature and includes a runnable test block.
    -   **Cons:** Tightly couples the main application with the resource-intensive VL model, potentially leading to high memory (VRAM) consumption.

-   **Strategy B: Microservice with Ollama (`vl_extractor.py`, `docker-compose.yml`)**
    -   **Architecture:** Defines a separate `ollama` service in Docker Compose to host the VL model. The main application communicates with it via an API, using `pydantic_ai` as a client.
    -   **Pros:** Superior service-oriented design. Decouples the VL model from the main application, improving scalability and reducing the main application's resource footprint. This is the preferred final architecture.
    -   **Cons:** The current implementation is more preliminary and introduces the complexity of inter-service communication and dependency on an external service.

-   **Identified Issues & Decisions:**
    -   A key inconsistency was found: the term for walls is `walls` in `hf_parser.py` but `blocked_cells` in other files. This must be standardized to `walls` to match the existing solver framework.
    -   **Decision:** The development will proceed in a phased approach. First, **Strategy A** will be completed to quickly deliver a functional end-to-end feature. Subsequently, this implementation can be refactored to follow the more robust **Strategy B** microservice architecture.

### Next Steps

-   Proceed with completing Strategy A (`hf_parser.py`).
-   Standardize all data structures and prompts in the `vl_models` directory to use the `walls` keyword and the `WallPair` Pydantic model for consistency.
-   Develop a standalone test script to validate the image-to-dictionary conversion before API and UI integration.



## 2025-10-21

### Dockerized Development Workflow Automation

To streamline the development process and simplify the startup of the containerized environment, this commit introduces a new automation script and enhances the project's containerization strategy.

-   **Docker Compose Enhancement**:
    -   The `docker-compose.yml` file was updated to define a complete, multi-service development environment, including the FastAPI backend (`zip-challenge-app`) and the Svelte frontend (`svelte-frontend`).
    -   Configuration was refined to ensure proper volume mounting for live code reloading and inter-container communication.

-   **Automated Startup Script**:
    -   Created a new Python script, `run_docker_dev.py`, to provide a one-command solution for launching the entire development stack.
    -   The script automates the following sequence:
        1.  Stops and removes any existing containers (`docker compose down`).
        2.  Builds fresh images and starts all services in the background (`docker compose up --build -d`).
        3.  Waits briefly for the main application container to initialize.
        4.  Executes the command to start the FastAPI server inside the running container, ensuring the virtual environment is activated.
    -   This script eliminates the need for manual `docker exec` commands and simplifies the developer onboarding experience.

-   **Documentation Update**:
    -   Updated the `README.md` and `README_zh-TW.md` files with a new "Running with Docker" section, explaining how to use the `run_docker_dev.py` script.
    -   This ensures that the documentation is synchronized with the latest, most efficient development workflow.

## 2025-10-20 (another commit)

### Svelte Frontend UX and Test Suite Refinements

This commit enhances the Svelte frontend's user experience and ensures the stability of the existing test suite.

-   **Svelte UI Enhancements**:
    -   **In-place Cell Editing**: The cell editing UX was significantly improved by replacing the browser's default `prompt()` dialog. A new, dynamic in-place editing mechanism was implemented. Now, clicking a cell overlays an `<input>` element directly onto the canvas grid, allowing for a more seamless and intuitive editing workflow.
-   **Test Suite Maintenance**:
    -   **Gradio Test Fix**: Corrected a failing test case in `test_gradio_app.py`. The assertion was updated to correctly handle the HTML-formatted error messages now returned by the Gradio UI, bringing the test suite back to a passing state.

## 2025-10-20

### Gradio UI Overhaul and Interactive Solver Implementation

This phase focused on building a highly interactive and user-friendly puzzle editor within the Gradio web UI, moving from a text-based input to a full "What You See Is What You Get" (WYSIWYG) experience.

-   **Interactive Puzzle Editor ("V2")**:
    -   Replaced the initial text-based "naive" solver tab with a new "Interactive" tab.
    -   Implemented a dynamic grid creation system where users can specify puzzle dimensions (`m x n`).
    -   **Refactored Wall Editor**: Based on user feedback regarding the initial confusing checkbox-based UI, the wall editor was completely redesigned.
        -   Users now input wall coordinates using four simple number boxes (`r1, c1, r2, c2`).
        -   A list view displays all current walls, with a proper "select-then-click" button to delete walls.
    -   **Live Image Preview**: Added a new preview panel that generates and displays an image of the puzzle in real-time. The preview automatically updates whenever the user edits the puzzle grid (adding numbers/obstacles) or modifies the wall list.
    -   **New UI Controls**: Implemented a "Reset" button to clear all interactive components to their default state.

-   **Debugging and Stability**:
    -   **Extensive Bug Fixing**: Resolved a long series of bugs discovered during iterative development, including `IndentationError`, `NameError`, `AttributeError`, `UnboundLocalError`, and several data format mismatches between the frontend and backend (e.g., `'x'` vs `'xx'`, `dict` vs `set`).
    -   **Enhanced Logging**: Added detailed `loguru` logging to both the frontend (`gradio_app.py`) and backend (`solver.py`). These logs capture the raw UI payload and the parsed puzzle data, which was critical in diagnosing the data flow issues. Also added logging for temporary file deletion in the backend.
    -   **Code Maintenance**: Fixed a `FutureWarning` from the `pandas` library by migrating from the deprecated `Styler.applymap` to `Styler.map`.

-   **Architectural Refinements**:
    -   The frontend `gradio_app.py` was refactored multiple times to serve as a robust "Adapter", translating intuitive user actions into the precise data formats expected by the backend API.
    -   The core backend logic in `utils.py` and `solver.py` was validated and corrected to ensure it properly handles obstacles and other puzzle constraints.

## 2025-10-16

### Implementation of Service-Oriented Architecture (Phase 1)

Following the pivot from pure algorithmic development, the first phase of the user-facing web service has been implemented. This phase establishes the core architecture and a functional user interface.

-   **Web Service Backend (FastAPI):**
    -   Initialized a FastAPI application (`src/app/main.py`) to serve as the backend.
    -   Implemented a robust, layered configuration system using `pydantic-settings` (`src/settings.py`) that reads from a `.env` file, making settings like port numbers easily configurable.
    -   Refactored the API structure into a scalable `routers` and `schemas` pattern. All API endpoints are now modularly organized (e.g., `src/app/routers/echo.py`, `src/app/routers/solver.py`).
    -   Created a `/api/solver/solve` endpoint that receives puzzle data, calls the appropriate core solver, and returns a JSON response containing the solution path and Base64-encoded images.
    -   Improved code quality by replacing magic numbers for HTTP status codes with `fastapi.status` constants.

-   **Web User Interface (Gradio):**
    -   Developed a multi-tab Gradio interface (`src/ui/gradio_app.py`) for user interaction.
    -   The UI is mounted directly within the FastAPI application, creating a single, unified service.
    -   Implemented a "Puzzle Solver naive version" tab that allows users to paste puzzle layouts and walls, select a solver, and receive a visual solution.
    -   The UI now displays both an animated GIF of the solution process and a static image of the final result.

-   **Visualization Enhancements:**
    -   Created a new `save_detailed_animation_as_gif` function in `utils.py` to generate GIFs with enhanced visuals, including a highlighted path head (blue) and sequential step numbers (green).
    -   Added a `save_solution_as_image` function to generate a static PNG of the final solved puzzle.
    -   The backend now uses these new functions to provide richer visual feedback to the user.

-   **Bug Fixes & Refinements:**
    -   Standardized file path comments in `.py` files to use forward slashes (`/`) for cross-platform consistency.
    -   Resolved a `PermissionError` on Windows related to `tempfile` by implementing a more robust file handling pattern in the solver API.
    -   Corrected multiple `IndentationError` syntax issues that arose during refactoring.
    -   Standardized type hint styles in Pydantic schemas to the modern `|` union operator as per project conventions.

### Quality Assurance and Refactoring

-   **Unit Test Implementation**: Added a comprehensive suite of unit tests for the new service-oriented architecture. This includes API endpoint tests using `TestClient` (`src/app/tests/`), UI logic tests using `unittest.mock` (`src/ui/tests/`), and smoke tests for new visualization utilities in `src/core/tests/`.
-   **Project Structure Refactoring**: To improve modularity, moved `puzzle_generator.py` and `generate_dataset.py` into a new dedicated `src/core/puzzle_generation/` directory and updated all corresponding import paths across the project.
-   **Performance Tuning**: Modified the `generate_dataset.py` script to limit the multiprocessing pool to 75% of available CPU cores, ensuring system responsiveness during heavy computation.

### Next Steps

-   **Interactive UI**: Implement the "Puzzle Solver interact version" tab in the Gradio UI.
-   **Containerization**: Introduce a `Dockerfile` to allow the entire web service to be built and run as a container.

## 2025-10-15

### Reinforcement Learning Development Paused

Due to the inherent challenges in reward function design and overall training complexity, the Reinforcement Learning (RL) development effort is being temporarily paused.

Future work in this area will be resumed after a period of deeper research into advanced RL concepts and architectures. The planned areas of study include:
-   Architectures of seminal models like **AlphaGo** and **AlphaZero**.
-   Reviewing the hands-on examples in the local `more_simple_reinforcement_learning` directory.
-   Studying the "Hands-on Reinforcement Learning" course materials (from `hrl.boyuai.com`).

When RL development resumes, a revised approach will be considered to simplify the problem, such as:
-   Reducing the `map_size` to a smaller dimension.
-   Relaxing the environment's constraints (e.g., allowing the agent to revisit paths, transforming the problem from finding a single Hamiltonian path to a more flexible pathfinding task).

### Project Pivot to Service-Oriented Architecture

The project's immediate focus will shift from algorithmic development to building a user-facing service. The goal is to create an application with a UI that allows users to upload their own puzzles and receive a computed solution.

### New To-Do List

-   **Service Backend:** Implement a web backend using **FastAPI**.
-   **User Interface:** Create an interactive web UI with **Gradio**.
-   **Future Exploration:** Investigate the integration of **MCP (Model-View-Controller Pattern)** and **multi-modal** capabilities.

### Archived Progress 

*This section documents the last active development goal before the pivot.*

The previous focus was on attempting to solve a 6x6 map using an RL approach. The strategy was to first test and solve the problem on a **single map** (i.e., achieve overfitting) as a proof of concept. The successful completion of this step would then serve as a foundation for the ultimate goal of **generalizing** the solution to arbitrary 6x6 maps. The starting point for this development was the implementation of the `src/core/rl/train_single_sb.py` script.

## 2025-10-13

### Deep Dive into Deterministic Loop & Reward Shaping

Following the successful overfitting of the MLP-based model during training and its subsequent failure in deterministic evaluation, a series of experiments were conducted to resolve the underlying "deterministic policy loop" issue with a new CNN-based model.

-   **Problem Persistence & State Representation Fix**: Despite refactoring the environment to use a 6-channel image-like state representation (including separate layers for walls and obstacles) and switching to a `CnnPolicy`, the agent continued to fail during deterministic evaluation. It achieved high rewards during training (with exploration) but fell into inescapable loops when `deterministic=True`.

-   **Hypothesis 1: Insufficient Penalty for Inefficiency.** The first hypothesis was that the `-1.0` time penalty was not enough to discourage looping.
    -   **Experiment:** A "soft constraint" was added to the reward function in `rl_env.py`, applying a `-2.0` penalty for revisiting any cell already in the `path_taken`.
    -   **Result:** **Failure.** The evaluation log (`evaluation_path_2025-10-13_13-37-36.log`) showed that while the agent explored more territory, it ultimately still fell into a tight loop (`(4, 0) <-> (5, 0)`), indicating the revisit penalty was not sufficient to overcome the root cause.

-   **Hypothesis 2: Dense Reward Traps.** The primary suspect shifted to the distance-based reward shaping (`(dist_before - dist_after) * weight`), which could be creating local optima ("reward traps") that are more attractive than exploring a path to the true goal.
    -   **Experiment:** The reward shaping weight was reduced by an order of magnitude, from `0.1` to `0.01`. The parameter was also refactored into the `PuzzleEnv` constructor and the training script's `CONFIG` for easier tuning.
    -   **Result:** **Failure.** The evaluation log (`evaluation_path_2025-10-13_14-05-36.log`) again showed the agent getting stuck in a terminal loop, proving that even a very small positive incentive towards the goal can create a powerful enough trap to derail the deterministic policy.

-   **Final Diagnosis:** The distance-based reward shaping, even with a minimal weight, is fundamentally at odds with the sparse penalty system. It encourages a "greedy" local-optimization behavior that results in policy loops. The agent is unwilling to incur a small penalty (by moving away from the target) to find a path around an obstacle, as the dense reward signal is too dominant.

### To-Do List

-   **[Next Step]** Completely eliminate the dense reward signal by setting `DISTANCE_REWARD_WEIGHT` to `0` in `train_single_cnn_sb.py`.
-   Re-train the model from scratch using the purely sparse reward function (only step/revisit/invalid penalties and waypoint/goal rewards).
-   Perform a deterministic evaluation on the new model to verify if the looping issue is finally resolved.
-   If the issue persists, the final recourse is to escalate the "soft constraint" on revisits to a "hard constraint" by making it an invalid move.

## 2025-10-12

### RL Agent Deep Debugging and Analysis

A deep-dive debugging session was conducted to diagnose why the DQN agent, despite successful training metrics, failed during deterministic evaluation.

-   **Initial State & Problem:** The agent, whether custom-built or using `stable-baselines3`, showed high average rewards during training but consistently failed to complete a puzzle during deterministic evaluation (`epsilon=0`), always timing out at the maximum step limit.

-   **Hypothesis 1: Insufficient Evaluation Steps.** The initial hypothesis was that the evaluation loop's step limit was too low. This was proven false, as increasing the limit in the evaluation script had no effect. The root cause was identified as a hardcoded `_max_steps` limit within the `PuzzleEnv` itself.

-   **Hypothesis 2: Flawed Reward Shaping.** The second hypothesis was that the distance-based reward shaping (`(dist_before - dist_after) * 1.0`) was creating a "reward trap" or local optimum, causing the agent to loop near the goal. An experiment was conducted by reducing the shaping weight to `0.1`. While this produced even better training metrics, the deterministic evaluation still failed in the exact same manner.

-   **Final Diagnosis: Deterministic Policy Loop.** The conclusive diagnosis is that the agent's learned deterministic policy contains an inescapable loop. The successful, shorter-episode training runs were an illusion created by random exploration (`epsilon > 0`) accidentally "bumping" the agent out of its learned loop, allowing it to reach the goal. When this randomness is removed, the policy's fatal flaw is revealed.

-   **Framework Enhancement:** To facilitate debugging, the `PuzzleEnv` was refactored to allow its `max_steps` limit to be configured externally during instantiation. The evaluation scripts (`evaluate_sb.py`) were updated to use this new parameter, providing a more flexible testing environment.

### Reinforcement Learning Framework Q&A

A summary of the RL agent's core mechanics was documented to clarify understanding.

-   **Q1: What are the agent's movement rules?**
    -   The agent has a discrete action space (Up, Down, Left, Right). It is permitted to reverse its direction and revisit cells it has previously occupied. There are no rules preventing revisits.

-   **Q2: What is the agent's goal and behavior?**
    -   **Goal:** To navigate from a starting position, visiting a sequence of numbered waypoints in the correct order, and finally arriving at the last waypoint.
    -   **Behavior:** The agent's behavior is governed by a policy network (an MLP). This network takes the current state (`agent_location`, `next_waypoint_location`) and outputs Q-values for each of the four actions. The agent selects the action with the highest Q-value, which it predicts will lead to the maximum cumulative future reward.

-   **Q3: How does the agent interact with the environment?**
    -   The interaction follows the standard RL loop. The agent submits an `action` to the environment via `env.step(action)`. The environment transitions to a `next_state` and returns a `reward`, a `terminated` flag (for goal completion), a `truncated` flag (for timeouts), and an `info` dictionary. The agent uses this feedback to update its policy.

-   **Q4: What is the reward function?**
    -   The reward function is composed of several components:
        -   `+1000.0` for reaching the final waypoint.
        -   `+200.0` for reaching an intermediate waypoint.
        -   `-10.0` for an invalid move (hitting a wall, obstacle, or boundary).
        -   `-1.0` as a time penalty for every step taken.
        -   `(dist_before - dist_after) * 0.1` as a small, dense reward for reducing the Manhattan distance to the next target.

-   **Q5: What logging is available besides the GIF animation?**
    -   **Console Logs:** Real-time statistical tables from `stable-baselines3` during training.
    -   **File Logs:** Detailed, timestamped logs saved by `loguru` to the `logs/` directory.
    -   **TensorBoard Logs:** The most powerful tool. Detailed, interactive graphs of all training metrics (reward, loss, etc.) are saved to `logs/sb_tensorboard/`. This can be launched via the command `tensorboard --logdir ./logs/sb_tensorboard/`.

### To-Do List

-   Review the visual `evaluation_sb.gif` and TensorBoard logs to pinpoint the exact location and pattern of the agent's deterministic loop.
-   Based on the loop's characteristics, redesign the reward function to specifically penalize or disincentivize the observed looping behavior.
-   If reward redesign is insufficient, consider redesigning the environment's rules of interaction (e.g., adding a penalty for immediately revisiting the previous state).

## 2025-10-12

### Reinforcement Learning (RL) Solver Framework

-   **Architectural Design**: Designed a complete framework to solve puzzles using Deep Reinforcement Learning. The approach is based on a DQN (Deep Q-Network) agent interacting with a custom environment, with a focus on making the training pipeline robust and reproducible.
-   **Custom RL Environment (`rl_env.py`)**: Implemented a `gymnasium.Env`-compatible environment, `PuzzleEnv`, to wrap the puzzle logic.
    -   Features a sophisticated **reward shaping** mechanism to provide dense rewards, guiding the agent by calculating the change in Manhattan distance to the next waypoint.
    -   The state space is defined by the agent's location and the next target waypoint, making the problem tractable for a neural network.
-   **DQN Agent (`dqn_agent.py`)**: Implemented a complete DQN agent, including:
    -   A `DQNModel` (MLP) to approximate the Q-function.
    -   A `ReplayBuffer` for experience storage and sampling.
    -   The core `DQNAgent` class encapsulating the learning logic, epsilon-greedy action selection, and target network updates.
-   **Two-Stage Training Pipeline**: Decoupled data generation from training for better workflow and reproducibility.
    -   **Dataset Generation (`generate_rl_dataset.py`)**: Created a multiprocessing-enabled script to generate and save large puzzle datasets (`6x6` and `7x7`). It outputs both a human-readable log for verification and a `pickle` file for the trainer to consume.
    -   **Training Script (`train.py`)**: Developed the main training script that loads the pre-generated dataset, manages the training loop, logs progress with `tqdm` and `loguru`, and saves the final trained model.

### Code Quality and Bug Fixes

-   **Pathing Logic**: Corrected a path calculation error in `generate_rl_dataset.py` and `train.py` that resulted in an incorrect, duplicated output directory path. The logic for determining the project root was made more robust.
-   **Linter Compliance**: Resolved a `SyntaxError` reported by `ruff` in `dqn_agent.py` by refactoring a multi-line expression to be more robust, ensuring the codebase passes all `pre-commit` checks.
-   **Dependency Management**: Identified and added necessary dependencies (`gymnasium`, `torch`, `tqdm`) for the new RL framework, using the project's `uv add` workflow.

## 2025-10-08

### Puzzle Generation Framework

-   **Procedural Puzzle Generator:** Created a new, sophisticated puzzle generation module (`src/core/puzzle_generator.py`).
    -   The core logic is built upon a **randomized backtracking algorithm** (`_generate_hamiltonian_path`) that generates a guaranteed valid solution path covering all visitable cells.
    -   Introduced a robust generation process with a **retry and decrement** mechanism: if generating a puzzle with `N` obstacles fails, it automatically retries, and if still unsuccessful, it gracefully degrades to attempt generation with `N-1` obstacles.
    -   Implemented a true **internal timeout** within the pathfinding algorithm to terminate and abandon attempts that take too long, preventing the process from hanging and saving CPU resources.
-   **Automated Dataset Creation Script:** Developed a powerful script (`src/core/generate_dataset.py`) to automate the creation of large puzzle datasets.
    -   Leverages the `multiprocessing` module to generate multiple puzzles in **parallel**, significantly speeding up the process.
    -   The script is highly configurable and creates a clean, **timestamped directory structure** for each run, organizing the generated puzzle data (`puzzles.py`) and GIF animations (`gifs/`) separately.
    -   Waypoint count is now **dynamically calculated** based on puzzle size (1/4 to 1/3 of path length) to create more balanced puzzles.

### Code Quality and Refactoring

-   **Improved Type Safety:** Introduced a `Puzzle` `TypedDict` in `utils.py` to provide a strict data contract for puzzle objects, replacing generic dictionaries and improving type safety across the codebase. All relevant functions (`puzzle_generator`, `utils`, etc.) were updated to use this precise type.
-   **DRY Principle Refactoring:** Refactored `puzzle_generator.py` to call the canonical `parse_puzzle_layout` function instead of manually re-implementing the puzzle object construction logic.
-   **Code Style and Conventions:**
    -   Standardized all new modules to use the `pathlib` library for path manipulations, adhering to project conventions.
    -   Updated all new modules to use absolute imports (e.g., `from src.core...`) as per user preference.
    -   Eliminated all "magic numbers" by defining them as named constants at the top of modules (e.g., `MAX_RETRIES_PER_COUNT`).
    -   Updated `gemini_readme_raw.md` to formally document the `pathlib` and absolute import style rules.
-   **Bug Fixes and Linting:**
    -   Fixed a critical `NameError` bug in `puzzle_generator.py` where `logger` was used but not imported.
    -   Fixed a `NameError` in the `generate_dataset.py` multiprocessing worker where `logger` was not available in the child process scope.
    *   Fixed a visual bug in `save_animation_as_gif` where `blocked_cells` were not being rendered; they are now correctly drawn as black squares.
    -   Resolved multiple `ruff` linter errors (`F841`: unused variable) in `utils.py`.

### Testing

-   **Generator Test Suite:** Created a new test file `src/core/tests/test_puzzle_generator.py`.
    -   Added a comprehensive **smoke test** (`test_generate_puzzle_smoke`) that validates the integrity of a complex generated puzzle (with walls and obstacles) and its solution.
    -   Added a dedicated test (`test_generate_puzzle_default_waypoints`) to verify the new **dynamic default waypoint calculation** logic.
-   **Standardized Test Output:** Replaced all `print()` statements in the new test file with `logger` calls to maintain consistency with project standards.

## 2025-10-04

### Expansion of Metaheuristic Solver Suite

-   **Simulated Annealing (SA) Solver:** Implemented `solve_puzzle_simulated_annealing` in a new `simulated_annealing.py` module. The development process uncovered a critical bug in the initial neighbor generation logic:
    -   An initial `2-opt` swap strategy, common in TSP-like problems, was found to produce non-contiguous paths (i.e., "jumps") on a grid. This bug was identified thanks to user feedback.
    -   The logic was corrected by replacing `2-opt` with a robust "truncate and regrow" strategy in the `_generate_neighbor_path` helper function, which guarantees path contiguity.
-   **Genetic Algorithm (GA) Solver:** Implemented `solve_puzzle_genetic_algorithm` in `genetic_algorithm.py`.
    -   To avoid the path contiguity issues inherent in traditional crossover operations, a pragmatic "no-crossover" variant was designed. 
    -   The implemented GA relies on elitism (carrying over the best solutions) and mutation (using the new `generate_neighbor_path` function) for reproduction and population evolution.
-   **Tabu Search (TS) Solver:** Implemented `solve_puzzle_tabu_search` in `tabu_search.py`.
    -   The solver uses a `collections.deque` with a fixed `maxlen` as an efficient short-term memory (the "tabu list").
    -   To save memory, hashes of path tuples (`hash(tuple(path))`) are stored in the tabu list instead of the paths themselves.
    -   An aspiration criterion is included. The logic for this criterion was significantly refined based on user feedback:
        -   A critical logical flaw in the initial implementation (`score > best_score`), where the condition would never be met for a tabu item, was identified by the user.
        -   The final, more flexible and effective implementation (`score >= aspiration_threshold * best_score`) was also proposed by the user, and the `aspiration_threshold` parameter was added accordingly.
-   **Particle Swarm Optimization (PSO) Solver:** Implemented a discrete adaptation of PSO in `particle_swarm_optimization.py`.
    -   A particle's "position" is defined as a path, and its "velocity" is defined as a list of swap operations.
    -   Discrete analogues for velocity and position updates were implemented. This approach relies on the fitness function's heavy penalty for non-contiguous "jumps" to guide the swarm toward valid paths.
    -   During a detailed review, the user correctly pointed out that the sequential application of swap operations (the "velocity") causes "distortion," as the effect of a later swap is dependent on the state change from an earlier swap. It was clarified that this is an accepted and inherent characteristic of this discrete PSO adaptation, providing a form of stochastic perturbation that aids in exploration, with the fitness function acting as the ultimate arbiter of path quality.

### Major Refactoring and Code Quality Enhancements

-   **Centralized Path Utilities:** To eliminate code duplication across solvers, the common helper functions `generate_random_path` and `generate_neighbor_path` were moved from individual solver files into the shared `src/core/utils.py` module. `monte_carlo.py` and `simulated_annealing.py` were refactored to use these new shared utilities.
-   **Fitness Function Hardening:** The `calculate_fitness_score` function in `utils.py` was made more robust. A Manhattan distance check was added to penalize non-contiguous path "jumps", which was a weakness identified during the SA implementation.
-   **Increased Test Coverage:** 
    -   Added smoke tests for all new metaheuristic solvers (SA, GA, TS, PSO) to ensure they run and produce correctly formatted output.
    -   Added new, dedicated unit tests to `test_utils.py` for the shared `generate_random_path` and `generate_neighbor_path` functions to validate their core logic (e.g., path contiguity, no duplicates, correct start point).
-   **Code Style and Linting:** Fixed several `pre-commit` errors reported by `ruff`, including an `F821 Undefined name` error from a missing `import` and an `E402 Module level import not at top of file` style violation.

## 2025-10-01

### Codebase Modernization and Toolchain Overhaul

-   **Path Handling Refactoring:** Replaced all instances of `os.path` with the modern `pathlib` library across the test suite (`conftest.py`, `test_dfs.py`). This improves path manipulation logic, making it more readable, consistent, and object-oriented.
-   **Alternative A* Solver Implementation:** Implemented a new A* solver variant, `solve_puzzle_a_star_sortedlist`, which leverages `sortedcontainers.SortedList` as its priority queue instead of the standard `heapq`. A corresponding parametrized unit test was added to `test_a_star.py` to ensure its correctness against the full puzzle suite.
-   **Pre-Commit and CI/CD Pipeline Refinement:**
    -   **Test Pathing Resolution:** Resolved a critical `ModuleNotFoundError` during test collection by migrating the Python path configuration from a `sys.path` manipulation in `conftest.py` to a centralized `pythonpath` setting in `pytest.ini`. This aligns with `pytest` best practices.
    -   **Toolchain Consolidation:** Diagnosed and fixed a persistent formatting conflict loop between `black`, `isort`, and `ruff`. The pre-commit configuration was completely refactored to use `ruff` exclusively for all linting, import sorting, and code formatting, removing `isort` and `black` for a faster and simpler CI pipeline.

### Metaheuristic Search Framework and Baseline Implementation

-   **Fitness Function Design & Implementation:**
    -   Designed and implemented a comprehensive `calculate_fitness_score` function in `utils.py`. This function establishes the core evaluation metric for all metaheuristic solvers, incorporating a system of penalties and rewards (for path length, waypoint sequencing, etc.).
    -   The function was enhanced to return both the path's current score and the puzzle's theoretical perfect score, providing a clear benchmark for solution quality.
    -   Added a full suite of unit tests in `test_utils.py` to validate the fitness function's behavior.

-   **Monte Carlo Solver:**
    -   Implemented the first metaheuristic solver, `solve_puzzle_monte_carlo`, as a baseline for performance comparison. The solver generates a specified number of random paths and returns the one with the highest fitness score.
    -   The solver's logging was integrated with the new fitness function output to display comparative scores (e.g., `Best score: 420200/1720360`).
    -   A unit test was created to verify the integrity of the Monte Carlo solver, ensuring it produces valid paths.

### Code Quality and Refactoring

-   **DRY Principle Refactoring:** Refactored all existing exact solvers (`dfs.py`, `a_star.py`, `cp.py`) to consume the `num_map` from the puzzle dictionary, eliminating redundant code.
-   **Bug Fixes:** Diagnosed and resolved multiple `NameError` exceptions in `a_star.py` and `test_utils.py` that were introduced during refactoring, ensuring the entire test suite passes.

## 2025-09-25

### Advanced Solver Implementation and Analysis

-   **A* Solver:** Implemented a complete A* solver (`a_star.py`) using a priority queue (`heapq`) and a Manhattan distance heuristic. Iteratively debugged the implementation, correcting a critical flaw in the `closed_set` logic to ensure proper state tracking, which resulted in all test cases passing.
-   **CP-SAT Solver:** Developed a solver using Google's OR-Tools (`cp.py`). Modeled the puzzle as a Constraint Satisfaction Problem, and after multiple iterations, resolved an `INFEASIBLE` status by re-modeling the problem. The final, successful implementation uses the "dummy node" technique to correctly represent a Hamiltonian path with an `AddCircuit` constraint.
-   **Algorithm Analysis:** Performed a detailed theoretical analysis of the Time and Space Complexity (TC/SC) for the DFS, A*, and CP-SAT solvers. Compared their trade-offs in terms of memory usage, practical speed, and implementation paradigm.

### Major Project Structure Refactoring

-   Relocated all solver implementations (`dfs.py`, `a_star.py`, `cp.py`) into a new, dedicated `src/core/solvers/` directory to improve modularity and separation of concerns.
-   Mirrored the source code structure within the test directory by creating `src/core/tests/solvers/` and moving the corresponding test files. This refactoring enhances test organization and future scalability.
-   Updated all relevant `import` statements across the test suite to reflect the new file locations, ensuring all 19 tests pass after the refactoring.

### To-Do List

-   **Metaheuristic Solvers:** Begin implementation of non-deterministic, metaheuristic algorithms.
    -   Define a robust **fitness/cost function** to score partial or imperfect solutions.
    -   Implement a baseline **Monte Carlo (Random Sampling) Search**.
    -   Implement other metaheuristics such as **Simulated Annealing**, **Genetic Algorithm**, or **Ant Colony Optimization**.
    -   All metaheuristic solvers should accept an `attempts` parameter to control the number of iterations.

## 2025-09-23

### Solver Verification and Visualization Overhaul

-   **DFS Solver Logic Verified:** Through a process of debugging and adding detailed logging, it was determined that the core DFS solver algorithm was logically correct. The previously observed test failures were traced back to incorrect reference solutions in the test data.
-   **Test Data Corrected:** Fixed typos in the ground-truth data within `conftest.py`, leading to all 9 unit tests passing and validating the solver's correctness.
-   **Advanced Visualization Implemented:** Iteratively redesigned and implemented multiple solution-visualization features in `utils.py` based on interactive feedback:
    -   Implemented two distinct console-based styles: a simple `[bracket]` highlighter and a more advanced ANSI background-color highlighter.
    -   Added console-based animation functions (`animate_solution_*`) to display the step-by-step pathfinding process, addressing the need to show path order.
    -   To handle layout "wobbling" during animation, the printing logic was refactored to pre-calculate and enforce a fixed grid size across all animation frames.
-   **GIF Animation Generation:** Implemented a new feature, `save_animation_as_gif`, using the Pillow library to generate and save high-quality, shareable GIF animations of puzzle solutions, complete with wall rendering.

## 2025-09-22

### Input System Refactoring and Test Data Integration

-   **Input Refactoring:** Overhauled the puzzle input system. Puzzles are now defined with a readable, text-based `puzzle_layout`, which is then processed by a dedicated `parser` in `utils.py`.
-   **Utility Functions:** Created `src/core/utils.py` to house shared functions, including the new `parse_puzzle_layout` parser and a `visualize_solution` function for displaying results.
-   **Test Data Enhancement:** Integrated the user-provided, ground-truth solutions for all six puzzles (`puzzle_01` to `puzzle_06`) into the `conftest.py` test suite, enabling strict path verification.

### To-Do List

-   **Unit Testing:** Write and pass unit tests for the new utility functions in `src/core/utils.py`.
-   **Algorithm Validation:** Run the full test suite to verify the DFS solver's correctness against all 6 ground-truth solutions.
-   **Debugging:** Based on test results, debug any discrepancies between the solver's output and the expected solutions.
-   **Visualization Polish:** Re-evaluate and possibly redesign the presentation of the visualized solution for better clarity during debugging.

## 2025-09-21

### Test Suite and Architecture Overhaul

-   **Test Case Expansion:** Transcribed and added puzzles 01 through 06 from image files into the test suite.
-   **Test Architecture Refactoring:** Refactored the entire test workflow to be scalable and reusable. Test data is now centralized in `conftest.py` and dynamically loaded into a single test function in `test_dfs.py` using `pytest.parametrize`.
-   **Input Refactoring:** Enhanced the core solver and input data structure to support "blocked cells" in addition to "walls", making the algorithm more versatile.
-   **Workflow Update:** Updated the internal Gemini README to define collaboration rules regarding package management and test execution.

## 2025-09-20

### Core Solver Implementation

-   Initialized the project structure.
-   Implemented the core puzzle-solving logic in `src/core/dfs.py` using a backtracking Depth-First Search (DFS) algorithm.
-   The solver handles grids with numbered waypoints and walls that blocking paths.

### Testing and Reporting Setup

-   Introduced `pytest` as the testing framework.
-   Created a test suite in `src/core/tests/test_dfs.py` with multiple test cases, including simple solvable puzzles, puzzles with walls, and puzzles designed to be unsolvable.
-   Iteratively refined the "unsolvable" test cases after discovering the solver was more robust than initially anticipated.

### Automation and Workflow Refinement

-   Set up `loguru` to provide detailed, professional-grade logging for test execution.
-   Configured the logger to output to timestamped files (`log_[timestamp].log`) with UTC timestamps in the filename and local timezone information in the log messages.
-   Engineered a system to automatically generate test reports that mirror the console output.
-   After exploring `pytest.ini` and `conftest.py` hooks, finalized the reporting mechanism using a `run_tests.bat` script for maximum reliability and platform consistency. This script redirects all console output to a timestamped `test_report_[timestamp].txt` file.
-   The final workflow is simplified to a single command: `.\run_tests.bat`
