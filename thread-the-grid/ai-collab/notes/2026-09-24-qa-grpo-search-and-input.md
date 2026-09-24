# 問答紀錄 — 2026-09-24：GRPO、往下搜尋、input 設計、端對端讀圖解題

> 專案收尾後，本人回頭問 RL 那段的幾個問題。**這一份放「懂了什麼」**：下一個人不必再問一次。
> 本輪**沒有新訓練、沒有新評估**；表格裡的數字全部引用既有報告（出處附在旁邊）。
> 外部查證日期 2026-09-24；一手／二手有標。名詞看不懂先翻 [`01-rl-methods-explained.md`](01-rl-methods-explained.md)。

---

## 一頁結論

| 問題 | 答案 |
|---|---|
| GRPO「徹底省去 Critic」對我們有幫助嗎？ | **幫助很小，不是解藥。** 它解決的兩個問題我們只有一個（critic 學不起來）；我們真正的失敗模式（多樣性崩、缺長程規劃）它不處理 |
| 模型是 CNN ＋ 參考 AlphaZero？ | **CNN 對，AlphaZero 只拿來對照、沒有實作**：沒有 MCTS、沒有 ResNet、沒有自我對弈 |
| AlphaZero 準是因為「往下探更深再判斷」？ | **對，但它用的是 MCTS，不是 alpha-beta**，而且搜尋有用的前提是有一個會評估局面的 value |
| 我們有沒有「往下探」？加了會更好嗎？ | **有推論期搜尋（best-of-N），盲目的回溯量過沒用**；缺的是「會判斷這局還解不解得開」的 value，有了它往下探才有意義 |
| 我們的 input 是 JSON 嗎？ | **不是。** 網路看到的是 8 張 8×8 的圖層 ＋ 8 個數字；JSON 只出現在「讀圖」那條路上 |
| input 設計得好嗎？ | **合理，而且不是瓶頸**；唯一值得改的是「綁死 8×8」，要跟網路架構一起改 |
| 最理想是訓練一個多模態網路直接讀圖解題？ | **做產品：不是**（「VLM 讀圖 → 精確解」比較好）；**做研究：值得**，而且正是 GRPO 最適合的地方 ⇒ **本人 2026-09-24 決定要做（路線 B）**，見 [`../roadmap.md`](../roadmap.md) 最上面 |

---

## Q1｜GRPO「徹底省去 Critic」對我們有幫助嗎？

### 白話

PPO 訓練時要請一位「評審」（critic），預估每個局面值幾分，才能判斷「這一步比預期好還是差」。
GRPO 不請評審：**同一題讓模型做好幾次，拿這幾次的平均分當及格線**，比平均好的做法加強、比平均差的削弱。

### 術語

GRPO（Group Relative Policy Optimization）是 PPO 的變體：對同一個輸入抽 G 個樣本，
以組內獎勵的平均（再除以標準差）當 baseline，取代學出來的 value function。
DeepSeekMath 論文 §4.1.1 給的理由有兩個（**一手**，[arXiv:2402.03300](https://arxiv.org/html/2402.03300)，2024-02-05）：

1. LLM 的 critic 通常是**另一顆和 policy 一樣大的模型**，吃記憶體也吃算力；
2. LLM 通常**只有最後一個 token 拿到獎勵**，要訓練一個「每個 token 都準」的 value 很難。

### 對照我們的情況

| GRPO 解決的問題 | 我們有沒有 | 出處 |
|---|---|---|
| critic 太大、吃記憶體 | **沒有**。我們的 value 只是同一個 14 MB checkpoint 裡的一個輸出頭 | [`model-weights.md`](../model-weights.md) §1 |
| 只有結尾有獎勵，critic 學不起來 | **有**。35 步才一個 +1；實測 BC 的 value head 在 PPO 微調時 explained variance ≈ −0.005，**等於什麼都沒學到** | [`rl-where-next.md`](../reports/2026-09-19_rl-where-next.md) §4.4 |
| （它沒處理）RL 微調讓多樣性崩、best-of-32 變差 | GRPO **一樣會**。Yue et al. 研究的正是這類 RLVR：pass@1 升、pass@k 降 | [`rl-where-next.md`](../reports/2026-09-19_rl-where-next.md) §4.5 |
| （它沒處理）要連對 14 次的長程規劃、搜尋沒進訓練迴圈 | GRPO 仍是「看一眼就走」的反射式策略 | [`rl-where-next.md`](../reports/2026-09-19_rl-where-next.md) §2.1–2.3 |

### 結論

- **如果以後還要做 RL 微調，用 GRPO 取代 PPO 的 critic 是合理的**——反正那個 critic 本來就沒學到東西。
- 但它**不會**讓 best-of-32 變好；要做就做「**pass@k 當獎勵**的 GRPO」，這已經是收尾報告的第 5 順位
  （[`rl-where-next.md`](../reports/2026-09-19_rl-where-next.md) §5）。
- 真正最值得做的仍是第 2 順位：**可解性 value**。

### ⚠ 兩種「value」別搞混

GRPO 省掉一個 value，我們的報告卻說「缺的就是 value」——不矛盾，它們做的是不同的事：

| | GRPO 省掉的 critic | 我們缺的可解性 value |
|---|---|---|
| 回答什麼 | 「這一步比平均好多少？」 | 「**這個局面還解得開嗎？**」 |
| 用在哪 | **訓練時**，降低梯度的雜訊 | **搜尋時**，決定要不要往下走、要倒回哪裡 |
| 標準答案從哪來 | 要自己學（這就是難處） | **精確解（CP-SAT／DFS）免費算得出來**——圍棋沒有這種東西 |

### 一個順帶的發現：ExIt 第一輪已經是「半個 GRPO」

ExIt 第一輪的做法是「每題抽 32 次 → 留下通過判定器的 → 拿來監督學習」
（[`rl-epoch-sweep-and-exit.md`](../reports/2026-09-19_rl-epoch-sweep-and-exit.md) §4）。
這就是 GRPO 的**分組抽樣**，只是**只用了正樣本**（這種做法叫 rejection-sampling fine-tuning）。
GRPO 多做的一件事是：**低於組內平均的樣本也拿來用——把它們往下壓**。
分組抽樣的程式已經在 `src/core/rl/collect_solutions.py`。

---

## Q2｜我們的模型是什麼？有參考 AlphaGo／AlphaZero 嗎？

**CNN 對。** 3 層 3×3 卷積（64 channel）→ 攤平 → `Linear(4104→256)` → policy／value 兩個輸出頭；
外殼是 sb3-contrib 的 MaskablePPO 架構（`src/core/rl/train_maskable_ppo.py` 的 `GridScalarExtractor`、
`src/core/rl/train_config.py` 的 `NetworkSettings`）。總參數 1,170,949，其中 **89.7% 在那個攤平後的全連接層**
（[`rl-lookahead-oracle.md`](../reports/2026-09-05_rl-lookahead-oracle.md) §7.1）。

**AlphaZero 只拿來對照，沒有實作。**

| | AlphaZero | 我們 |
|---|---|---|
| 網路 | 19–39 個 residual block × 256 channel | 3 層 conv × 64 channel |
| 怎麼訓練 | 自我對弈，每一步都經過 MCTS | **行為克隆**：模仿出題器給的標準路徑（服務中的 `bc_multi_456_e6`）|
| 推論 | MCTS | 抽樣 32 次，第一條通過判定器的就是答案 |
| 搜尋有沒有進訓練 | 有（這是核心）| 只有 **ExIt 第一輪**（師父是「抽 32 次＋判定器」，不是 MCTS；實驗、未上線）|

小訂正：這題不是迷宮，是**帶順序約束的 Hamiltonian path**——要走滿每一格，不是找一條到終點的路。

---

## Q3｜AlphaZero 準，是因為會往下探更深再判斷嗎？alpha-beta 剪枝呢？我們有沒有？

### 白話

對。AlphaZero 的網路像是「棋感」：看一眼就知道哪幾步像樣、這盤大概誰贏。
但它**不會只憑棋感就下**——它會沿著像樣的那幾步往下推演幾百、幾千次，用棋感評估推演到的局面，再決定這一步。
**網路負責「猜」，搜尋負責「驗證」。**

### 術語與數字

- **只用網路、不搜尋，差很多**：AlphaGo Zero 論文報告，原始網路不做任何前瞻約 **3,055 Elo**，
  加上 MCTS 是 **5,185 Elo**（**二手**：搜尋摘要轉述論文；原始 Nature 論文 PDF 本機抽不出文字，
  與 [`rl-lookahead-oracle.md`](../reports/2026-09-05_rl-lookahead-oracle.md) §8 同一個限制。
  搜尋關鍵字 `AlphaGo Zero raw neural network without lookahead Elo`）。
- **AlphaZero 用的不是 alpha-beta**：論文明寫它**不用** alpha-beta，改用通用的 Monte-Carlo tree search；
  西洋棋每秒只搜 8 萬個局面，Stockfish（alpha-beta）是 7,000 萬——**少搜約千倍，靠網路挑重點**
  （**一手**，[arXiv:1712.01815](https://ar5iv.labs.arxiv.org/html/1712.01815)）。

### alpha-beta 和 MCTS 的差別

| | alpha-beta | MCTS（AlphaZero 版）|
|---|---|---|
| 適用 | **兩人對弈**（我下一步、你下一步，minimax）| 兩人或單人都行 |
| 剪什麼 | 「對手不會讓你走到的分支」 | 不剪，**按網路給的機率＋目前平均分決定要多探哪一支** |
| 需要什麼 | 一個評估函數（手寫或學的）| policy（該探哪）＋ value（探到的局面值多少）|
| 本 repo 的實例 | **`board-game-rl/` 的井字遊戲 Alpha-Beta agent** | 沒有 |

**這題是單人謎題、沒有對手**，alpha-beta 的「剪掉對手不會選的」不適用。
單人版的對應物是「**DFS 回溯＋剪枝**」——這正是 `dfs.py`、A\*、CP-SAT 在做的事（精確解 solver）。

### 我們有沒有「往下探」？

| 搜尋形式 | 有沒有 | 量到什麼 | 出處 |
|---|---|---|---|
| 精確解 solver（DFS／A\*／CP-SAT）| **有**，而且就是 100% | 但它們不是學出來的 | — |
| 推論期重抽（best-of-N）| **有，服務中** | 6×6 best-of-16 **+0.240**，比所有訓練期改動加起來還多 | [`rl-lookahead-oracle.md`](../reports/2026-09-05_rl-lookahead-oracle.md) §6 |
| 策略排序的 DFS 回溯 | **量過** | 4×4 贏；**6×6 超過約 100 節點就輸給整局重抽** | 同上 §10 |
| 帶重啟的策略 DFS | **量過** | 兩個設定都輸，這一族結案 | 同上 §11 |
| 完美的一步前瞻剪枝 | **量過（oracle 上界）** | 只值 +0.016（4×4）／+0.029（6×6）| 同上 §5 |
| **會判斷「還解得開嗎」的 value 引導搜尋** | **沒有** | — | [`rl-where-next.md`](../reports/2026-09-19_rl-where-next.md) §5 第 2 順位 |
| 搜尋進訓練迴圈（MCTS／ExIt）| **只有 ExIt 第一輪** | 6×6 deterministic +0.125（單 seed）| [`rl-epoch-sweep-and-exit.md`](../reports/2026-09-19_rl-epoch-sweep-and-exit.md) |

### 加了會更好嗎？

**要看加的是哪一種。**

- **盲目往下探（回溯、前瞻）已經量過沒用**。原因是這題「**錯誤犯得早、爆得晚**」：6×6 有 59% 的窄化決策是四個方向全部必敗，
  等看得出來時局早就輸了；DFS 回溯是從尾巴往回改，改的全是沒用的那段。
- **AlphaZero 的搜尋有用，是因為它的 value 會評估「推演到的局面」**。我們的 value head 什麼都沒學到（Q1），
  所以就算照搬 MCTS，葉節點也沒有東西可以評估。
- ⇒ **順序是：先有可解性 value（第 2 順位）→ 再讓它引導搜尋 → 再把搜尋放進訓練迴圈（第 4 順位）。**
  這題的好處是 value 的標準答案可以用精確解免費算出來。

---

## Q4｜我們的 input 是一個 JSON 檔嗎？

**不是。** 每一層長得不一樣：

| 層 | 實際格式 | 出處 |
|---|---|---|
| API `/api/solver/solve` | JSON request body，但謎題本身是 **Python literal 字串**（`puzzle_layout_str`、`walls_str="set()"`），用 `ast.literal_eval` 解開，再交給唯一的 parser `parse_puzzle_layout()` | `src/app/routers/solver.py` |
| 讀圖路徑 | 圖片 → VLM 輸出 **JSON** → `Puzzle` dict → solver | `src/app/routers/vision.py` |
| **RL 網路實際看到的** | **8 張 8×8 圖層 ＋ 8 個數字**（下一節）| `src/core/rl/rl_env_v2.py` 的 `_get_obs()` |
| 訓練資料 | `dataset.pkl` ＋ `manifest.json` | `datasets/rl_datasets_v2/`（不進版控）|

**網路看到的 8 張圖層**（每張 8×8，盤面小於 8×8 的部分補 0）：

| # | 圖層 | 內容 |
|---|---|---|
| 0 | valid | 這格是不是盤面上的格子（分出 padding）|
| 1 | wall right | 這格右邊有沒有牆 |
| 2 | wall down | 這格下面有沒有牆 |
| 3 | visited | 走過了沒 |
| 4 | agent | 筆現在在哪 |
| 5 | next waypoint | 下一個要收的號碼在哪 |
| 6 | future waypoints | 之後的號碼，亮度 ＝ 號碼 ÷ 最大號碼 |
| 7 | done waypoints | 已收過的號碼 |

**8 個數字**：走滿比例、號碼進度、上一步方向（one-hot 4 個）、盤高 ÷ 8、盤寬 ÷ 8。

---

## Q5｜這個 input 設計得好嗎？

### 白話

整體是**合理的，而且已經有證據顯示它不是瓶頸**：模型在每一個「真正要選」的岔路口已經有約 96% 選對，
輸在要連續選對 14 次。把更多資訊塞進 input 救不了「連對 14 次」。

### 做得好的地方

1. **一張圖層一種意思**（語意分層）：卷積網路最容易學這種格式。
2. **牆只存「右」和「下」**：每道牆只記一次，不會出現「左邊說有、右邊說沒有」的矛盾；左／上的牆，3×3 卷積看一下鄰格就知道。
3. **號碼拆成「下一個／之後的／已完成」三層**：把「要依序收」這條規則直接寫進觀測，網路不必自己算下一個是誰。
4. **`valid` 圖層把 padding 和真格子分開**：所以同一個模型能吃 4／5／6×6（服務中的就是一個模型通吃三種）。
5. 非法的走法由 **action mask** 在輸出端拿掉，不靠 input 去「暗示」。

### 證據：input 不是瓶頸

| 實驗 | 結果 | 出處 |
|---|---|---|
| 把「盤面是否已裂成兩塊」直接餵進 input（從不誤報）| 策略**根本不用它** | [`rl-connectivity-feature.md`](../reports/2026-09-05_rl-connectivity-feature.md) |
| 完美的一步前瞻上界 | 只值 +0.02 | [`rl-lookahead-oracle.md`](../reports/2026-09-05_rl-lookahead-oracle.md) §5 |
| 每個真正選擇的正確率 | 服務模型 95.80% | [`rl-where-next.md`](../reports/2026-09-19_rl-where-next.md) §2.1 |

### 可以改的地方（依重要性）

1. **綁死 8×8**（最實際的缺點）：input 補到 8×8，網路再把它攤平進 `Linear(4104→256)`
   ⇒ **7×7 直接回 400**。要改得**跟網路架構一起改**（全卷積、不攤平，收尾報告第 3 順位），單改 input 沒用。
2. **「之後的號碼」用亮度編碼順序**：號碼越多，相鄰兩個號碼的亮度差（1 ÷ 最大號碼）越小，
   網路要從細微的亮度差分出「第 5 還是第 6」。可以改成只標「下一個」和「下下一個」，或標步數距離。
   **這是推論，沒量過**；而且依上面的證據，改了也很可能看不到差別。
3. **「上一步方向」其實是多餘的**：一筆畫下，從「走過的格子＋目前位置」就推得出來。無害，不急著拿掉。

**⛔ 不建議再往 input 加拓撲類特徵**（連通性、割點、動作條件版）——已經量過、已經結案
（[`handover-rl-solver.md`](../handover-rl-solver.md) §2）。

**API 那一層**：用 Python literal 字串（`"set()"`）當 JSON 欄位是早期留下來的格式，
`ast.literal_eval` 只解字面值、不執行程式碼，安全上沒問題；但對外 API 用結構化 JSON（陣列＋座標對）會比較正常。
專案已收尾，**不值得為此改 API**。

---

## Q6｜最理想是不是訓練一個多模態網路，直接讀圖解題？

### 做產品：不是

現在的「**VLM 讀圖 → 精確解 solver**」比較好：

1. **只有精確解能保證 100%**：學出來的策略沒有完備性（[`rl-where-next.md`](../reports/2026-09-19_rl-where-next.md) §2.6），
   CP-SAT 是「解得出來，或證明無解」。
2. **中間結果看得到、改得了**：VLM 讀出來的 JSON 可以顯示、可以用 Svelte 編輯器修正，錯了分得出是「讀錯」還是「解錯」。
3. **兩個難題不要疊在一起**：就算給完美的盤面，RL 策略 6×6 一次走完也只有 0.54；端對端等於還要同時學會看圖。

### 做研究：值得，而且正是 GRPO 最適合的地方

讓一個 VLM 自己推理出路徑，**由我們的判定器當獎勵**（RLVR）——這就是 restart plan 的「路線 B」
（[`2026-08-15_rl-restart-plan.html`](../reports/2026-08-15_rl-restart-plan.html) §7）。
在這條路上，GRPO 的兩個賣點**全部成立**：critic 真的是一顆 4B 級的大模型、獎勵真的只在最後。

restart plan 當時的兩個關鍵建議（2026-08-15，開工時要重新確認）：

1. **先純文字、後圖片**：先把盤面用文字餵進去，不要一開始就吃圖片——否則答錯時分不清是「看錯圖」還是「不會解」。
2. **從 4×4 起步**：7×7 一開始必然全部 0 分。

**本人 2026-09-24 決定要做這條，硬體由本人解決** ⇒ 記在 [`../roadmap.md`](../roadmap.md) 最上面。

---

## 出處

**本專案**：各節連結的報告與程式檔。

**外部**（查證 2026-09-24）：
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*（GRPO 出處，§4.1.1）— <https://arxiv.org/html/2402.03300>（**一手**）
- Silver et al., *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*（AlphaZero 用 MCTS 不用 alpha-beta、每秒搜尋局面數）— <https://ar5iv.labs.arxiv.org/html/1712.01815>（**一手**）
- Silver et al., *Mastering the game of Go without human knowledge*（原始網路 3,055 Elo vs 加 MCTS 5,185 Elo）— <https://discovery.ucl.ac.uk/10045895/1/agz_unformatted_nature.pdf>（**二手**：數字來自搜尋摘要，原始 PDF 本機抽不出文字）

搜尋關鍵字：`DeepSeekMath GRPO critic arXiv 2402.03300`、`AlphaGo Zero raw neural network without lookahead Elo`。
