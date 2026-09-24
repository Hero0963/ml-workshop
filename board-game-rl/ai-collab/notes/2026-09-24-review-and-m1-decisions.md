# 2026-09-24 討論紀錄：暫停 5.5 個月後的複習，與 M1 純 MCTS 的決策

> 這份筆記整理 2026-09-24 的討論：複習 Stage 1 學到什麼、為什麼下一步選純 MCTS、M1 的設計取捨。
> **信心標示**：「實測」＝當天實際跑過；「出處」＝讀過的本機文件；「推論」＝還沒驗證的判斷。

---

## 1. 複習：Stage 1 留下了什麼

### 1.1 四種對手，四種讓電腦下棋的方法

| 對手 | 白話 | 術語 | 戰績（出處：[`../project_guide.md`](../project_guide.md)）|
|---|---|---|---|
| Alpha-Beta | 每條路都算到底，但明顯比較差的路就提早放棄 | Minimax ＋ Alpha-Beta 剪枝，完整搜尋 | 完美解 |
| Q-Learning | 自己下很多盤，把「每個盤面走哪步好」記在一張表 | Tabular Q-Learning，Bellman 更新 | 不敗 |
| DQN | 把那張表換成神經網路，用「猜」的取代「背」的 | Q 函數近似 ＋ Experience Replay ＋ Target Network | 對 Random 當後手約 2.5% 敗率 |
| Random | 亂下 | 均勻隨機策略 | 陪練 |

### 1.2 當時學到的四件事（出處：[`../dev_log.md`](../dev_log.md) 2026-03-28、2026-04-11）

1. **只跟 Alpha-Beta 練行不通**：它是確定性對手，同一盤面永遠同一步，只會把 agent 帶到約 165 個狀態。
   ⇒ 訓練資料的**覆蓋率**跟對手強度一樣重要。
2. **Hybrid 對手**（前 1 至 3 步亂下、之後交給 Alpha-Beta）：亂下負責讓盤面夠多樣，Alpha-Beta 負責讓對手夠強。
3. **D4 對稱加上棋盤正規化**：旋轉、翻轉共 8 種對稱，加上把先後手統一成「自己是 1」的視角，一張 3,441 個狀態的表就能涵蓋所有盤面。
4. **DQN 比查表多輸 2.5%**：神經網路是用「相似盤面分享參數」換掉「每格各記各的」，
   冷門盤面就猜不準。這是**泛化換記憶**的本質代價，不是 bug。

### 1.3 井字遊戲到底多大（實測：`uv run python scripts/count_states.py`，2026-09-24）

| 量 | 數字 |
|---|---|
| 理論上限 $3^9$ | 19,683 |
| 唯一合法盤面 | **5,478**（對弈中 4,520 ＋ 終局 958）|
| 完整對弈路徑（不同的棋局）| 255,168 |
| 空盤面出發：暴力 Minimax 走過的節點 | 549,946 |
| 空盤面出發：Alpha-Beta 走過的節點 | 48,383（省 91.2%）|

`models/alphabeta_cache.json` 的 4,520 個狀態正好就是「所有對弈中的盤面」——**井字遊戲已經被窮舉完了**。
這個事實決定了 §3 的結論。

### 1.4 基線（實測，2026-09-24）

- `uv sync --locked`：成功（torch `2.11.0+cpu`、gradio 6.9.0、Python 3.13.4）。
- `pytest`：暫停前的 28 個測試全過；補上 12 個 Q-Learning 測試後共 40 個全過。
- Alpha-Beta 在空盤面想第一步約 42 ms（`logging` 關掉後量的，單次、未重複）。

---

## 2. 下一步為什麼是純 MCTS

### 2.1 MCTS 一句話

**白話**：Alpha-Beta 是「每條路都算到底」。MCTS 改成從目前盤面「隨機下完很多盤」，比較常贏的那步就多試幾次，
但試得很少的步也給一點機會。它**不需要知道「怎樣的盤面算好」**，只要會照規則推演就能下。

**術語**：每次模擬四個階段——Selection（用 UCB1 往下挑）→ Expansion（長一個新節點）→ Simulation（隨機下到終局）
→ Backpropagation（把勝負沿路寫回）。Selection 把「挑哪個子節點」當成多臂吃角子老虎問題：
$\text{UCB1}_j = \bar{X}_j + c\sqrt{\ln N / n_j}$，前一項偏向目前看起來好的（exploitation），後一項給試得少的加分（exploration）。
這套做法叫 UCT（Kocsis & Szepesvári 2006，[Springer](https://link.springer.com/chapter/10.1007/11871842_29)、
[Chess Programming Wiki: UCT](https://chessprogramming.org/UCT)）。完整教材會在 M1 寫進 `docs/`。

### 2.2 三個候選，為什麼選這個

| 候選 | 結論 | 理由 |
|---|---|---|
| **純 MCTS（井字遊戲）** | ✅ 選它 | 只多一個新東西（搜尋），而且有**完美裁判**：Alpha-Beta 能告訴我們它對不對 |
| 直接做 AlphaZero | ❌ 往後排 | 一次多兩個新東西（搜尋＋神經網路）。下得爛的時候**分不出是搜尋寫錯還是網路沒學好** |
| 先換 Connect Four | ❌ 往後排 | 同時換遊戲又換演算法，一樣是兩個變數；而且 repo 裡還沒有 Connect Four 的完美對手可當裁判 |
| 繼續調 DQN | ❌ 已定案不做 | 2026-04-11 就決定了（見 [`../roadmap.md`](../roadmap.md)「已定案」）|

**原則**：一次只改一個變數。這和 thread-the-grid 的教訓一樣——同時改好幾個東西，就不知道是哪一個造成結果。

### 2.3 和 thread-the-grid 的連結（出處，推論部分另外標）

- thread-the-grid 量到「**推論時做搜尋**」是最大的槓桿（best-of-N 抽樣），
  見 [`thread-the-grid/ai-collab/notes/2026-09-24-qa-grpo-search-and-input.md`](../../../thread-the-grid/ai-collab/notes/2026-09-24-qa-grpo-search-and-input.md)。
- 那邊也做了 **ExIt（Expert Iteration）第一輪**：「師父」是「抽 32 次＋判定器」，**不是 MCTS**，
  見 [`thread-the-grid/ai-collab/notes/01-rl-methods-explained.md`](../../../thread-the-grid/ai-collab/notes/01-rl-methods-explained.md) §5。
- **AlphaZero 就是「師父換成 MCTS」的 ExIt**：MCTS 搜出比網路更好的走法 → 網路學著模仿 → 更好的網路讓 MCTS 搜得更好。
  所以 M1 → M2 等於在一個有完美裁判的乾淨環境裡，把 thread-the-grid 摸索出來的結論用教科書版本重做一次。
  （推論：兩者在概念上的對應是成立的；「在井字遊戲上會不會觀察到同樣的效果」要等 M2 量過才知道。）

---

## 3. 井字遊戲上，MCTS 贏不了 Alpha-Beta 快取——那 M1 還值得做嗎？

**會贏不了是確定的**：§1.3 說明整個遊戲只有 4,520 個對弈中盤面，Alpha-Beta 快取把每一個的最佳解都存好了，
查表是 O(1)，而且保證最優。MCTS 最好的結果也只是「追平」。

**還是值得做，因為 M1 的目標不是比強**，而是三件事：

1. **把演算法做對**：有 Alpha-Beta 當裁判，對錯一目了然。之後到 Connect Four 就沒有這種裁判了。
2. **量出一條曲線**：模擬次數從 10 加到 1,600，棋力怎麼變？要多少次才不會輸？這條曲線是 MCTS 最核心的性格。
3. **看懂它的弱點**：純 MCTS 靠隨機推演評估盤面，遇到「只有一步能活、其他都輸」的陷阱會被平均值騙到。
   在小遊戲上看得到它怎麼被騙，比在大遊戲上盲猜容易。

**MCTS 的優勢在哪裡才會出現**：在查表放不下、也寫不出好評估函數的遊戲（圍棋是經典例子）。
那是 M3 以後的事。

---

## 4. M1 的設計取捨（2026-09-24 本人確認，全部定案）

本人同時把**今天的目標**定為「能在 Gradio 跟 MCTS 下棋」，所以 S5（接上 API 與 Gradio）提前到 S3 掃描之前做。

| # | 決策 | 推薦 | 被否決的做法與理由 | 狀態 |
|---|---|---|---|---|
| D1 | MCTS 放哪、怎麼拿到規則 | `agents/mcts_agent.py`（通用）＋ `games/base.py` 定義 `GameRules` 介面 ＋ `games/tic_tac_toe/rules.py` 實作 | 放 `games/tic_tac_toe/`（像 Alpha-Beta）：M3 要整份重寫。直接複製 `TicTacToeEngine` 來推演：它是可變物件、用 2D list，複製又慢又容易改錯 | 定案 |
| D2 | 盤面怎麼表示 | 不可變的 `tuple`（9 格）＋輪到誰 | numpy array：不能當 dict key（`hash()` 直接 `TypeError`），9 格的小陣列反而比 tuple 慢（實測見下）。2D list：可變，樹節點共享時會被意外改到 | 定案 |
| D3 | 分數從誰的角度記 | 每個節點記「**走進這個節點的那一方**」的得分：贏 1、平 0.5、輸 0 | 用 [-1, 1]：UCB1 的理論保證假設獎勵落在 [0, 1]，$c=\sqrt{2}$ 就是那個設定下的值（Auer, Cesa-Bianchi & Fischer 2002，[PDF](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)）。分 max／min 兩層寫：兩份幾乎一樣的程式碼，容易一邊改一邊忘 | 定案 |
| D4 | 搜完選哪一步 | 被拜訪**最多次**的子節點 | 選平均分最高的：可能只試了 2 次、剛好運氣好 | 定案 |
| D5 | 探索常數 $c$ | 固定 $\sqrt{2}$，M1 不調 | 邊做邊調：會和「模擬次數」混在一起，變成兩個變數 | 定案 |
| D6 | 要不要加強化手段 | **都不加**：不重用上一步的樹、不加「一步必勝直接下」、推演用純隨機、不用對稱合併節點、不平行化 | 這些都是 M1 之後的延伸實驗，一次加一個才量得出各自的效果 | 定案 |
| D7 | 隨機性 | 每個 agent 自帶 `random.Random(seed)` | 用全域 `random`：實驗和測試都無法重現（Q-Learning agent 目前用全域的，不改它）| 定案 |

**D1 的 Python 類比**：`typing.Protocol` 是 duck typing 的正式寫法——不用繼承，只要物件有那幾個方法、簽名對得上，
型別檢查就當它符合介面（結構化子型別）。好處是 `games/` 不需要 import `agents/` 的任何東西。

**D2 的實測**（2026-09-24，`timeit` 各跑 200,000 次取平均，本機單次量測，只看數量級）：

| 操作 | `tuple` | numpy `int8` 陣列 |
|---|---|---|
| 列出合法步 | 281 ns | 1,449 ns |
| 產生下一個盤面 | 109 ns | 198 ns |

MCTS 每次模擬都要做好幾次這兩件事，所以 5 倍的差距會直接反映在「每秒能模擬幾盤」。
numpy 的優勢在大陣列的批次運算；9 個數字的時候，每次呼叫的固定開銷反而佔大頭。

**D2 的已知技術債**：「判斷誰贏」會變成第三份（`engine.py` 和 `alphabeta_agent.py` 已各一份）。
M1 **不順手重構**（外科式改動），改用**差分測試**保證三份一致：隨機下很多盤，每一步比對新規則和 `TicTacToeEngine` 的合法步與勝負。

**D3 為什麼是最容易錯的地方**：節點記的是「走進來那一方」的分數，所以往上寫回時，每一層的「贏」是對不同的人而言。
寫反的話，MCTS 會認真地**幫對手下棋**。這一段原本規劃由本人親手寫，本人後來改請 agent 代寫；
抓錯的兩個測試照樣保留：「有一步必勝要拿」「對手有一步必勝要擋」。

### 4.1 實作時多做的決定（2026-09-24）

| 決定 | 理由 |
|---|---|
| 泛型用 `TypeVar` ＋ `Generic[...]`，**不用** Python 3.12 的 `class Node[S]:` 語法 | repo 根的 ruff 設定以 `requires-python = ">=3.9"` 為目標，新語法會被當成語法錯誤。子專案本身是 3.13，但 lint 走的是根設定 |
| `legal_actions()` 只保證在「還沒分出勝負」時有意義 | 每個呼叫點本來就先問 `winner()`；讓 `legal_actions()` 自己再判一次勝負，等於每步多算一次 8 條線 |
| 新節點的 `untried_actions` 先洗牌 | 展開順序若固定（0→8），拜訪次數平手時永遠偏向同一格；洗牌後平手變成隨機，而且同一個 seed 仍可重現 |
| API／Gradio 的模擬次數暫定 **2,000**（具名常數 `MCTS_SIMULATIONS`）| 實測 1,000 次只要約 15 ms，2,000 次仍是即時回應；正式數字等 S3 量出門檻再換 |

### 4.2 實作時量到的事（實測）

- **速度**：空盤面每秒約 **67,000 次模擬**（1,000 次 14.8 ms、10,000 次 149.3 ms；seed 0，單次量測）。
  比寫計畫時的推估快一個數量級，所以 S3 的掃描不需要特別排時間，也不必平行化。
- **測試真的抓得到錯**（突變測試）：
  - 規則：故意拿掉一條勝利線 → 差分測試失敗。
  - MCTS：故意把 backpropagation 的視角寫反 → 10 個戰術與對局測試**全部**失敗。
  這證明「必勝要拿、必輸要擋」確實能抓到 D3 寫反，不是剛好通過。

---

## 5. 今天順手修的小問題

| 問題 | 處理 |
|---|---|
| [`../commands.txt`](../commands.txt) 還寫「12 個測試通過」、「當前重點任務」停在 2026-03-28 | 改成實際數字，重點任務改指向 `roadmap.md` |
| 子專案沒有 `roadmap.md`（repo 級 `AGENTS.md §2` 預期會有；SessionStart 簡報也會讀它）| 新增 [`../roadmap.md`](../roadmap.md)；`handover.md` 的「下一步」改指向它，避免兩份正本 |
| Q-Learning agent 沒有單元測試 | 新增 `tests/agents/test_q_learning_agent.py`（12 個）：正規化、D4 對稱查表、Bellman 更新、存讀檔，以及「已提交的 Q-table 對 Alpha-Beta 先後手不輸」的回歸測試 |
| `pytest` 沒寫進 `pyproject.toml`：乾淨的 `uv sync` 之後 `uv run pytest` 會找不到 | 依 `AGENTS.md §5` 規則 3，新增套件由本人執行 `uv add --dev pytest`；加入前先用 `uv run --with pytest pytest` |
| 新的討論與決策沒有地方放 | 新增本資料夾與 [`README.md`](README.md)（分工表） |

---

## 6. 雙前端：同一套 API、兩種前端（2026-09-24 討論；本人決定列為 future work）

> **結論**：先不做，寫進 [`../handover.md`](../handover.md)「Future Work」。當天只用 Gradio 跟 MCTS 下棋。

**問題**：thread-the-grid 有 Gradio 以外的第二個前端，這裡能不能也做到「後端 API 同一套、前端兩種」？

**thread-the-grid 的做法**（出處：該子專案原始碼）：
- **一個 FastAPI process、一個埠**（7440）同時提供三樣東西（`thread-the-grid/src/app/main.py`）：
  `/api/...` 端點、`/ui`（Gradio，用 `gr.mount_gradio_app` 掛上去）、`/svelte-ui`（Svelte＋Vite 建置出來的靜態檔，用 `StaticFiles` 掛上去）。
- **兩個前端都走 HTTP 呼叫同一組 API**：Gradio 用 `requests.post`（`src/ui/gradio_app.py`），Svelte 用 `fetch`（`Index.svelte`）。
- **清單類資料由 API 提供**：Svelte 的 solver 下拉選單來自 `GET /api/solver/list`，前端不寫死。

**board-game-rl 現況**（出處：本子專案原始碼）：
- 兩個獨立 process：`start.sh` 分別起 uvicorn（8000）與 Gradio（7860）。
- **Gradio 沒有經過 API**：它直接 import `get_optimal_move`（`ui/gradio_app.py`），`/predict` 目前沒有任何前端在用。
- 勝負判定只存在 Gradio 端（用 `TicTacToeEngine` 自己算）；對手清單是寫死的字串，`inference.py` 用子字串比對（`"MCTS" in agent_type`）。

**評估：做得到，差距在三件事**
1. **單一入口**：FastAPI 掛上 Gradio 與靜態前端，一個埠全包。
2. **Gradio 改走 HTTP**（和 thread-the-grid 一樣），兩個前端才真的吃同一份 API 契約；API 壞了兩邊一起壞，不會有「Gradio 正常、另一邊壞掉」的落差。
3. **API 補上前端需要、但現在只有 Gradio 內部才有的東西**：對手清單、下完之後的盤面與勝負。

**推薦（之後真的要做時的起點）**
- 勝負判定放**後端**：API 回傳「agent 的那一步＋下完後的盤面＋勝負」，前端只負責畫，規則不在 JavaScript 裡再寫一份。
- 第二個前端用 **Svelte＋Vite**：和 thread-the-grid 同一套工具鏈（本機 Node v22.19.0／npm 11.5.2）。代價是多一個 `npm run build` 步驟與 `node_modules/`。
- **時機**：另開一條 track、寫自己的計畫書；它會動到 `api/`、`ui/`、`inference.py`，和 M1 剩下的 S3／S4／S6（腳本與文件）幾乎不重疊。

---

## 7. 問答（2026-09-24）

### Q1：`http://localhost:7860` 是用 Docker 啟動的嗎？

**不是。** 它是本機直接跑的 Python process（實測：7860 埠的擁有者是這個 worktree 的
`board-game-rl/.venv/Scripts/python.exe src/board_game_rl/ui/gradio_app.py`；當時 `docker ps` 沒有任何容器）。
啟動設定在 repo 根的 `.claude/launch.json`（`board-game-rl-gradio`），等同在 `board-game-rl/` 執行：

```bash
uv run --locked python src/board_game_rl/ui/gradio_app.py
```

| | 本機直接跑（今天用的）| Docker（`docker compose up -d`）|
|---|---|---|
| 跑的是什麼 | 只有 Gradio（7860），Gradio 在同一個 process 裡直接呼叫 `get_optimal_move` | `start.sh` 同時起 FastAPI（8000）與 Gradio（7860）|
| Python 環境 | 本機 `uv` 建的 `.venv`（torch CPU 版）| 映像檔裡的環境（`Dockerfile` 基底是 `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`）|
| 改程式後 | 要重啟 process | `./src` 掛進容器；依 [`../rules.md`](../rules.md)「執行環境」會自動重載（今天沒驗證）|

今天的目標只是「能下棋」，Gradio 不經過 API，所以不需要 FastAPI，本機跑最快。Docker 那條路今天沒有驗證。

### Q2：MCTS 跟 backpropagation 是什麼關係？

**白話**：MCTS 每想一次（一次模擬）分四步：**挑路 → 長一個新節點 → 隨便下到終局 → 把結果一路報回去**。
backpropagation 就是第四步「報回去」：沿著剛才走過的路往回走，每個節點記一筆「又被走過一次，而且這次對『走進我的那個人』來說是贏／和／輸」。
沒有這一步，前面三步的結果就沒有留下來；下一次模擬挑路（用 UCB1 算分數）時，看到的還是一棵什麼都不知道的樹。

**術語**：MCTS 的 backpropagation 是**統計量的回填**（更新 $n$ 與 $\sum$ 得分），不是神經網路的 backpropagation（用鏈鎖律算梯度）。
兩者只是撞名：本專案 DQN 的 `loss.backward()`（`agents/dqn_agent.py:208`）是後者；MCTS 裡**沒有任何梯度，也沒有任何參數被訓練**。
它能變強，是因為「統計量越來越準」，不是「參數被學出來」——這也是它為什麼不需要訓練、開局就能下的原因。

**四步之間的關係**：Selection 靠的 UCB1 $= \bar{X}_j + c\sqrt{\ln N / n_j}$，裡面的 $\bar{X}_j$（平均分）和 $n_j$（拜訪次數）**全部來自 backpropagation 寫進去的數字**。
所以 backpropagation 是把「這一次模擬學到的東西」交給「下一次模擬的決策」的唯一管道。

**實際例子**（實測，seed 3；局面 `O X O / . X . / . . .`，輪到 X，X 走 7 立刻贏）：

第 6 次模擬走了兩層：X 走 5 → O 走 3 → 隨機下到終局，X 贏。回填時每一層加的分數不一樣：

| 節點 | 走進這個節點的人 | X 贏了，對他來說是 | 這次加 | 更新後 visits／score_sum |
|---|---|---|---|---|
| X 走 5 之後 → O 走 3 之後 | O | 輸 | 0 | 1 ／ 0.0 |
| X 走 5 之後 | X | 贏 | 1 | 2 ／ 2.0 |
| 根節點 | （O，名義上）| 輸 | 0 | 6 ／ 0.5 |

同一個結果「X 贏」，在 O 的節點記成 0、在 X 的節點記成 1——這就是 D3 的「視角」。
O 在「X 走 5 之後」這個節點挑子節點時，會看到「走 3」的平均分很低，於是下次改試別的走法。

200 次模擬後的根節點（X 的選擇）：

| X 走 | 拜訪次數 | X 的平均分 |
|---|---|---|
| **7** | **67** | **1.00** |
| 5 | 41 | 0.89 |
| 3 | 37 | 0.86 |
| 6 | 33 | 0.82 |
| 8 | 22 | 0.70 |

走 7 是終局節點，每次拜訪都是贏，平均分正好 1.00，UCB1 就一直把模擬分給它，所以拜訪最多、被選中。
其他步的平均分也不低（0.70 到 0.89），因為隨機推演時 O 常常忘了擋——這正是純 MCTS 用「隨機下完」估值的偏差：
**它評估的是「雙方都亂下」時的勝率，不是「雙方都下最好」時的勝率**。模擬次數夠多時，樹會長得夠深，把這個偏差蓋過去。

---

## 8. 還沒回答的問題（M1 要量的）

- 純 MCTS 需要多少次模擬，才能先後手都不輸給 Alpha-Beta？當後手會不會需要更多？
- 模擬次數很少時，它輸在哪種局面？是不是 §3 說的陷阱？
- 空盤面上，MCTS 最後偏好角落還是中央？價值估計會不會收斂到 0.5（完美對弈是和局）？
