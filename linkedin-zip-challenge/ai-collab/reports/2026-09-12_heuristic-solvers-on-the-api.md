# 六種啟發式 solver 上 API：為什麼要包一層驗證，以及預算為什麼不開成參數

> Track C（branch `feat/expose-heuristic-solvers`）｜量測日期 **2026-09-12**｜一手實測
> 原始數據：[`artifacts/heuristic-api-budget/budget-measurements.json`](artifacts/heuristic-api-budget/budget-measurements.json)
> 重跑方式：`cd linkedin-zip-challenge && PYTHONPATH=. uv run python ai-collab/reports/artifacts/heuristic-api-budget/measure_budget.py`
> seed `20260912`，測資是 `src/core/tests/conftest.py` 的六題（精確 solver 全部解得出來，所以每題都確定有解）

---

## 0. 一句話

**不包驗證就上線的話，91% 的回應會是「畫成答案的半成品」**——180 次原始呼叫裡有 164 次回傳了一條路，但只有 16 次真的是解。
包上「重跑到通過驗證為止、每請求固定 5 秒」之後，成功率從 **8.9% 變成 58.3%**，最慢的請求 **5.04 秒**。

---

## 1. 問題：啟發式沒有「找不到」這個回傳值

六種啟發式的共同寫法是「維護一個候選，回傳**看過最好的那個**」，而好壞由
`src/core/utils.py` 的 `calculate_fitness_score()` 決定。它們**不會**回 `None` 表示失敗——
分數低的路徑跟分數滿分的路徑，回傳型別一模一樣。

而 `POST /api/solver/solve` 與截圖端點都是拿到什麼就畫什麼。兩者接起來的結果是：
**一條少一格、或中間跳了一步的路，會被畫成 GIF 送到使用者面前，沒有任何警告。**

## 2. 量測一：不包裝，單次呼叫會回什麼

六種 solver × 六題 × 5 次重複 = **180 次呼叫**，全部用各自程式碼裡的預設參數。

| Solver | 單次是真解 | 平均單次耗時 |
|---|---|---|
| Genetic Algorithm | **8/30（26.7%）** | 0.096s |
| Simulated Annealing | 3/30（10.0%） | 0.045s |
| Tabu Search | 3/30（10.0%） | 0.037s |
| Ant Colony Optimization | 2/30（6.7%） | 0.034s |
| Particle Swarm Optimization | **0/30** | 0.038s |
| Monte Carlo | **0/30** | 0.024s |
| **合計** | **16/180 ＝ 8.9%** | 最慢單次 **0.128s** |

**其中 164 次回傳了一條路但不是解。** 這 164 就是「不包驗證會被畫出來的假答案」，
佔全部回應的 **91.1%**。

**PSO 是 0/30，而且這是它的設計決定的**：它靠「交換兩格」往全域最佳靠攏，
而交換會把一條相鄰步構成的路拆成不相鄰的跳步——它探索的空間裡，合法路徑幾乎是零測度集。
README 寫「expect it to give up」不是保守措辭，是這個數字。

## 3. 量測二：包上 `_until_verified` 之後

同樣六題，跑 registry 真正服務的那個包裝版（`HEURISTIC_TIME_BUDGET_SECONDS = 5.0`），
格子裡是「解出來時用了第幾次重跑」，`X` 是用完預算放棄：

| Solver | p01 | p02 | p03 | p04 | p05 | p06 | 解出 | 平均請求耗時 |
|---|---|---|---|---|---|---|---|---|
| Ant Colony Optimization | 2 | 27 | 71 | 67 | 3 | X | 5/6 | 1.79s |
| Genetic Algorithm | 3 | 8 | 2 | 1 | 4 | X | 5/6 | 1.10s |
| Simulated Annealing | 4 | 20 | 2 | 4 | 3 | X | 5/6 | 1.13s |
| Tabu Search | 1 | 104 | 33 | 13 | 3 | X | 5/6 | 1.95s |
| Monte Carlo | 41 | X | X | X | X | X | 1/6 | 4.38s |
| Particle Swarm Optimization | X | X | X | X | X | X | **0/6** | 5.02s |
| **合計** | | | | | | | **21/36 ＝ 58.3%** | 最慢 **5.04s** |

**重跑的效果很不平均**：GA 幾乎都在個位數次內解掉，Tabu Search 在 p02 花了 104 次。
重跑買到的是「多擲幾次骰子」，不是「想得更久」——每次重跑都是完全獨立的一次隨機重啟。

## 4. 沒人解得出的那一題，和一個反直覺的發現

**`puzzle_06` 六種全滅**（raw 0/30、served 0/6）：7×7、49 格、**21 個數字**、**0 道牆**。

反直覺的是 **`puzzle_04` 同樣是 7×7／49 格，卻有四種解得出來**——它有 **14 道牆**。
**牆讓題目變簡單**：牆砍掉合法後繼步，隨機走法更不容易走進死路。
所以「盤面大小」不是難度的好指標，**「數字密度高 ＋ 牆少」才是**。

## 5. 決定一：預算固定在 registry，不開成 API 參數

舊 roadmap 的 done 條件寫的是「`SolverRequest` 加 `attempts` 參數」。**這條刻意不做。**

**理由是六種的預算單位根本不可共量**：`attempts`（隨機走法數）、`num_iterations`（迭代）、
`num_generations` × `population_size`（世代 × 族群）、`cooling_rate`（溫度排程）。
開一個 `attempts` 欄位只有 Monte Carlo 吃得到；要正確對應就得開六組不同欄位，
而**六組不同的旋鈕沒辦法拿來比較**——上線的目的正是「同尺度比較」。

**秒是它們唯一的共同單位**，所以預算用秒、固定在伺服器端。
代價講清楚：呼叫端**不能**為難題加預算。要加的話正確做法是加一個對六種都有意義的
`budget_seconds` 欄位（並設上限），不是 `attempts`。

## 6. 決定二：同步就夠，不需要背景任務

roadmap 的開放問題「逾時要怎麼處理（同步阻塞 vs 背景任務）」：**同步**。
單次最慢 0.128s、整個請求最壞 5.04s，而 Gradio 那端的 `timeout=120`。
5 秒 ≪ 120 秒，加背景任務只會多一個狀態機要維護。

⚠ 這個結論綁在 **5 秒**這個值上。若哪天把預算調到分鐘級，這題要重新回答。

## 7. 決定三：驗證用獨立的 `verify.py`，不用 fitness score

`src/core/solvers/verify.py` 重新實作了判定，**沒有**沿用 `calculate_fitness_score()`。

**因為啟發式正在最佳化那個分數，不能讓它自己當裁判**——一條靠著分數漏洞拿高分的路，
會被同一個漏洞認證為解。實際存在的兩個漏洞：
第一步之後不再檢查 blocked cell；負座標會從網格另一側繞回去索引而不是報錯。

`verify.py` 採用的規則是精確 solver 的停止條件（`dfs.py` 的 base case，`rl_env_v2._is_solved` 也是同一套）：
每個開放格恰好走一次、每步走到相鄰格且不穿牆、從 1 出發、數字依序收集。
和那兩者一樣，**不要求路徑結束在最後一個數字上**。

## 8. 對使用者可見的連帶修正：`solvable` 變三態

截圖端點原本是「solver 找不到解 ⇒ 判定圖讀錯了」。
這個推論**只對精確 solver 成立**：每張真實 Zip 盤面都是從一條完整路徑生成的，
所以精確 solver 說無解，就證明讀錯了。但**啟發式或 RL 放棄，什麼都沒證明**。

所以 `VisionSolveResponse.solvable` 從 `bool` 改成 `bool | None`：

| 情況 | `solvable` | 使用者看到 |
|---|---|---|
| 解出來了 | `True` | 答案 |
| **精確** solver 說無解 | `False` | 「至少一道牆或一個數字讀錯了」 |
| 非精確 solver 放棄 | **`None`** | 「未判定，這不代表圖讀錯，換 CP-SAT 可以確定」 |

request log 裡記的也是三態，所以事後分析不會把「啟發式放棄」誤算成解析失敗率。

---

## 9. 誠實的結論

**上線它們是為了比較，不是為了效能。** CP-SAT 在這六題上每一題都是毫秒級的確定答案；
最好的啟發式（GA）要 1.10 秒而且有一題解不出來。這個落差本身就是這條 track 的產出：
**在這種尺寸的組合最佳化問題上，通用的隨機搜尋輸給專用的約束求解器，而且不是小輸。**

現在十種都在同一個 registry、同一個端點、同一份測資下可以互相比較，
這件事的價值不在於多了六個選項，而在於**那個落差第一次可以被量出來**。
