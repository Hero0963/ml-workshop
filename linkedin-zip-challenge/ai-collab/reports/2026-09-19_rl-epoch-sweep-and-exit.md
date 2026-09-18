# 掃 BC 的 epoch ＋ ExIt 第一輪 ＋ 判定器改嚴格（RL Track，2026-09-19）

> 分支 `feat/rl-exit`（從 `a8c3626` 長出，中途 merge 了 `main` 的 Track C）｜worktree `zip-rl`｜本人當次授權「交給你規劃、直接開工」。
> 所有數字都是 2026-09-19 本機實測，產物路徑附在各表下。**本人要求收尾時實驗做一半**——沒做完的部分在 §6，
> 指令與成本在 [`handover-rl-solver.md`](../handover-rl-solver.md) §0.3。

---

## 0. 一句話結論

| 問題 | 答案 | 可信度 |
|---|---|---|
| BC 該訓練幾個 epoch？ | **e2–e5 是一段平台**（6×6 best-of-32 0.9595–0.967，彼此差距在 ±0.01 評估雜訊內），e6 開始下滑（0.9465）、e10 0.8535 | 寬鬆尺、單 seed；平台內**分不出贏家** |
| 這些題真的多解嗎？ | **是**：訓練題有別條解的比例 4×4 54%／5×5 75%／**6×6 87.5%**（嚴格規則、下界）| 實測；出題器不檢查唯一解 |
| ExIt 第一輪有用嗎？ | **單次嘗試大贏**：同為 10 epochs，6×6 deterministic **0.6455 vs 0.5205（+0.125）**、5×5 +0.095、4×4 +0.015 | **單 seed、寬鬆尺**；+0.125 遠大於任何量過的雜訊，但仍待確證 |
| ExIt 在判定基準 best-of-32 上贏嗎？ | 嚴格尺 e4：**0.9755 vs 0.9605（+0.015）**、det **+0.059**；PPO 微調是「det 升、bo32 降」，ExIt 兩頭都沒輸 | 只量完 e4 一對、單 seed ⇒ 有動但未確證 |
| 判定器有問題嗎？ | 有一個全專案的寬鬆點：**7 個判定器都不檢查終點**。已依本人定案全部改嚴格 | 實測影響約 −0.005（bo32）、−0.027（det）|

---

## 1. 為什麼做這兩件

- 兩個 goal 早已達標（`bc_multi_456_e6`：4×4 best-of-32 0.9953、6×6 0.9465，寬鬆尺）。
  本人 2026-09-11 定案「**這個專案就是要用 RL 做**」⇒ 目的不是衝分數，而是**量 RL 式的方法到底買到多少、為什麼**。
- 2026-09-12 量到「**訓練越久，策略越尖**」：PPO 微調（更尖）在 best-of-32 輸、BC 早停（更散）在 best-of-32 贏。兩個開著的問題剛好沿著這條軸：
  1. **早停的最佳點在哪？**（只量過 6 與 10 兩點）——便宜、配方現成。
  2. **能不能不靠「少訓練」就保住多樣性？**——ExIt：用搜尋找到的**其他合法解**當額外標籤。
     BC 每題只看過一條解；如果題目其實多解，BC 等於在每個岔路口把「其他合法的走法」當成錯誤來懲罰，
     這正是「訓練越久越尖」的一個可能機制。

**ExIt 是什麼**（本人問過）：Expert Iteration，Anthony、Tian、Barber 2017〈Thinking Fast and Slow with Deep Learning and Tree Search〉
（[arXiv:1705.08439](https://huggingface.co/papers/1705.08439)）。白話是師徒制：徒弟（策略網路）靠直覺走、師父（搜尋＋驗證）慢但準，
把師父找到的解拿回來教徒弟，徒弟變強後師父的起點也更好。論文用 MCTS 當師父；這裡用「每題抽 32 次＋判定器」。
LLM 圈的 STaR／rejection-sampling fine-tuning 是同一族。

---

## 2. 掃 epoch

### 2.1 做法：一次訓練拿齊所有點

`train_behaviour_cloning.py` 新增 `--checkpoint-every-epoch`（每個 epoch 存 `model_epoch_<N>.zip`）。
BC 的訓練軌跡對同一個 seed 是決定性的，所以**同一次訓練的第 N 個 epoch 就是「只訓練 N 個 epoch」的模型**。

**決定性複驗（10/10 通過）**：`bc_multi_456_sweep` 十個 epoch 的 loss 與 val 準確率和已發表的 `bc_multi_456` **逐位相同**，
結尾的 deterministic 評估（`eval_test.json` 的 `per_size`）也一模一樣 ⇒ sweep 的 e10 **就是** `bc_multi_456`。
存檔不會擾動訓練，另有單元測試釘住（`test_saving_every_epoch_leaves_the_training_trajectory_alone`）。

評分用**同一支** `../hi-collab/scratch/probe_cross_size.py`（只加了選 checkpoint 的旗標，評分迴圈沒動），
6×6 held-out 2,000 題、best-of-32、seed 20260815、GPU。

### 2.2 量尺對照

**必跑的對照**：拿 sweep 的 e6 重跑，必須復現已發表的 `bc_multi_456_e6`。
**結果：全部逐位相同**——det 0.543、bo1 0.4265、bo2 0.595、bo4 0.7375、bo8 0.84、bo16 0.91、**bo32 0.9465**、每題 5.24 次
（`logs/rl_probes/cross_size_bc_multi_456_sweep_e6_6x6_test.json`）⇒ 改過的 probe 沒有換尺。

### 2.3 曲線（寬鬆尺，6×6 held-out 2,000 題）

| epoch | det | bo1 | bo4 | bo16 | **bo32** | 次／題 @32 | val 選擇準確率 |
|---|---|---|---|---|---|---|---|
| 1 | 0.3860 | 0.1655 | 0.4545 | 0.7870 | 0.8840 | 9.97 | 0.8698 |
| 2 | 0.5440 | 0.2905 | 0.6445 | 0.9165 | **0.9665** | 5.73 | 0.8856 |
| 3 | 0.5600 | 0.3330 | 0.6995 | 0.9185 | 0.9605 | 5.39 | 0.8884 |
| 4 | 0.5760 | 0.3755 | 0.7350 | **0.9350** | **0.9670** | 4.82 | 0.8798 |
| 5 | **0.5915** | 0.4270 | **0.7585** | 0.9315 | 0.9595 | **4.64** | 0.8984 |
| 6 | 0.5430 | 0.4265 | 0.7375 | 0.9100 | 0.9465 | 5.24 | 0.8999 |
| 10 | 0.5205 | **0.4810** | 0.6935 | 0.8075 | 0.8535 | 7.76 | 0.8891 |

產物：`logs/rl_probes/cross_size_bc_multi_456_sweep_e{1..6}_6x6_test.json`；e10 = 已發表的 `cross_size_bc_multi_456_6x6_test.json`。

**讀法**：

- **e2–e5 是平台**，相鄰差距 <0.01（評估雜訊約 ±0.01）⇒ 依 handover 的判讀規則「**這段是平台，分不出來**」，不硬選贏家。
  若要挑一個：e5 的 det 最高、每題嘗試最少（最便宜），e4 的 bo32 最高，差距都在雜訊內。
- **bo1 隨 epoch 單調上升、bo32 在 e5 之後下滑** ⇒「訓練越久越尖」第三次被量到（前兩次是 PPO 微調與 6 vs 10）。
- **val 選擇準確率不能拿來挑 epoch**：它在 e4 反而是低點（0.8798），而 e4 是 bo32 最高的點。

---

## 3. ExIt 前置檢查：同一題真的有別條解嗎？

**白話**：ExIt 的賣點是「把模型自己找到的**其他**正確走法也教給它」。如果每題其實只有一條解，
模型找到的永遠是資料集那一條，ExIt 什麼都沒加。所以先量。

**做法**：新模組 `src/core/rl/collect_solutions.py`。用服務中的 `bc_multi_456_e6` 在**訓練集**上每題抽樣 32 次
（1,024 局並行、一次前向算完所有局，CPU），把解開的路徑去重，
**每一條都再用 `calculate_fitness_score` 獨立驗一次**（它和 env 的 `_is_solved` 不共用任何程式碼）。

**結果**（訓練集全部 47,435 題 × 32 次，1,927 秒；`logs/rl_exit/exit_r1_e6_k32/summary.json`）：

| 盤面 | 題數 | 有別條解（寬鬆）| **有別條解（嚴格）** | 解開的題平均相異解數 | 資料集那條有被抽到 | 獨立檢查器退件 |
|---|---|---|---|---|---|---|
| 4×4 | 15,439 | 55.8% | **54.1%** | 2.37 | 99.5% | 0 |
| 5×5 | 15,996 | 76.4% | **75.1%** | 4.06 | 96.2% | 0 |
| 6×6 | 16,000 | 88.2% | **87.5%** | 5.34 | 89.6% | 0 |

- 「嚴格」＝只算停在最大數字的解（§5）。這是**下界**：只數得到模型自己抽得到的解。
- **多解不是「題目出得好」**：出題器是「先隨機畫一條哈密頓路、再挑 1/4–1/3 的格子標數字、隨機加牆」，
  **完全不檢查唯一解**（`puzzle_generator.py:74-125`）。真題是否唯一解沒查證——**本人定案不在意**：只要在自家題上表現好。
- 過去「多解」只是推論（BC 對標籤只同意 88.11%，卻解開 89.47%）；**現在是直接數到的**。

---

## 4. ExIt 第一輪

### 4.1 設計：只改一個變數

| | sweep（對照）| ExIt（實驗）|
|---|---|---|
| seed、打亂順序、epoch 數、每 epoch 步數 | 20260815、同、10、同 | **全部相同** |
| 每題的標籤 | 資料集那一條 | 每個 epoch 從「資料集那條 ＋ 收集到的嚴格別條解」**均勻抽一條** |

- 選標籤用**獨立的** `random.Random(seed)`，全域 `random`（負責打亂）的抽取和 sweep 完全一樣。
- 同一題的所有解**長度相同**（都走滿所有格）⇒ 每個 epoch 的梯度步數也相同。
- **均勻抽「相異的解」而不是照模型找到的頻率抽**：否則等於把收集者自己的偏好餵回去，又會變尖。
- 收集者是服務中的 `bc_multi_456_e6`（一輪就是「服務中的搜尋當師父」）。135,818 條別條解、34,364 題（`solutions_strict.json`）。

### 4.2 機制先看到了：loss 有地板

| epoch | 1 | 2 | 4 | 6 | 8 | 10 |
|---|---|---|---|---|---|---|
| BC 的訓練 loss | 0.1651 | 0.1067 | 0.0798 | 0.0618 | 0.0454 | **0.0337** |
| ExIt 的訓練 loss | 0.1651 | 0.1073 | 0.0861 | 0.0783 | 0.0738 | **0.0709** |

**白話**：同一個局面有好幾個合法答案時，最好的預測就是把機率分給它們——loss 有一個**降不下去的地板**
（術語：標籤分布的條件熵）。BC 可以靠「押死一條路」把 loss 一直壓低，那正是它變尖的方式；ExIt 被地板擋住了。

### 4.3 結果

**deterministic，三個尺寸，10 epochs**（寬鬆尺，訓練腳本結尾的評估；`logs/rl_a2/<run>/eval_test.json`）：

| | 4×4 | 5×5 | 6×6 |
|---|---|---|---|
| BC（`bc_multi_456` = sweep e10）| 0.9404 | 0.7496 | 0.5205 |
| **ExIt（`exit_r1_e6k32` e10）** | **0.9555** | **0.8451** | **0.6455** |
| 差 | +0.015 | **+0.095** | **+0.125** |

6×6 的 0.6455 **比 BC 任何一個 epoch 都高**（BC 最高是 e5 的 0.5915）。

**best-of-32，嚴格尺，e4 對照**（`logs/rl_probes/cross_size_strict_*_e4_6x6_test.json`）：

| 6×6 嚴格尺 | det | bo1 | bo4 | bo16 | bo32 | 次／題 @32 |
|---|---|---|---|---|---|---|
| BC e4 | 0.5495 | 0.3800 | 0.7225 | 0.9245 | 0.9605 | 4.97 |
| **ExIt e4** | **0.6085** | 0.3775 | **0.7485** | **0.9370** | **0.9755** | **4.57** |
| 差 | **+0.0590** | −0.0025 | +0.0260 | +0.0125 | +0.0150 | −0.40 |

⇒ **單次嘗試贏得最明顯（det +0.059）**；best-of-32 +0.015、每題少試 0.4 次，方向對但**小於 0.02，在雜訊邊緣、單 seed ⇒ 有動但未確證**。
和 PPO 微調（det 升、bo32 **降**）不同，ExIt **兩頭都沒輸**——這正是「保住多樣性」該長的樣子。

### 4.4 還沒分開的兩個機制

ExIt 的標籤變化同時做了兩件事：
(a) **岔路口多標籤**——同一個局面的目標分布分散到所有合法走法；
(b) **新局面**——別條解會走過資料集那條解**從沒經過的局面**，等於更多樣的訓練狀態（像沒付專家成本的 DAgger）。
BC 在 6×6 嚴重過擬合（訓練集 0.9005 vs 測試集 0.5200），所以 (b) 很可能貢獻不小。
**要分開得另設一臂**（例如只在資料集路徑上的局面用多標籤）。

---

## 5. 判定器改嚴格

**本人問「判定器有問題嗎」⇒ 查出一個全專案的寬鬆點。**
env 的 `_is_solved`、`dfs.py`、A* 兩版、`cp.py`、`calculate_fitness_score`、`solvers/verify.py`（API 的判定器）、
VLM 的 `path_is_legal`——**7 個判定器都不檢查終點**，「收完最大數字還繼續走完剩下格子」也算解開。

**規則本身是模糊的**：LinkedIn 官方說明只寫「填滿所有格子、依序經過數字、不穿牆」
（[LinkedIn Help](https://www.linkedin.com/help/linkedin/answer/a7445030)），**沒說**終點必須是最大數字；
第三方教學站寫「停在最大數字」（[zipgameunlimited](https://www.zipgameunlimited.com/how-to-play)）；
本專案出題器**永遠**把最大數字放在路的最後一格（資料集 0 例外）。搜尋關鍵字：`LinkedIn Zip puzzle rules path must end on highest number fill every cell`。

**影響（改之前量的）**：收集到的別條解有 4×4 5.5%／5×5 4.0%／6×6 3.6% 停在別處；
`bc_multi_456_e6` 在 6×6 測試集 best-of-32 寬鬆 0.948 → 嚴格 0.9435（用收集器的抽樣量，`logs/rl_exit/endpoints_exit_r1_e6_k32.json`）；
sweep e4 用 probe 量：bo32 0.967 → 0.9605、**det 0.576 → 0.5495**（det 只有一條路，停錯格就沒第二次機會，所以掉得比 bo32 多）。

**處理**（本人定案「7 處全改」）：七處改成同一條規則。判定器**刻意各自獨立**（`verify.py` 存在就是為了不讓啟發式被自己優化的分數評分），
所以沒有抽共用函式，改在 `src/core/tests/test_end_on_last_number.py` **一次釘住七個**。
用的 2×2 盤面上 DFS 先試「往右」，**舊版 DFS 真的會回傳那條走過頭的路、舊版 `verify` 也接受它**（用 git 裡的舊檔實跑確認）⇒ 測試分得出兩種規則。
全體 331 passed、8 xfailed ⇒ 沒有既有測試依賴寬鬆規則。
⚠ **VLM 的端到端評估沒有重跑**（預期不變：CP-SAT 現在只回傳停在最大數字的路，真實盤面的標準答案也停在那裡）。

---

## 6. 還沒做／還沒排除的

1. **嚴格尺下的 BC vs ExIt 完整對照**只量完 e4（§4.3）。e2／e5／e6／e10 兩臂都要補。
2. **seed 雜訊**：BC 與 ExIt 各多 2 個 seed。**6×6 的 BC seed 雜訊至今沒量過**。收尾時中途停掉，不完整的目錄已軟刪除。
3. ExIt 的兩個機制沒分開（§4.4）。
4. ExIt 只跑了**一輪**；真正的「迭代」是用這一輪的贏家再收集、再訓練。
5. 服務模型沒換（仍是 `bc_multi_456_e6`）——要不要換由本人決定，而且要先補 4×4／5×5 的嚴格 best-of-32。

---

## 7. 資源與踩到的坑

- **GPU 預算限制的是 probe 並行數，不是 CPU**：probe 一次只餵一個觀測，GPU 被極小的 kernel 塞滿；
  有空核心時一個 probe 約佔 **40%** GPU 時間（`nvidia-smi pmon`）。5 個 probe ＋ 訓練 = 98%、2 個 probe = 84–88%
  ⇒ **同時只能 1 個 probe**。兩次超標都先停再修，被停掉的 probe 沒有產生任何數字。
  同一件事的 GPU 佔用會隨 CPU 爭用變動，所以預算要在「實際要跑的組合」下量。
- `powershell -File x.ps1 -Epochs 6,2,3` 會把 `6,2,3` 當成一個字串 ⇒ 用 `-Command "& x.ps1 ..."`。
- 停掉背景的「依序訓練」迴圈時，只殺子行程會讓迴圈直接開下一個 ⇒ **先殺父 shell**（用子行程的 `ParentProcessId` 找）。

---

## 8. 重現指令

```bash
cd linkedin-zip-challenge
# 掃 epoch：訓練一次、每個 epoch 存檔
uv run python -m src.core.rl.train_behaviour_cloning --goal goal3_multi --run-id bc_multi_456_sweep \
    --epochs 10 --eval-split test --eval-episodes 1 --checkpoint-every-epoch
# 評分某個 epoch（--label 必填，避免覆蓋已發表產物）
PYTHONPATH=. uv run python ../hi-collab/scratch/probe_cross_size.py --run-id bc_multi_456_sweep \
    --checkpoint model_epoch_4 --label strict_bc_multi_456_sweep_e4 --dataset seed20300000_n20000_456 \
    --size 6 --max-attempts 32
# ExIt 收集（全量約 32 分鐘，CPU）
uv run python -m src.core.rl.collect_solutions --run-id bc_multi_456_e6 --goal goal3_multi \
    --attempts 32 --device cpu --collection-id exit_r1_e6_k32
# ExIt 訓練（solutions_strict.json 是判定器改嚴格前收集、再過濾的；判定器改嚴格後新收集的 solutions.json 已經是嚴格的）
uv run python -m src.core.rl.train_behaviour_cloning --goal goal3_multi --run-id exit_r1_e6k32 \
    --epochs 10 --eval-split test --eval-episodes 1 --checkpoint-every-epoch \
    --extra-solutions logs/rl_exit/exit_r1_e6_k32/solutions_strict.json
# 彙整
uv run python ../hi-collab/scratch/summarise_probes.py strict_bc_multi_456_sweep_e4 strict_exit_r1_e6k32_e4
```

⚠ 評分與彙整腳本在 `hi-collab/scratch/`（不進版控）——沿用它是因為已發表的基準就是它產的，換一支等於換尺。
