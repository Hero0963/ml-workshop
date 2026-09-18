# 掃 BC 的 epoch ＋ ExIt 第一輪（RL Track，2026-09-19）

> 分支 `feat/rl-exit`（從 `a8c3626` = `main` 長出）｜worktree `zip-rl`｜本人當次授權「交給你規劃、直接開工」。
> 所有數字都是 2026-09-19 本機實測，產物路徑附在各表下。🚧 **撰寫中：數字陸續補上。**

---

## 0. 一句話結論

（待補）

---

## 1. 為什麼做這兩件

- 兩個 goal 已經達標（`bc_multi_456_e6`：4×4 best-of-32 0.9953、6×6 0.9465）。
  本人 2026-09-11 定案「**這個專案就是要用 RL 做**」⇒ 目的不是衝分數，而是**量 RL 式的方法到底買到多少、為什麼**。
- 2026-09-12 量到「**訓練越久，策略越尖**」：PPO 微調（更尖）在 best-of-32 輸、BC 早停（更散）在 best-of-32 贏。
  兩個開著的問題剛好沿著這條軸：
  1. **早停的最佳點在哪？**（只量過 6 與 10 兩點）——便宜、配方現成（handover §3）。
  2. **能不能不靠「少訓練」就保住多樣性？**——ExIt：用搜尋找到的**其他合法解**當額外標籤。
     BC 每題只看過一條解；如果題目其實多解，BC 等於在每個岔路口把「其他合法的走法」當成錯誤來懲罰，
     這正是「訓練越久越尖」的一個可能機制。

---

## 2. 掃 epoch

### 2.1 做法：一次訓練拿齊所有點

`train_behaviour_cloning.py` 新增 `--checkpoint-every-epoch`（每個 epoch 存 `model_epoch_<N>.zip`）。
BC 的訓練軌跡對同一個 seed 是決定性的，所以**同一次訓練的第 N 個 epoch 就是「只訓練 N 個 epoch」的模型**。

**決定性複驗（10/10 通過）**：`bc_multi_456_sweep` 十個 epoch 的 loss 與 val 準確率和已發表的 `bc_multi_456` **逐位相同**，
結尾的 deterministic 評估（`eval_test.json` 的 `per_size`）也一模一樣 ⇒ sweep 的 e10 **就是** `bc_multi_456`。
每個 epoch 存檔不會擾動訓練，這點另外有單元測試釘住
（`test_saving_every_epoch_leaves_the_training_trajectory_alone`：同 seed 兩次訓練、一次中途存檔，loss 與權重逐位相同）。

評分用**同一支** `../hi-collab/scratch/probe_cross_size.py`（只加了選 checkpoint 的旗標，評分迴圈沒動），
6×6 held-out 2,000 題、best-of-32、seed 20260815、GPU。

### 2.2 量尺對照

**必跑的對照**（handover §3）：拿 sweep 的 e6 重跑，必須復現已發表的 `bc_multi_456_e6`。
**結果：全部逐位相同**——det 0.543、bo1 0.4265、bo2 0.595、bo4 0.7375、bo8 0.84、bo16 0.91、**bo32 0.9465**、每題 5.24 次
（`logs/rl_probes/cross_size_bc_multi_456_sweep_e6_6x6_test.json`）⇒ 改過的 probe 沒有換尺。

### 2.3 曲線

（待補）

---

## 3. ExIt 前置檢查：同一題真的有別條解嗎？

**白話**：ExIt 的賣點是「把模型自己找到的**其他**正確走法也教給它」。如果每題其實只有一條解，
模型找到的永遠是資料集那一條，ExIt 什麼都沒加。所以先量。

**做法**：新模組 `src/core/rl/collect_solutions.py`。用服務中的 `bc_multi_456_e6` 在**訓練集**上每題抽樣 32 次
（1,024 局並行、一次前向算完所有局，CPU），把解開的路徑去重，
**每一條都再用專案既有的獨立評分器 `calculate_fitness_score` 驗一次**（它和 env 的 `_is_solved` 不共用任何程式碼）。

**結果**（6×6 訓練集前 500 題，`logs/rl_exit/gate_6x6_train500_k32/summary.json`）：

| 指標 | 值 |
|---|---|
| 至少一次解開 | 0.992 |
| **至少找到一條「≠ 資料集那條」的解** | **0.874** |
| 解開的題平均相異解數 | **5.04** |
| 資料集那條解有出現在抽樣中（解開的題）| 0.909 |
| 別條解總數 | 2,047（平均每題 4.1 條）|
| 被獨立評分器退件 | **0** |

⇒ **「這些題其實多解」從推論變成實測**：過去的證據只是「BC 對標籤只同意 88.11%，卻解開 89.47%」這個落差；
現在直接數到 6×6 訓練題 **87% 有別條解**，而 BC 每題只看過一條。ExIt 有東西可加。

---

## 4. ExIt 第一輪

（待補）

---

## 5. 還沒排除的

（待補）

---

## 6. 資源與踩到的坑

- **GPU 預算是 probe 的並行上限，不是 CPU**：probe 一次只餵一個觀測，GPU 被極小的 kernel 塞滿；
  有空核心時一個 probe 約佔 **40%** GPU 時間（`nvidia-smi pmon`）。5 個 probe ＋ 訓練 = 98%、2 個 probe = 84–88%
  ⇒ **同時只能 1 個 probe**。兩次超標都先停再修，被停掉的 probe 沒有產生任何數字。
- `powershell -File x.ps1 -Epochs 6,2,3` 會把 `6,2,3` 當成一個字串，`[int[]]` 讀成 `623` ⇒ 用 `-Command "& x.ps1 ..."`。

---

## 7. 重現指令

```bash
cd linkedin-zip-challenge
# 掃 epoch：訓練一次、每個 epoch 存檔
uv run python -m src.core.rl.train_behaviour_cloning --goal goal3_multi --run-id bc_multi_456_sweep \
    --epochs 10 --eval-split test --eval-episodes 1 --checkpoint-every-epoch
# 評分某個 epoch（--label 必填，避免覆蓋已發表產物）
PYTHONPATH=. uv run python ../hi-collab/scratch/probe_cross_size.py --run-id bc_multi_456_sweep \
    --checkpoint model_epoch_4 --label bc_multi_456_sweep_e4 --dataset seed20300000_n20000_456 \
    --size 6 --max-attempts 32
# ExIt 前置檢查
uv run python -m src.core.rl.collect_solutions --run-id bc_multi_456_e6 --goal goal3_multi \
    --size 6 --limit 500 --attempts 32 --device cpu --collection-id gate_6x6_train500_k32
```
