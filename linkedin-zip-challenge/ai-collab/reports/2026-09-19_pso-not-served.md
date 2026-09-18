# PSO 不上線：它在 4×4／6×6 能得幾分，以及為什麼留著卻不開放

> 2026-09-19 · Track C（solvers）· 一手量測；腳本與原始數據在 [`artifacts/pso-score/`](artifacts/pso-score/)
> 查證日期 2026-09-19。題目、seed、抽樣的題號全部落盤，可重跑。

---

## 1. 結論

| 盤面 | 單次呼叫（決定性） | 包 5 秒預算（API 當時給的樣子） | 對照：RL best-of-32（同一批測試題） |
|---|---|---|---|
| 4×4 | **0.590**（1,139／1,931） | 待補 | 0.9953 |
| 6×6 | **0.000**（0／2,000） | 待補 | 0.9465 |

**本人定案（2026-09-19）：PSO 不開放，但實作、測試、量測全部保留。**
從 registry 拿掉之後，API、截圖端點、Gradio、Svelte 同時少了它（上線剩九種）；
`test_registry.py::test_pso_is_implemented_but_not_served` 釘住這個決定。

## 2. 為什麼它解不出來

**白話**：PSO 模仿鳥群覓食——每隻鳥記得自己找過最好的位置、也看得到整群最好的位置，然後往那邊飛。
放到 Zip 上，「位置」是一條路徑，「往好的方向飛」被寫成**交換路徑上兩格的順序**。
可是 Zip 的路徑必須一格接一格相鄰：把路線上兩站對調，路線就從地圖一端瞬移到另一端，變成不合法。
所以它幾乎一直在不合法的路徑之間打轉。

**術語**：它的移動算子（swap，`src/core/solvers/particle_swarm_optimization.py` 的 `_apply_velocity`）
**不保持可行性約束**（相鄰）。在 swap 鄰域裡，合法路徑佔的比例隨盤面長大迅速趨近零，
搜尋幾乎全在不可行區域進行。對照組 SA 用「截斷再重長」，每一步都保證路徑連續
（[`2026-09-12_heuristic-solvers-on-the-api.md`](2026-09-12_heuristic-solvers-on-the-api.md) §2）。

**數據上看得到這個機制**：4×4 單次呼叫的解出率隨牆數**上升**——

| 牆數 | 0 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 4×4 單次解出率 | 0.449 | 0.517 | 0.648 | 0.790 | 0.914 |

牆砍掉合法的後繼步，路徑的選擇變少，隨機產生的初始路徑就更容易剛好合法。
這和 2026-09-12 在六題標準盤面上看到的「牆少反而難」是同一件事。

## 3. 怎麼量

- **題目**：RL 資料包 `seed20300000_n20000_456` 的 **test split**，4×4 1,931 題、6×6 2,000 題。
  RL 的 best-of-32 用的是同一個 split（評分腳本預設 `--split test`），所以兩邊可以直接對照。
  資料包只在跑過 RL 的 worktree 裡（不進版控），這次原地唯讀。
- **單次呼叫（`raw`）**：每題 `random.seed(20260919 + 題號)` 後呼叫一次預設參數的 `solve_puzzle_pso`，
  用 `verify.is_solution` 判定。決定性，不受機器忙碌影響。全部題目，單一 process，137.8 秒。
- **包 5 秒預算（`served`）**：用 API 當時服務它的同一個包裝 `registry._until_verified`（重跑到通過驗證、5 秒牆鐘）。
  每個尺寸用 `random.Random(20260919)` 抽 200 題，8 個 worker。
  ⚠ 預算是牆鐘，機器被別的工作佔用時分數會被壓低——**只能在 `.agent-heavy-job` 是 `free` 時跑**。

```powershell
cd linkedin-zip-challenge
PYTHONPATH=. uv run python ai-collab/reports/artifacts/pso-score/measure_pso.py raw <dataset 目錄>
PYTHONPATH=. uv run python ai-collab/reports/artifacts/pso-score/measure_pso.py served <dataset 目錄>
```

## 4. 誠實的限制

- 單次呼叫的 4×4 分數是**一次呼叫**的機率，不是 PSO 的上限；API 給的是包了預算的版本（§1 第三欄）。
- 6×6 單次 0／2,000 不等於「機率是零」：單次成功率的 95% 信賴上界約 **0.0015**（1 − 0.05^(1/2000)）。
- 抽樣 200 題的 `served` 分數有抽樣誤差（二項分布，p≈0.5 時標準誤約 ±0.035）。
