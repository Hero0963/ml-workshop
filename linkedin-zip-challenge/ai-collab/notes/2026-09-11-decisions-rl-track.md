# 決策紀錄 — 2026-09-11：RL track 的收尾方向

> **這條 track 在什麼時間點、根據什麼證據、做了哪些決定。**
> 給的是「為什麼是這樣」而不是「做了什麼」——後者在 [`../dev_log.md`](../dev_log.md)。
> 概念解釋在 [`01-rl-methods-explained.md`](01-rl-methods-explained.md)、
> [`02-reading-the-numbers.md`](02-reading-the-numbers.md)、
> [`03-inference-and-serving.md`](03-inference-and-serving.md)。
> 分支 `feat/rl-a2-training`

---

## 脈絡

從「接手 handover、對齊 goal」開始，變成一次**收尾盤點**：把所有還開著的問題一次量掉，
再決定這條 track 要怎麼結束（或繼續）。

---

## 一、支撐這些決定的量測（都是實跑，不是引用）

| # | 問題 | 答案 | 成本 |
|---|---|---|---|
| 1 | 環境還是好的嗎 | `261 passed, 8 xfailed`、`ruff check` 綠，與 handover 記錄一致 | 21s |
| 2 | **best-of-32／64 能不能過 0.85** | **N=32 ＝ 0.8500（剛好踩線）、N=64 ＝ 0.8835** | 665s |
| 3 | **5×5 能解嗎** | bc_6x6 借過去 **0.2941**（greedy 0.0176）⇒ **跨尺寸泛化真的存在** | 39s |
| 4 | **4×4／6×6 是同一個模型嗎** | 不是，兩個模型。交叉矩陣顯示泛化**單向**：大盤面往下可以，小盤面往上不行 | 120s |
| 5 | ruff 版本不一致 | pre-commit 釘 v0.4.8、pyproject 釘 0.14.1 ⇒ 統一到 0.14.1 並排除 notebook | — |

**交叉尺寸矩陣（deterministic）**：

| 模型 ＼ 盤面 | 4×4 | 5×5 | 6×6 |
|---|---|---|---|
| bc_4x4 | **0.8947** | 0.0706 | 0.0105 |
| bc_6x6 | 0.5456 | **0.2941** | **0.4620** |
| greedy | 0.1170 | 0.0176 | 0.0041 |

---

## 二、兩個被當場抓到的判讀錯誤（自己的）

1. **「best-of-16 ＝ 16 倍算力」是錯的**。因為第一次成功就停，**實際是 5.35 倍**。
   本人直接指出「best-of-N 現在 N=16 喔」。
2. **外推又錯了一次**。事前用 hazard 0.035 預測 best-of-32 ≈ 0.887，實際 **0.8500**。
   這是這條 track 第三次外推失準 ⇒ 規則升級成「**hazard 表也只能用來排除，不能用來預測**」。

還有一個**沒有成為錯誤但值得記**的：ruff 版本提升後，第一次 `pre-commit run --all-files`
改寫了 29 本教材 notebook（ruff 從 0.6 起預設處理 `.ipynb`）。已還原並用 `types_or` 永久排除。
**教訓：版本提升不是無副作用的，跨子專案的工具改動要先跑一次看它碰了什麼。**

---

## 三、被問到、而答案值得留下來的問題

| 問題 | 去哪看完整答案 |
|---|---|
| BC 是什麼 | [`../01-rl-methods-explained.md`](01-rl-methods-explained.md) §1 |
| 「只訓練 9 分鐘」是在訓練什麼 | 同上 §2 |
| DAgger 是什麼 | 同上 §3 |
| BC → PPO 微調是什麼、為什麼能推翻「這題不該用 RL」 | 同上 §4 |
| AlphaZero 式的「**自我改進迴圈**」是什麼意思 | 同上 §5（含窮人版 ExIt）|
| 「比較訓練方法只能用 deterministic」是什麼意思 | [`../02-reading-the-numbers.md`](02-reading-the-numbers.md) §2 |
| 「4×4 達標」是能從頭走到尾嗎 | 同上 §1（是，沒有部分分數）|
| 6×6 還差什麼、是資料量嗎 | 同上 §5（不是資料量，是單步錯誤率要砍 5.4 倍）|
| 加大 N 有沒有幫助、N=32 會不會「突然學會」 | 同上 §4（不會學會，只會碰到；hazard 沒收斂到零）|
| 實務上怎麼推理 | [`../03-inference-and-serving.md`](03-inference-and-serving.md) |
| 訓練時看得到即時 log 嗎 | 本檔 §五 |

---

## 四、定下來的決定

| 決定 | 內容 | 理由 |
|---|---|---|
| **判定基準用 best-of-32** | 不在 N=32 vs 64 上著墨 | 本人：「這是練習用專案」。⚠ 但 0.8500 剛好等於門檻、評估雜訊 ±0.01 ⇒ 報告要標「踩線」 |
| **多尺寸模型：做** | 一份 4／5／6 的資料集訓一個模型 | 跨尺寸泛化已證實存在；「一個模型打天下」還沒被測過 |
| **單尺寸對照組：做** | 否則只有數字沒有結論 | — |
| **★ 就是要用 RL 做** | 本人明確表示知道有別的 solver，但這個專案就是要用 RL | ⇒ BC→PPO 微調從「選項」升級成收尾後的主線；排序見 `../01-rl-methods-explained.md` §6 |
| **ruff 排版全上** | 21 個檔的純排版改動全部保留，含跨子專案的 3 個 | 已 commit `fccddc2` |
| **討論要整理成 notes** | 建立 `ai-collab/notes/`，把對話沉澱成可重讀的筆記 | 做中學的產出是理解；只留在對話裡等於沒有 |

---

## 五、訓練時看得到什麼（讀碼確認）

- **BC**：每個 epoch 結束**立刻** append 一行到 `logs/rl_a2/<run-id>/bc_progress.jsonl`
  （`train_behaviour_cloning.py:329`），同時 loguru 印到 stdout。
  欄位：`policy_loss`／`value_loss`／`gradient_steps`／`val_choice_accuracy`／`val_action_accuracy`／`seconds`。
  **粒度是 epoch**（4×4 約 10 秒、6×6 約 23 秒一筆），不是每個 batch。可以 `Get-Content -Wait` 追。
- **PPO**：TensorBoard 在 `logs/rl_a2/<run-id>/tensorboard/`，另有 curriculum 的 jsonl。
- ⚠ **評估探針目前只有跑完才輸出**，中途沒有進度——665 秒的等待就是這樣來的。**待改**。
- ⚠ **BC 沒有中途 checkpoint**，只存 `model_final`。

---

## 六、留給下一步的

計畫在 [`../plans/2026-09-11_wrapup-docker-and-multisize.md`](../plans/2026-09-11_wrapup-docker-and-multisize.md)，
四條線：多尺寸模型、Docker 一鍵啟動、A5 把 RL solver 掛進 API、文件。
收尾之後的主線是 **BC → PPO 微調**。
