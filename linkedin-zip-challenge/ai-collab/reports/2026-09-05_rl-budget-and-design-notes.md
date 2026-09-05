# RL 續訓 — 加預算的結果，以及這套設計到底在學什麼

> 2026-09-05（Asia/Taipei）｜分支 `feat/rl-a2-training`｜worktree `zip-rl`
> 一句話：**續訓 3M 步讓 curriculum 從 k=27 推到 30、held-out 從 0.344 升到 0.409 ⇒「加預算有效」成立；但每級晉級成本約以 ×2 成長，推到全長是小時級的事。**
> 原始資料：`logs/rl_a2/goal2_6x6_bigdata/`（`train.log`／`progress.jsonl`／`eval_test.json`）
> 前一輪：[`../handover-rl-solver.md`](../handover-rl-solver.md) §0｜設計來源：[`2026-08-15_rl-restart-plan.html`](2026-08-15_rl-restart-plan.html)

---

## 0. 這份報告為什麼有第 5 節

restart plan §8 早就寫過本專案的定位：RL **不是為了比 CP-SAT 快或準**（不會贏），
是為了學 masking／PPO／curriculum／potential shaping／MCTS-AlphaZero 這一整套。
**2026-09-05 本人再次明確定案：這個 side project 的產出是「做中學」，不是單純的指標數字。**

所以本報告除了記錄這一輪量到什麼，第 5 節把當天對話中問到的設計問題
（為什麼顯存這麼少、8 個 channel 是什麼、為什麼選 MaskablePPO、和圍棋 AI 什麼關係）一併落盤。
**那些理解與數字同等是產出**——不寫下來就等於沒發生過，下一個 session 得重新問一次。

---

## 1. 做了什麼

接續 2026-08-29 那輪 `goal2_6x6_bigdata`（停在 5,005,312 步、k=27/36、held-out 0.344），
依 handover §6 指定的順序做第一件事：**只加訓練預算，其他一律不動**。

```powershell
uv run python -m src.core.rl.train_maskable_ppo --goal goal2_6x6 --run-id goal2_6x6_bigdata `
  --resume --timesteps 3000000 --eval-split test --eval-episodes 20
```

**開跑前先備份**（`--resume` 會覆寫同名檔，上一輪的證據不能被蓋掉）：

| 備份 | 內容 |
|---|---|
| `models/rl_a2/goal2_6x6_bigdata/checkpoints/model_final_at5005312.zip` | 上一輪最終權重（0.344 那個） |
| `logs/rl_a2/goal2_6x6_bigdata/train_state_at5005312.json` | 上一輪 curriculum 狀態 |
| `logs/rl_a2/goal2_6x6_bigdata/eval_test_at5005312.json` | 上一輪逐題評估結果 |

`progress.jsonl` 是 append 模式，曲線歷史自動保留，不必備份。
`checkpoints/model_5000000.zip` 也仍在（與被覆寫的 `model_final.zip` 只差 5,312 步）。

**續訓確實帶回了 curriculum**（handover 陷阱 #10 說這裡會靜默失敗，所以要看實際 log 而不是相信程式碼）：

```
Resuming goal2_6x6_bigdata from model_final.zip at 5005312 steps, k=27
Curriculum starts at k=27
```

---

## 2. 結果

```
Trained 3,000,000 steps in 749.5s (4002 fps), k=30, promotions=9
model          solve=0.409  dead_end=0.591  coverage=0.678  (n=2000)
masked_random  solve=0.000  dead_end=1.000  coverage=0.318  (n=40000)
greedy         solve=0.004  dead_end=0.996  coverage=0.588  (n=40000)
Done condition (solve >= 85% at full length): not met
```

| | 上一輪（5M） | 本輪（8M） | 變化 |
|---|---|---|---|
| held-out test solve | 0.344 | **0.409** | **+0.065** |
| dead_end | 0.656 | 0.591 | −0.065 |
| coverage | 0.626 | 0.678 | +0.052 |
| curriculum k | 27/36 | **30/36** | 晉級 1 級 |
| promotions | 8 | 9 | |

**兩個 baseline 完全沒動**（greedy 0.004／random 0.000，與 2026-08-15、2026-08-29 相同）
⇒ 評估協定跨三輪一致，這些數字可以直接比較。

**門檻仍未達成**（0.85）。但這一輪要回答的問題不是「有沒有達標」，是「**加預算這條路走不走得通**」——
答案是走得通，只是比想像貴。

---

## 3. curriculum 的晉級成本：形狀對，但代價在指數上升

### 3.1 k=30 仍在單調上升，不是停滯

`progress.jsonl` 在 k=30 有 298 筆紀錄，五等分後的 `rollout_success_rate` 平均：

| 區段（步數） | k=27 | k=30 |
|---|---|---|
| 第 1 段 | 0.710 | 0.679 |
| 第 2 段 | 0.752 | 0.712 |
| 第 3 段 | 0.774 | 0.731 |
| 第 4 段 | 0.799 | 0.745 |
| 第 5 段 | 0.819 | **0.764** |

（k=27 涵蓋 2,711,552 → 5,570,560 步；k=30 涵蓋 5,578,752 → 8,011,776 步。晉級門檻 0.90。）

**兩條曲線的形狀一模一樣**：單調上升、斜率遞減、被預算截斷。
k=27 那條後來確實晉級了 ⇒ **k=30 這條沒有理由判定為停滯，它只是還沒到。**

> ⚠ 再強調一次（handover 陷阱 #21）：`rollout_success_rate` **不是能力指標**。
> k=30 表示前 6 格已預先走好、題目來自訓練集、動作是隨機取樣——三層有利條件疊在一起。
> 同一個模型從真起點、在 held-out、deterministic 是 **0.409**。
> 這個 0.764 唯一的用途是**判斷 curriculum 該不該晉級**與**加預算有沒有用**。

### 3.2 每級成本約以 ×2 成長

| 晉級 | 達成步數 | 本級花費 | 相對前一級 |
|---|---|---|---|
| k=3→6 | 416 | 416 | — |
| k=6→9 | 4,160 | 3,744 | ×9.0 |
| k=9→12 | 51,536 | 47,376 | ×12.7 |
| k=12→15 | 128,160 | 76,624 | ×1.62 |
| k=15→18 | 297,136 | 168,976 | ×2.21 |
| k=18→21 | 586,832 | 289,696 | ×1.71 |
| k=21→24 | 1,399,776 | 812,944 | ×2.81 |
| k=24→27 | 2,707,056 | 1,307,280 | ×1.61 |
| **k=27→30** | **5,574,832** | **2,867,776** | **×2.19** |
| k=30→33 | 未達成 | 已花 2,436,944 | — |

照最近三級的 ×1.6–2.8 外推：30→33 約需 **4–8M** 步，33→全長再乘一次
⇒ **推到全長還要約 15–30M 步 ≈ 1–2 小時**（本機 4,002 fps）。

> ⚠ **這個外推本身不可信**，理由是 handover §3.8 已經記載的教訓：
> 上一輪用「×1.29」推出「1.3M 步到全長」，**實際差三倍**。
> 這裡寫下區間只是為了讓「要不要繼續」有個量級可談，**不是預測**。

### 3.3 一個沒被推翻的替代解釋

「加預算就會到」目前只被**一次**晉級支持（k=27→30）。還沒排除的是：
**k 越大，題目越接近「從頭走完整盤」，而模型可能有一個學不動的天花板**——
若真如此，成本不是 ×2 成長而是發散，再多預算也不會到 k=None。

要分辨這兩者，最便宜的證據是**下一級（30→33）到底花多少**：
落在 4–8M 內 ⇒ ×2 模型成立；顯著超出 ⇒ 該考慮換路（調超參、加網路容量、或改推論策略）。

---

## 4. GPU 到底吃多少（實測，含對照）

起因是一個很自然的問題：既然在 `cuda` 上跑，為什麼 `train_state.json` 記的 GPU 只有 83.8 MiB？

### 4.1 量到的數字

| 情境 | `utilization.gpu` | `memory.used`（全機） |
|---|---|---|
| 訓練中（取樣 5 次） | 36／37／39／40／40 % | 3,283 MiB |
| **訓練結束後（對照，取樣 5 次）** | 0／1／5／2／1 % | 2,926 MiB |

**對照是必要的**：`nvidia-smi` 同時列出 Chrome、VS Code、Discord、Antigravity 等桌面程式也在 GPU 上，
不扣掉桌面基線就無法把 36–40% 歸因給訓練。扣掉後結論是：**那約 38 個百分點確實來自訓練。**

### 4.2 但「utilization 38%」不等於「算力用了 38%」

`nvidia-smi` 的 `utilization.gpu` 定義是**取樣期間至少有一個 kernel 在執行的時間比例**，
不是 SM 佔用率。117 萬參數的小網路高頻發射小 kernel，就會出現「忙碌時間長、算力佔用低」。

支持這個判讀的證據：**兩輪的 fps 幾乎相同**（上一輪 3,968、本輪 4,002）。
若 GPU 是瓶頸，utilization 會貼近 100% 且 fps 會隨 GPU 負載變動。
handover §3.10 早已量過：整個訓練程序約只吃 1 個核心，瓶頸是**單執行緒的 Python env step**。

### 4.3 顯存為什麼只有 83.8 MiB

照 `GridScalarExtractor` 重建一次網路實測（`batch=512`，重建版 1,130,693 參數，
實際 policy 1,170,949——差額是 SB3 預設的 `net_arch` MLP head）：

| 項目 | 顯存 |
|---|---|
| 權重（fp32） | 4.31 MiB |
| ＋梯度 ＋ Adam 兩個動量 | allocated 30.21 MiB |
| ＋前向中間激活（3 層 × 64ch × 8×8 × 512） | peak **115.58 MiB** |
| 實際 run 記錄 | alloc **83.8 MiB**／reserved 106 MiB |
| 對照：7B LLM 光權重（fp16） | **13.0 GiB**（約 3,000 倍） |

三個原因：

1. **輸入極小**：8×8×8 ＝ 512 個數字。ImageNet 一張圖是 224×224×3 ＝ 15 萬個。
2. **網路極淺窄**：3 層 conv（64 通道）＋ 一層 `Linear(4104→256)`。
   **參數幾乎全在那層 Linear**（約 105 萬，佔 90%），conv 只有約 7.7 萬。
3. **rollout buffer 不在顯卡上**：SB3 的 buffer 是 `np.zeros(...)`
   （`.venv/Lib/site-packages/stable_baselines3/common/buffers.py:392`），存在 CPU，
   只有每個 minibatch 要算時才 `to_torch` 搬上去。8,192 筆 transition 約 17 MB，全在 RAM。

⇒ `ResourceSettings.gpu_memory_fraction=0.75`（12,281.6 MiB）**是防呆，不是為這個網路設的**，
在可預見的未來都不會被逼近。要它有意義，得是網路容量放大兩三個數量級以後的事。

---

## 5. 設計筆記：這套東西和圍棋 AI 的關係

> 這一節是「做中學」的落盤。內容來自 2026-09-05 的討論，
> 外部事實已查證（見文末來源），專案內的決策理由來自 `2026-08-15_rl-restart-plan.html`，**不是推測**。

### 5.1 8 個 channel 就是 CNN，而且就是 AlphaGo 的 feature planes

`GridScalarExtractor` 第一層是 `nn.Conv2d(8, 64, 3, padding=1)`——
**8 是輸入通道數，和彩色照片的 RGB＝3 在同一個位置**。差別只是每個 channel 不是顏色強度，
而是一張關於盤面的布林地圖：

```
valid / wall_right / wall_down / visited / agent / wp_next / wp_future / wp_done
```

白話：把盤面想成 8 張疊在一起的透明片，每張只回答一個問題（「這格能走嗎」「右邊有牆嗎」
「走過了嗎」「我在哪」…）。3×3 卷積核一次看一個小區域、**同時翻閱這 8 張透明片**，
學出「右邊有牆而且下面走過了 ⇒ 快變死路」這種局部模式。

這正是圍棋 AI 的做法，術語叫 **feature planes**：

| | 盤面 | 特徵平面數 |
|---|---|---|
| AlphaGo（2016） | 19×19 | 48（含大量人工圍棋知識） |
| AlphaGo Zero（2017） | 19×19 | 17（幾乎只剩規則的原始表示） |
| **本專案** | 8×8 | **8** |

**輸入端的設計哲學一致，而且性質更接近 Zero**（我們這 8 張沒有人工啟發式，都是規則直接推得的事實）。
輸出端則差很多：圍棋要從 361＋1 個點裡挑，這裡只有 4 個方向。

### 5.2 為什麼選 MaskablePPO（理由不是圍棋，是「遮罩」這個硬需求）

決策理由在 restart plan §5.1 白紙黑字：

| 候選 | 報告的評語 |
|---|---|
| **sb3-contrib** | 「**MaskablePPO 在這裡**」——專案已鎖 SB3 2.7.0 |
| CleanRL | 「造輪子時的最佳參考」 |
| TorchRL／RLlib | 「對 8×8 網格是**殺雞用牛刀**」 |
| JAX 系（Brax／PureJaxRL） | 「要重寫 env 為 JAX，**成本不划算**」 |

**遮罩為什麼是硬需求**：Zip 的規則裡一大堆動作根本不合法（出界、有牆、走過的格子、號碼順序不對）。
沒有 mask，agent 得花好幾百萬步去學「不要撞牆」這種本來就寫在規則裡的事。
Mask 是直接把不合法的選項從選單上劃掉，一步都不用浪費。
**附帶效果**：2025-10 那個「在兩格之間來回跳」的 deterministic policy loop，在一筆畫 ＋ mask 之下
**定義上不可能發生**（走過的格子被遮掉了）——這是把 bug 用設計消掉，而不是調 reward 權重去壓。

### 5.3 圍棋 AI 有被參考，只是排在後面

restart plan §9 的執行計畫表已經寫著：

```
A6（選）  自寫 PPO 對拍 / AlphaZero-lite（MCTS＋雙頭網路）   1 週＋
```

而路線比較表的「綜效」欄寫的是「**`board-game-rl` 的 MCTS/AlphaZero 下一步**」——
同一個 monorepo 裡的另一個子專案本來就規劃了 MCTS/AlphaZero，兩邊是要互相餵的。

**為什麼不一開始就做 AlphaZero**：AlphaZero ＝ MCTS ＋ 雙頭網路 ＋ self-play，
是 PPO 的超集合，複雜度高一個檔次。先用 MaskablePPO 把 env、觀測、curriculum、
評估協定這些地基驗證過，再談要不要加搜尋——這是正確的順序，而 A0 的教訓
（env v1 餵標準答案也不會過關）正說明地基沒驗證就往上疊會發生什麼事。

### 5.4 讀 AlphaGo 論文的建議：讀 Zero 那篇，但現在還不是時候

**推薦 AlphaGo Zero（2017）而非 AlphaGo（2016）**：2016 那篇有一半在講「從 16 萬盤人類棋譜做監督學習」，
而本專案**沒有人類示範資料**，那半篇用不上。Zero 把人類知識整個拿掉，只剩「規則 ＋ 自我對弈 ＋ 搜尋」，
與這裡的處境（沒有示範，但有一個完美的驗證器 `dfs.py`）近得多。

**讀了能立刻換成行動的一個概念**：AlphaGo 的核心洞見**不是「網路很強」，是「網路把搜尋的分支砍掉」**。
它的網路單獨下棋只有業餘水準，是 MCTS 把它撐到職業水準。

這與本專案**已經量到的數字**直接對上（handover 陷阱 #19）：

| 推論方式 | solve（4×4） |
|---|---|
| deterministic | 0.870 |
| 抽樣 best-of-2 | 0.903 |
| 抽樣 best-of-16 | **0.967** |

同一個 policy，只是多給幾次嘗試就從 0.870 跳到 0.967——**這就是「policy 引導搜尋」的雛形**，
只是目前的搜尋是最笨的一種（獨立重抽）。AlphaZero 式做法是把同樣預算花在**有結構的搜尋**上。
而本專案已經有現成的 DFS ⇒ **拿 policy 去排序 DFS 的分支順序，就是最省事的 AlphaGo 式混合**，
比自寫 MCTS 便宜得多，也是 A6 的自然起點。

**為什麼現在不是時候**：目前卡住的是 curriculum 預算，不是演算法選錯。
換演算法會讓人分不清進步來自哪裡——這正是本 track 反覆強調的「一次只改一件事」。

### 5.5 兩個本質差異（不記住會套錯）

1. **圍棋是雙人零和，Zip 是單人確定性搜尋。**
   沒有對手 ⇒ **沒有 self-play 這回事**。AlphaZero 一半的機器（自我對弈產生資料、ELO 評估、對手池）
   在這裡不存在，取而代之的是出題器 ＋ CP-SAT 驗證。
2. **圍棋沒有「正確答案」只有勝負，Zip 有完美 verifier。**
   這是**優勢**：它讓 best-of-N 成為正當手段（可驗證哪次對了，不是猜），
   也讓 RLVR／GRPO 那條路線（restart plan §7）成立。圍棋 AI 沒這個東西可用。

---

## 6. 本報告未驗證的部分

- **只有單一 seed**。所有結論仍建立在 `seed=20260815` 一次訓練上，跨 seed 重複沒做。
- **「加預算就會到全長」只被一次晉級支持**（見 §3.3），替代解釋尚未排除。
- **§3.2 的外推不可信**，理由已寫在該節。
- **`shaping_lambda=0` 的對照仍未跑**（handover §6 的第二件事），
  所以目前所有數字都是在一個「與 restart plan 規格不符」的設定下量到的。
- **GPU utilization 的桌面基線是同一天量的**，但桌面程式的負載本身會變動，
  38 個百分點是單次對照的差值，不是統計量。

---

## 來源

- AlphaGo Zero 論文全文（Nature 2017）：<https://discovery.ucl.ac.uk/id/eprint/10045895/1/agz_unformatted_nature.pdf>
- AlphaGo feature planes 與網路結構整理：<https://jonathan-hui.medium.com/alphago-how-it-works-technically-26ddcc085319>
- AlphaGo 概觀：<https://towardsdatascience.com/alphago-how-ai-mastered-the-game-of-go-b1355937c98d/>
- 搜尋關鍵字（去識別化）：`AlphaGo Zero 17 input feature planes AlphaGo 48 planes policy network architecture`
- 專案內：`ai-collab/reports/2026-08-15_rl-restart-plan.html` §5.1／§8／§9；
  `src/core/rl/train_config.py`；`src/core/rl/train_maskable_ppo.py`；
  `logs/rl_a2/goal2_6x6_bigdata/`
