# P4c 正式微調結果 — Qwen3.5-4B ＋ bf16 LoRA

> 2026-08-22（Asia/Taipei）｜分支 `feat/vlm-parser`｜worktree `zip-vlm`
> 一句話：**微調完全成功，合成 held-out 200/200 全項滿分；也因此這個評估集已經飽和、失去鑑別力。**
> 原始資料：[`artifacts/vl-p4c/`](artifacts/vl-p4c/)｜執行存證：[`../../notebooks/p4c_finetune_8000.ipynb`](../../notebooks/p4c_finetune_8000.ipynb)（保留全部輸出）

---

## 1. 結果

在 200 筆**未參與訓練**的合成 held-out 上：

| 指標 | 未微調 baseline（六張真實截圖，`--prompt sized`） | **P4c 微調後（200 筆合成 held-out）** |
|---|---|---|
| JSON 可解析率 | 6/6 | **200/200** |
| 格盤尺寸正確 | 6/6 | **200/200** |
| 逐格準確率 | 0.947 | **1.000** |
| 號碼格召回 | 0.917 | **1.000** |
| 牆 F1（有牆題） | **0.438** | **1.000**（189 題有牆） |
| micro 牆 precision / recall | — | **1.000 / 1.000** |
| **端到端完全正確** | 2/6 | **200/200** |

⚠ **兩欄不可直接比較**——左欄是真實截圖、右欄是合成圖，任務難度不同。列在一起只是為了看出「牆」這個瓶頸被解決的幅度。

按牆數分層，**每一個桶都是 1.00**：

```
  walls     n   exact   valid  wall F1
      0    11    1.00    1.00    1.000
      1    16    1.00    1.00    1.000
      2    18    1.00    1.00    1.000
      3    15    1.00    1.00    1.000
      4    12    1.00    1.00    1.000
      5    15    1.00    1.00    1.000
      6    13    1.00    1.00    1.000
      7    16    1.00    1.00    1.000
      8    15    1.00    1.00    1.000
      9    20    1.00    1.00    1.000
     10    16    1.00    1.00    1.000
     11     9    1.00    1.00    1.000
     12    24    1.00    1.00    1.000
```

連 24 題 12 道牆的都一道不差。

## 2. 滿分是 bug 報告，除非證明不是

滿分先當成自己的程式壞掉來查。五項對抗性檢查全過：

| 檢查 | 結果 | 意義 |
|---|---|---|
| `raw_output` 是否偷抄 `label` | **200/200 逐位元組相同** | 這是**預期的**——訓練目標就是 `to_prompt_json` 那個確切序列化。單獨看不足以證明什麼，要配下一列 |
| 生成時間是否為真 | `generation_seconds` **200 個相異值**，17.4～50.8 s | 每次呼叫都實測，不是常數，確實在生成 |
| 切分是否為宣稱的尾巴 200 筆 | 檔名順序完全吻合；**與訓練集檔名交集 0** | 切分正確 |
| 內容層級是否洩題 | 標籤重複 **0**、渲染 recipe 重複 **0**、**圖片位元組重複 0** | held-out 是真的沒看過 |
| 不經 `score_predictions` 再算一次 | 獨立結構比對 **200/200 一致** | 排除 scorer 自身錯誤 |

驗證腳本：`scratchpad/sanity_check_p4c.py`（一次性，未進版控）。
預測檔 sha256 `fc96cb30770ba2226012fa7da5fa63732afd6b7c3001df2d0e843ef103694e22`。

## 3. 訓練實測

資料：`zip_vl_6x6_8000_20260822.tar`（sha256 `69c753e1…0fbf`），前 7,800 筆訓練、**尾巴 200 筆 held-out**。

| | P4a 外推 | **實際** |
|---|---|---|
| 步數 | 975 | **975** |
| s/step | 7.54 | **5.77** |
| 總時長 | 2.04 h | **1.56 h** |
| 峰值 VRAM | 16.57 GiB (75%) | **20.90 / 22.03 GiB (94.9%)** |
| `train_loss`（全程平均） | — | 0.0138 |

loss 曲線（每一步看到的都是全新樣本，1 epoch 不重複）：

```
step  10  0.901588      step 490  0.000431
step 130  0.000941      step 730  0.000020
step 250  0.000071      step 970  0.000004
```

**第 250 步（約 2,000 筆）就掉到雜訊地板。** 完整 97 列在 notebook 裡，機器可讀版在 Drive 的 `checkpoint-975/trainer_state.json`。

視覺層確實學到，且動得比語言層大——與「瓶頸是視覺」的假設一致：

```
visual   : 96/96  lora_B non-zero, max|B| = 2.723e-01   (P4a 50 步時 0.114)
language : 248/248 lora_B non-zero, max|B| = 1.662e-01   (P4a 0.059)
```

**VRAM 94.9% 推翻了「batch 可以往上調」的假設。** P4a 的 75% 是 50 步沒抽到大圖；跑滿 975 步一定會遇到 998px 的樣本。**batch 2 在 L4 上已經頂到天花板。**

## 4. 成本結算

| | 時間 | CU |
|---|---|---|
| 訓練 | 1.56 h | ~2.4 |
| 推論（200 筆 × 34.5 s） | 1.92 h | ~3.0 |
| 安裝／載模型／dry run | ~0.4 h | ~0.6 |
| **合計** | **~3.9 h** | **~6.0** |

原規劃 3.2 CU。**超支全在推論，且是設計錯誤造成的**（見 §5.3）。

## 5. 這一輪暴露的四個缺陷（都是本次新寫的程式造成的）

### 5.1 held-out 差點用到跟訓練集重疊的資料

原計畫拿現成那包 120 筆 `smoke_6x6` 當 held-out。實測它與訓練用的 `main_6x6`：**渲染 recipe 120/120 完全相同、標籤 82/120 完全相同**。

根因：`dataset_builder.draw_recipe` 用 `random.Random(seed + index)`，兩包 seed 只差 1（20260823 vs 20260822）⇒ **同一個亂數流位移一格**，`smoke[i]` 就是 `main[i+1]`。只有 `generate_puzzle` 的 wall-clock 不決定性讓 38 筆長得不一樣。

**教訓：要獨立的資料集，seed 要差得夠遠，或直接從同一包切 disjoint 的 slice。** 後者零成本又保證不重疊，P4c 採用之。
（不影響 P4a：它訓練 `smoke[4:]`、評估 `smoke[:4]`，對它自己確實未見過。）

### 5.2 dry run 的計時投影完全不能用

第 7 節的 lr=0 短跑量到 **37.59 s/step**，印出「projected **10.18 h**（15.7 CU）」——實際 5.77 s/step / 1.56 h，**誤差 6.5 倍**。

根因：5 步裡第一步吃掉全部的編譯與 autotune 成本（推算約 170 秒），平均下來就爆掉。
**修法：丟掉第一步再平均，或把那個數字標成「上限」而非「預估」。** 它的其他用途（驗證 plumbing、量記憶體、證明權重不動）都成立且有價值，只有計時要修。

### 5.3 ★ 評估用 batch 1 逐筆生成 — 本輪最大的成本錯誤

推論 3.0 CU > 訓練 2.4 CU，本末倒置。

機制：batch 1 自回歸解碼要把 9.16 GB 權重**每個 token 讀一遍**，L4 的 300 GB/s 給出 30.5 ms/token 的地板；輸出約 300 tokens ⇒ 理論 9.2 s，實際 34.5 s，**只跑到 roofline 的 27%**。差額是每 token 約 80 ms 的固定開銷：344 個未 merge 的 LoRA adapter（每 token 多 688 次 kernel 發動）、HF `generate` 迴圈的 Python 開銷、沒有 CUDA graph。**GPU 大部分時間在發呆。**

**修法（下一輪必做）**：① 批次生成（一次 8～16 筆，因為瓶頸是每 token 固定成本，批次化近乎線性加速，估計 2 小時 → 15 分鐘）② 推論前 merge LoRA ③ 或直接走 P4d 匯出後的 Ollama／llama.cpp。

### 5.4 「逐行寫入 Drive 所以斷線也保得住」是錯的

`run_predictions` 用 `with out_path.open("w")` 開在 Drive 掛載上，逐行 `flush()`。但 **Colab 的 Drive FUSE 只在檔案關閉時才上傳到雲端**——`flush()` 只推到 FUSE 層。實際後果：跑到一半時檔案在 drive.google.com 上**完全看不到**，而且真的斷線的話那批會全部消失。

**修法：寫本機 `/content`，每批用 `shutil.copy` 覆蓋到 Drive**（copy 會開檔關檔，才會真的觸發上傳）。

### 5.5 附帶修正一個文件錯誤

P4a notebook 的 markdown 寫「訓練沒有 `<think>` 區塊」，P4c 沿用了那句。實測輸出：

```
prompt ends with: '...<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

**訓練渲染本來就含空的 `<think>\n\n</think>\n\n`。** 這反而證明 `build_inference_prompt` 的設計是對的——它從訓練渲染路徑推導，**不依賴我對訓練渲染的描述正確**，而我確實又描述錯了一次。

## 6. 硬體對照（順帶量的，決定「下一輪在哪裡跑」時要用）

同一段 4096×4096 matmul（`colab_smoke_test.ipynb` 第 8 格），兩邊各量一次：

| | 本機 RTX 4070 Ti SUPER | Colab L4 | 比值 |
|---|---|---|---|
| 架構 / capability | Ada, 8.9 | Ada, 8.9 | 同代 |
| VRAM | 15.99 GiB | 22.03 GiB | — |
| bf16 matmul | **90.06 TFLOP/s** | 64.11 TFLOP/s | **本機快 1.40×** |
| fp32 matmul | 31.18 | 12.43 | 本機快 2.51× |
| 記憶體頻寬 | **588 GB/s**（實測 copy） | 300 GB/s（官網規格，GDDR6，非 HBM） | 本機快 1.96× |

**L4 不是比較快，是 VRAM 比較大。** 訓練峰值 20.90 GiB ⇒ 本機在 batch 2 完全沒機會。
訓練期間的 MFU 約 **42%**（`6 × 4.578e9 × 5,600 tokens ÷ 5.77 s ÷ 64.11 TFLOP/s`），頻寬只佔約 4%，**訓練是算力／發動綁死，不缺頻寬**。

### batch size：往上調反而虧

圖是變動尺寸的（`cell_size` 隨機），micro-batch 會被 padding 補到最長者。抽 600 張真實訓練圖量測（token 數以 E1 的 656px→529 tokens 校準）：

```
image side  min 472  max 998        tokens  min 274  median 594  max 1099

 micro-batch  padded tokens/sample    waste  vs batch 2
           1                   621    0.0%       1.00x
           2                   767   18.9%       1.00x   ← 本輪採用
           4                   902   30.6%       1.18x
           8                   993   37.2%       1.29x
```

**batch 2 已經浪費 18.9% 的算力在 padding 上**，往上調要多付 18～29%，而 MFU 只有 42%、能賺回來的有限。**反而 `batch=1, accum=8` 可以把 padding 歸零、少算 19% 的 token**（代價是 GEMM 變瘦），值得用 lr=0 短跑實測。標準解是 `group_by_length=True`，但與 `skip_prepare_dataset=True` ＋ lazy transform 能否共存**未驗證**。

## 7. 結論與影響

**✅ 微調完全成功。** 未微調時唯一的瓶頸「牆」被徹底解決，端到端從 2/6 到 200/200。管線 `image → JSON → Puzzle → solver` 的讀圖端已經達標。

**⚠ 但這個評估集同時飽和了。** 所有指標都是 1.000 ⇒ **它再也量不出任何差異**。往後任何改動——視覺層消融、CoD 變體、更少的訓練資料、batch 調整——**在這個集合上都會是 1.000**，無法比較。

**★ 本人 2026-08-22 定案：不做真實截圖，就用自產的合成資料。** 因此：

- 所有數字的意義是「**學會了我們的 renderer**」，**不是**「看得懂 LinkedIn 截圖」。這是明示接受的取捨，不是疏漏。
- 要重新獲得鑑別力，唯一剩下的路是**把合成資料變難**（更多視覺雜訊、多種 renderer 風格、模擬截圖壓縮與縮放失真、更大盤面）。**這是未來若要再訓練時的前置工作**，不做就沒有尺可量。

**一個還能在合成集上量到東西的實驗**：Drive 上還有 `checkpoint-200/400/600/800`。loss 在第 250 步就到地板，若 checkpoint-200 對同一批 held-out 也是 200/200，代表 **1,600 筆就夠了**，本輪 4/5 的訓練量是白付的。這個比較的是「不同模型」而非「同一個滿分」，所以仍有鑑別力，且成本只有一次批次推論。

## 8. 產物位置

| 東西 | 位置 |
|---|---|
| LoRA adapter（168 MB） | Google Drive `colab_finetune/p4c_qwen35_4b_zip_lora` |
| checkpoint 200/400/600/800/975 | Google Drive `colab_finetune/p4c_qwen35_4b_zip_checkpoints/` |
| 200 筆原始預測 | [`artifacts/vl-p4c/p4c_holdout_predictions.jsonl`](artifacts/vl-p4c/p4c_holdout_predictions.jsonl) |
| 算分結果（含逐筆） | [`artifacts/vl-p4c/p4c_holdout_predictions_scored.json`](artifacts/vl-p4c/p4c_holdout_predictions_scored.json) |
| 訓練資料集 | 本機 `datasets/vl/main_6x6/`（不進版控）＋ Drive 的 tar |
| 執行存證（含 loss 表與全部輸出） | [`../../notebooks/p4c_finetune_8000.ipynb`](../../notebooks/p4c_finetune_8000.ipynb) |
