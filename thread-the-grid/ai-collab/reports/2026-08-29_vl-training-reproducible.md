# VLM 微調完整訓練報告（可復現）— Qwen3.5-4B ＋ bf16 LoRA

> 2026-08-29（Asia/Taipei）｜分支 `feat/vlm-parser`｜worktree `zip-vlm`
> 涵蓋 2026-08-22 的 P4a／E0／E1／P4c 四段執行，以「照著做能重跑一次」為標準寫。
> 結果報告（發生了什麼、數字多少）是 [`2026-08-22_vl-p4c-results.md`](2026-08-22_vl-p4c-results.md)；
> **本檔補的是「怎麼做到的」**：資料怎麼生、哪些沒見過、每一個超參數、參考了什麼、以及從零復現的步驟。

---

## 0. 這份報告怎麼用

| 你想做的事 | 讀哪幾節 |
|---|---|
| 只想知道結論 | §1 |
| 想知道「模型到底沒看過什麼」 | §3（四個層級 ＋ 五項對抗性檢查） |
| 想原封不動再跑一次 | §4 ＋ §8 |
| 想改一個變數再跑 | §4 ＋ §9（先讀 §9，很多坑已經踩過） |
| 想知道花了多少錢 | §6 |
| 想知道當初憑什麼這樣選 | §7 |

**一句話定位**：這一輪證明的是「模型學會了讀**我們自己畫的**盤面」。它**不證明**「看得懂 LinkedIn 的真實截圖」——這是本人 2026-08-22 明示接受的取捨，理由見 §10。

---

## 1. 一頁摘要

| 項目 | 值 |
|---|---|
| 基礎模型 | `unsloth/Qwen3.5-4B`（4.58B 參數，vision + language） |
| 微調方式 | **16-bit（bf16）LoRA**，r=16 / alpha=16，vision 與 language 全開 |
| 可訓練參數 | **38,756,352 / 4,578,021,888 = 0.85%** |
| 訓練資料 | 自產合成資料 **7,800 筆**（6×6、0–12 道牆） |
| held-out | 同一包的**尾巴 200 筆**，未參與訓練 |
| 硬體 | Google Colab **L4**（22.03 GiB、capability 8.9、原生 bf16） |
| 訓練量 | 1 epoch＝**975 步**，effective batch 8 |
| 實際耗時 | **1.56 小時**、5.77 s/step |
| 峰值 VRAM | **20.90 / 22.03 GiB（94.9%）** |
| 最終 train_loss | 0.0138（全程平均）；第 250 步就到雜訊地板 |
| held-out 成績 | 四層指標**全部 1.000**，端到端 **200/200** |
| 總花費 | 約 **10 個 Colab 運算單元 ≈ 1 美金**（拆解見 §6） |

---

## 2. 訓練資料：用了什麼

### 2.1 資料集本體

| 欄位 | 值 |
|---|---|
| 名稱 | `main_6x6` |
| 打包檔 | `zip_vl_6x6_8000_20260822.tar` |
| **打包檔 SHA-256** | `69c753e16d030ecba7a7da505f046b747e275ec7ca956c0bada80a6d7ce70fbf` |
| 筆數 | 8,000（訓練 7,800 ＋ held-out 200） |
| `metadata.jsonl` SHA-256 | `2c203114adb71d3bd557bcf057910b7083913baf5d348ffd7b18e8fab0818b73` |
| 影像位元組 SHA-256 | `7c1e4a53c9f0c21585472a48c1ae0670bc721495350a04628b1c04e4e80e3277` |
| 產生種子 | `20260822` |
| 格盤尺寸 | **100% 6×6**（`size_histogram: {"6": 8000}`） |
| CP-SAT 驗證 | 全數通過（`verified_with_cp_sat: true`、`unsolvable_rejected: []`） |
| 生成失敗筆數 | 0 |

牆數分布（刻意壓平，每一桶約 600 筆）：

```
walls    0    1    2    3    4    5    6    7    8    9   10   11   12
  n    589  610  633  575  648  640  613  641  584  611  627  601  628
```

主題分布：`light` 6,046 / `dark` 1,954。

> ⚠ **重跑同一個指令不會得到同一包資料。** `generate_puzzle` 用**牆鐘時間**中斷隨機回溯搜尋，
> 被砍掉的搜尋消耗的亂數量與跑完的不同，所以同 seed 兩次會分岔（實測 0.5s 預算下 30 筆差 8 筆，
> 5s 下仍差 3 筆）。**可復現的單位是「那一包資料」，不是「那個指令」**——所以 manifest 存了三個
> SHA-256，Colab 上讀檔前會逐位元組比對。

### 2.2 一筆樣本是怎麼生出來的

```
generate_puzzle(6×6, has_walls=False)      # 先畫一條 Hamiltonian path 再挖題
      ↓
sample_walls(path, wall_count, rng)        # 只從「解答路徑不會穿過」的邊裡抽牆
      ↓
render_puzzle(puzzle, recipe)              # LinkedIn 風格：格線淺灰、牆粗黑
      ↓
to_prompt_json(from_puzzle(puzzle))        # 標籤 ＝ 推論要輸出的那個確切字串
      ↓
solve_puzzle_cp(puzzle)                    # CP-SAT 確認仍可解，不可解就丟掉
```

**「只抽解答不穿過的邊」保證加牆不會讓題目變成無解**——這在理論上是顯然的，但計畫的 done 條件
要求實證，所以每一筆仍然過 CP-SAT。這個性質後來變成產品端的免費驗證器（§10）。

### 2.3 每一筆的隨機維度（全表）

每筆用**自己的亂數流** `random.Random(seed + index)`，抽完寫進 `metadata.jsonl`，所以任何一筆都可單獨稽核或重畫。

| 維度 | 取值 | 機率／分布 |
|---|---|---|
| `size` | 6 | 固定（本輪範圍縮到 6×6） |
| `wall_count` | 0–12 | 均勻 |
| `theme` | `light` / `dark` | dark 機率 **0.25** |
| `cell_size` | 72 / 86 / 100 / 116 / 132 px | 均勻（base 100） |
| `show_buttons` | bool | **0.65** |
| `show_cursor` | bool | **0.20** |
| `rotation_degrees` | ±2.0° | **0.30** 的機率才旋轉，否則 0 |
| `jpeg_quality` | 60–95 | **0.35** 的機率存成 JPEG，否則 PNG |

實際影像尺寸因此是變動的：抽樣 600 張量到**邊長 472–998 px**、token 數 274–1099（中位 594）。
這件事後來決定了 batch size 的選擇（§9.3）。

一筆 `metadata.jsonl` 長這樣：

```json
{
  "file_name": "images/000000.jpg",
  "label": "{\n  \"layout\": [...],\n  \"walls\": [...]\n}",
  "grid_size": 6, "wall_count": 2, "waypoint_count": 10,
  "theme": "light", "cell_size": 86,
  "show_buttons": true, "show_cursor": false,
  "rotation_degrees": 0.0, "jpeg_quality": 91, "seed": 20260822
}
```

### 2.4 標籤格式 ＝ 推論格式（這不是小事）

`label` 就是 `schema.to_prompt_json()` 的輸出，也就是**推論時要求模型吐的那個確切序列化**
（鍵序、縮排、牆的排序全部固定）。因此微調沒有任何「轉換層」。

2025-10 那一輪不是這樣：標籤是 Chain-of-Draft 結構，要再轉一次才變成 parser 吃的格式，
於是**訓練目標從來就長得跟推論要的不一樣**。`src/core/tests/vl_models/test_schema.py` 就是
用來釘住這件事不再走鐘的。

### 2.5 為什麼舊資料集不能用（2026-08-22 實測 4,000 對）

`zip_puzzles/cod_dataset_20251028_124732`：

| 檢查 | 結果 | 後果 |
|---|---|---|
| 標籤正確性 | 60/60 可解 | ✅ 標籤本身沒錯 |
| 格盤尺寸 | **100% 6×6** | 學不到尺寸變化 |
| 牆數 | **只有 2–5 道** | 真實截圖有 0 道也有 10 道 |
| 障礙格 | **零** | 學不到 `xx` |
| **牆的顏色** | **與格線同為黑色** | ★ **它教的辨識線索在真實截圖上不存在** |

最後一列是致命的：舊 renderer 把牆和格線畫成同色，模型只能靠粗細分辨，而真實截圖裡格線是淺灰的。
**可以拿來當 P4a 煙霧測試的輸入，不可以當正式訓練集。**

---

## 3. 訓練時沒見過的資料

「沒見過」有四個強度不同的層級，混在一起講就會高估自己。

### 3.1 層級一：held-out 200 筆（本輪的正式評估集）

- **怎麼切**：`records[:-200]` 訓練、`records[-200:]` held-out。同一個檔案、同一個 SHA-256，不必多上傳一包。
- **為什麼從尾巴切而不是另外生一包**：零成本，而且**保證不重疊**（見 §3.2 的教訓）。
- 檔名交集：**0**。
- held-out 的牆數分布：`{0:11, 1:16, 2:18, 3:15, 4:12, 5:15, 6:13, 7:16, 8:15, 9:20, 10:16, 11:9, 12:24}`
- held-out 主題：`light` 156 / `dark` 44。

### 3.2 層級一差點沒守住：`smoke_6x6` 洩題事件

原計畫拿現成那包 120 筆 `smoke_6x6`（seed 20260823）當 held-out。上機前實測它與訓練用的
`main_6x6`（seed 20260822）：

- **渲染 recipe 120/120 完全相同**
- **標籤 82/120 完全相同**

根因：`draw_recipe` 用 `random.Random(seed + index)`，兩包 seed 只差 1
⇒ **同一條亂數流整包位移一格**，`smoke[i]` 就是 `main[i+1]`。只有 `generate_puzzle` 的牆鐘不決定性
讓另外 38 筆長得不一樣。

> **教訓：要獨立的資料集，seed 要差得夠遠，或直接從同一包切 disjoint 的 slice。**
> 後者零成本又保證不重疊，本輪改用之。

### 3.3 五項對抗性檢查（滿分先當自己的 bug 來查）

四層指標全 1.000 的時候，第一件事是懷疑自己的程式壞了。五項全過：

| 檢查 | 結果 | 意義 |
|---|---|---|
| `raw_output` 是否偷抄 `label` | 200/200 逐位元組相同 | **這是預期的**——訓練目標就是那個確切字串。單看不足以證明什麼，要配下一列 |
| 生成時間是否為真 | `generation_seconds` **200 個相異值**（17.4–50.8 s） | 每次都真的在生成，不是常數 |
| 切分是否為宣稱的尾巴 200 筆 | 檔名順序吻合、與訓練集交集 **0** | 切分正確 |
| 內容層級洩題 | 標籤重複 **0**、渲染 recipe 重複 **0**、**圖片位元組重複 0** | 真的沒看過 |
| 不經 scorer 再算一次 | 獨立結構比對 200/200 一致 | 排除 scorer 自身錯誤 |

預測檔 SHA-256：`fc96cb30770ba2226012fa7da5fa63732afd6b7c3001df2d0e843ef103694e22`

### 3.4 層級二～四：更強意義的「沒見過」

| 層級 | 內容 | 訓練時見過嗎 |
|---|---|---|
| 二｜**不同的渲染 recipe** | held-out 用到的主題／格子大小／旋轉／JPEG 品質組合 | 個別維度見過，**該組合的圖片位元組未見過** |
| 三｜**六張真實 LinkedIn 截圖** `illustrations/puzzle_01..06` | 完全沒進訓練集 | ❌ 完全沒見過 |
| 四｜**真實截圖的分布** | 不同 renderer、壓縮、縮放、UI 版本 | ❌ 且**本輪沒有量過** |

> ★ 層級三的六張截圖雖然沒進訓練，但**其中 `puzzle_01`–`03` 的答案寫在 baseline prompt 的
> few-shot 範例裡**，所以它們的成績不能當泛化證據。這是 baseline 測量的既有限制，不是本輪引入的。

---

## 4. 怎麼訓練的（可復現）

### 4.1 硬體與執行環境

| | 值 |
|---|---|
| 平台 | Google Colab（付費，Pay As You Go） |
| GPU | **NVIDIA L4**，capability (8, 9)，VRAM 22.03 GiB |
| **原生 bf16** | `torch.cuda.is_bf16_supported(including_emulation=False)` → **True** |
| Python | 3.12.13 |
| 驅動端 | Torch 2.8.0+cu128、CUDA Toolkit 12.8、Triton 3.4.0 |
| 系統 RAM | 53.0 GiB |
| 操作方式 | VS Code 連遠端 Colab kernel（不需 WSL2） |

**為什麼一定要 L4 而不是免費 T4**：Unsloth 不建議對 Qwen3.5 用 QLoRA（量化誤差偏大），
所以這條路需要原生 bf16；而 T4 是 Turing，**沒有 bf16 硬體**。

> ⚠ `torch.cuda.is_bf16_supported()` 預設 `including_emulation=True`，在 T4 上**照樣回 True**。
> 一律問 `including_emulation=False`。notebook 第 2 節就是為了這件事存在，並在 capability < 8.0 時直接 assert 失敗。

### 4.2 套件安裝（**逐行照抄，不要「現代化」**）

整段抄自 Unsloth 官方的 `Qwen3_5_(4B)_Vision.ipynb`。這個堆疊是建在 torch 2.8.0 上的，
而 Colab 預設更新；CUDA extension 只對釘住的那版編得起來。

```python
!pip install --upgrade -qqq uv
!uv pip install -qqq \
    "torch==2.8.0" "triton>=3.3.0" numpy pillow torchvision bitsandbytes \
    xformers==0.0.32.post2 \
    "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+https://github.com/unslothai/unsloth"
!uv pip install -qqq --no-deps "torchcodec==0.7.0"
!uv pip install --upgrade --no-deps "tokenizers>=0.22.0,<=0.23.0" trl==0.22.2 unsloth unsloth_zoo
!uv pip install transformers==5.2.0
!uv pip install --no-build-isolation flash-linear-attention causal_conv1d==1.6.0
# capability >= 8 才裝：
!uv pip install --no-deps "apache-tvm-ffi==0.1.9" "tilelang==0.1.8"
!uv pip install --no-deps --upgrade "torchao>=0.16.0"
```

實際跑起來的版本：**Unsloth 2026.8.19、transformers 5.2.0、trl 0.22.2**。

> ⚠ **不要用 `%%capture` 包安裝格**。上游用它讓 notebook 乾淨，但這裡如果安裝靜默失敗，
> 一小時後會變成莫名其妙的 `import unsloth` 錯誤。
> ⚠ **cell magic 必須在第一行**：在 `%%capture` 上面放註解，Jupyter 會報
> `Line magic function not found`，整格不執行、後面全部連鎖失敗。

### 4.3 模型與 LoRA 設定

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3.5-4B",
    load_in_4bit = False,              # -> 16-bit LoRA（不是 QLoRA）
    use_gradient_checkpointing = "unsloth",
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True,   # ★ 不能關，見下
    finetune_language_layers   = True,
    finetune_attention_modules = True,
    finetune_mlp_modules       = True,
    r = 16, lora_alpha = 16, lora_dropout = 0,
    bias = "none", random_state = 3407,
    use_rslora = False, loftq_config = None,
)
```

**為什麼 vision 層一定要開**：未微調時的失敗**純粹是視覺的**——逐格 0.961、號碼召回 0.910、
格盤尺寸 6/6 全對，**只有牆 F1 是 0.438**。凍結 vision tower 省記憶體會剛好凍掉唯一需要學的東西。
P4a 先驗證過 vision 層確實會動，而且動得比 language 層大。

輸出的 `adapter_config.json`（實際落盤值）：`peft_version 0.19.1`、`task_type CAUSAL_LM`、
`use_dora false`、`use_rslora false`、`fan_in_fan_out false`、`modules_to_save null`、
`rank_pattern {}`、`alpha_pattern {}`。`target_modules` 是一條同時涵蓋
`vision|image|visual|patch` 與 `language|text` 的正規表示式。

### 4.4 資料載入：必須是 lazy 的

P4a 那支 loader 把每張圖解碼成 PIL 物件放進 list。120 筆沒事，**8,000 筆會死**：
一張解碼後約 1.2 MB ⇒ 全集約 **9.1 GB**，而標準 Colab VM 記憶體約 12.7 GB。

```python
train_dataset = Dataset.from_list(
    [{"file_name": r["file_name"], "label": r["label"]} for r in train_records]
)
train_dataset.set_transform(transform)   # 存取時才 Image.open
```

實測（跑 300 次存取）：**RSS 只增加 +9.7 MB**，而同樣 300 筆若 materialise 要約 360 MB。

> `imagefolder` 是顯而易見的替代方案，而且**兩處都錯**：它把 `images/` 當類別目錄讀，
> 而且它自動產生的 `label` 欄位會跟這個資料集既有的 `label` 撞名。

### 4.5 花兩小時之前先做 lr=0 的 dry run

跑 5 個**真步**，但 `learning_rate = 0.0`。dataloader、lazy 解碼、collation、forward、backward、
optimizer 全部真的走一遍，而權重**可證明**沒動：

```python
before = sample_lora_b.detach().clone()
dry_run.train()
assert torch.equal(before, sample_lora_b)   # 這行讓「可證明」不只是宣稱
```

實測輸出：`weights unchanged: True`、峰值 VRAM 13.93 GiB。

> ⚠ **它量到的 s/step 不能用來外推**（見 §9.2）。這一格量到 37.59 s/step、投影 10.18 小時，
> 實際是 5.77 s/step、1.56 小時，**差 6.5 倍**。plumbing 驗證與記憶體量測有效，只有計時無效。

### 4.6 訓練超參數（全表）

```python
SFTConfig(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,        # effective batch = 8
    num_train_epochs            = 1,
    warmup_steps                = 30,       # ~3% of 975
    learning_rate               = 2e-4,
    logging_steps               = 10,
    optim                       = "adamw_8bit",
    weight_decay                = 0.001,
    lr_scheduler_type           = "linear",
    seed                        = 3407,
    output_dir                  = <Drive>/p4c_qwen35_4b_zip_checkpoints,
    save_steps                  = 200,
    save_total_limit            = 2,        # ★ 見 §9.5
    report_to                   = "none",
    dataloader_num_workers      = 2,
    remove_unused_columns       = False,
    dataset_text_field          = "",
    dataset_kwargs              = {"skip_prepare_dataset": True},
    max_length                  = 2048,
)
data_collator = UnslothVisionDataCollator(model, tokenizer)
```

幾個值的理由：

- **`max_length = 2048`**：最長的標籤是 769 字元（約 250 token），所以這個預算幾乎全給影像 token，不是答案。
- **`warmup_steps = 30`**：P4a 用 5 是因為那是 50 步測試的 10%；975 步的常規約 3%。
- **`save_steps = 200`**：往 Drive FUSE 寫約 350 MB 很慢，存太密會把一大塊時間花在 I/O，存太稀又可能損失半小時以上的進度。

### 4.7 checkpoint 與斷線續跑

存到 Google Drive，斷線後從頭重跑 notebook 並把 `RESUME = True`。
`resume_from_checkpoint` 要有意義必須 seed 與資料順序都固定，兩者都固定了。

### 4.8 實測結果

| | P4a 外推 | **實際** |
|---|---|---|
| 步數 | 975 | **975** |
| s/step | 7.54 | **5.77** |
| 總時長 | 2.04 h | **1.56 h** |
| 峰值 VRAM | 16.57 GiB (75%) | **20.90 / 22.03 GiB (94.9%)** |
| train_loss（全程平均） | — | **0.0138** |

> ★ **94.9% 推翻了「batch 還可以往上調」的假設。** P4a 的 75% 是 50 步沒抽到大圖；
> 跑滿 975 步一定會遇到 998 px 的樣本。**別用短跑的 VRAM 決定 batch size。**

### 4.9 loss 曲線（每 10 步記錄，共 97 點）

每一步看到的都是全新樣本（1 epoch、不重複）：

```
step  10  0.901588     step 210  0.000273     step 560  0.000007
step  20  0.378245     step 260  0.000090     step 660  0.000080
step  60  0.002926     step 310  0.000210     step 760  0.000006
step 110  0.001069     step 410  0.000228     step 860  0.000004
step 160  0.000335     step 460  0.000204     step 960  0.000002
```

**第 250 步（約 2,000 筆）就掉到雜訊地板。** 完整 97 列在 notebook 的執行輸出裡，
機器可讀版在 Drive 的 `checkpoint-975/trainer_state.json`。

### 4.10 LoRA 到底有沒有學到東西

「視覺層有沒有真的動」不是修辭問題，因為瓶頸就在視覺：

```
visual   :  96/96  lora_B non-zero, max|B| = 2.723e-01   (P4a 50 步時 0.114)
language : 248/248 lora_B non-zero, max|B| = 1.662e-01   (P4a 0.059)
```

**視覺層動得比語言層大**，與「瓶頸是視覺」的假設一致。

> 2026-08-29 補一個獨立佐證：把這個 adapter merge 進 base 之後逐張量比對，
> `visual` 96/96 全部改變、`max|delta| = 2.24e-2`；`language` 248/248 全部改變、`max|delta| = 6.03e-3`。
> **兩種完全不同的量法給出同一個方向**（視覺 > 語言），而且張量數 96/248 與訓練當下印出的完全吻合。

---

## 5. 推論與評分

### 5.1 ★ 推論 prompt 必須從「訓練那條渲染路徑」推導

這是整條 track 最貴的教訓，**兩次都是因為我對訓練渲染的描述是錯的**。

```python
def build_inference_prompt(tokenizer, instruction: str) -> str:
    """從訓練用的同一個 template 呼叫渲染，再從答案處切開。"""
    sentinel = "@@ANSWER@@"
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": instruction},
                                     {"type": "image"}]},
        {"role": "assistant", "content": [{"type": "text", "text": sentinel}]},
    ]
    return tokenizer.apply_chat_template(conversation, tokenize=False).split(sentinel)[0]
```

同一個 adapter、同樣 4 張 held-out，**只換 prompt 渲染方式**：

| | JSON 可解析 | 結果 |
|---|---|---|
| 修好（`build_inference_prompt`） | **4/4** | 版面全對、牆 F1 0.958 |
| 壞掉（手動拼 prompt） | **0/3** | 完全不可用 |

**內容全對、格式全錯。** 兩處不一致分別是 ① thinking 區塊 ② text/image 順序。

而實際印出來的 prompt 結尾是：

```
'...<|im_start|>assistant\n<think>\n\n</think>\n\n'
```

——**訓練渲染本來就含空的 `<think>\n\n</think>\n\n`**，而 P4a 的 notebook markdown 寫的是
「訓練沒有 `<think>` 區塊」，P4c 沿用了那句錯的描述。這正好證明這個設計是對的：
**它不依賴我對訓練渲染的描述正確。**

> `enable_thinking=False` **不等價**於什麼都不做——它會吐出 `<think>\n\n</think>\n\n`；
> 但這裡的重點不是哪個對，是**不要用人工同步兩邊的參數**。

### 5.2 生成設定

```python
MAX_NEW_TOKENS = 700          # 最長標籤 769 字元 ≈ 250 token
model.generate(**inputs, max_new_tokens=700, use_cache=True, do_sample=False)   # greedy
```

### 5.3 評分：notebook 不算指標

notebook 只把原始輸出寫成 JSONL，指標**一律回本機算**：

```powershell
uv run python -m src.core.vl_models.score_predictions <path>\p4c_holdout_predictions.jsonl
```

這是刻意的。這個專案已經為「benchmark 程式碼與出貨 parser 各走各的」付過一次代價，
而在 notebook 裡重寫一份指標正是同一件事會再發生的方式。`score_predictions` 走的是
**出貨用的 `parse_model_output`** ＋ **benchmark 的指標函式**。

### 5.4 結果

| 指標 | 值 |
|---|---|
| JSON 可解析率 | 200/200 |
| 格盤尺寸正確 | 200/200 |
| 逐格準確率 | 1.000 |
| 號碼格召回 | 1.000 |
| 牆 F1（189 題有牆） | 1.000 |
| micro 牆 precision / recall | 1.000 / 1.000 |
| **端到端完全正確** | **200/200** |

按牆數分層，**13 個桶每一個都是 1.00**，連 24 題 12 道牆的都一道不差。

---

## 6. 成本結算

Colab 的 runtime **只要連著就在計費，閒置也算**。這是本輪最容易被忽略的成本來源。

### 6.1 逐段拆解

| 階段 | 內容 | 時間 | CU |
|---|---|---|---|
| **P4a** 煙霧測試 | 安裝 ＋ 載模型 ＋ 50 步訓練 ＋ 前後各一次推論 | ~1.0 h | ~1.5 |
| **E0／E1** 驗證 | 載已存 adapter，比較 prompt 渲染修法 ＋ 解析度定價（不訓練） | ~1.0 h | ~1.5 |
| **P4c** 訓練 | 975 步 | 1.56 h | **~2.4** |
| **P4c** 推論 | 200 筆 × 34.5 s | 1.92 h | **~3.0** |
| **P4c** 安裝／載模型／dry run | | ~0.4 h | ~0.6 |
| **合計** | | **~5.9 h** | **≈ 9–10** |

> **P4c 那三列（~6.0 CU）是 notebook 執行輸出直接算出來的**（`cost ~2.4 compute units` 那一行
> ＋ 200 × 34.5 s）。**P4a 與 E0/E1 兩列是回推的估計**，因為那兩本 notebook 沒有印 CU；
> 它們的量級與帳戶端觀察到的總量一致，但**不要把它們當實測數字引用**。
> L4 費率為 **1.54 CU/小時**（2026-08-22 於 Colab runtime 選單實測）。

### 6.2 換算

本人於帳戶端觀察到的總消耗為 **約 10 個運算單元 ≈ 1 美金**，即約 **US$0.10／CU**。
以此換算：**訓練本身（2.4 CU）約 0.24 美金**，而**推論（3.0 CU）比訓練還貴**——這是設計錯誤，見 §9.4。

---

## 7. 參考了什麼

### 7.1 直接照抄的

| 來源 | 用途 |
|---|---|
| Unsloth 官方 `Qwen3_5_(4B)_Vision.ipynb`（<https://github.com/unslothai/notebooks>） | **安裝格逐行照抄**、`FastVisionModel` 的用法、`UnslothVisionDataCollator` |
| <https://unsloth.ai/docs/models/qwen3.5/fine-tune> | Qwen3.5 的微調指引，含「不建議 QLoRA」這條 |
| 本專案 `notebooks/p4a_finetune_smoke.ipynb` | plumbing 全套（P4c 是它的放大版） |
| 本專案 `notebooks/p4a_verify_e0_e1.ipynb` | `build_inference_prompt` 的修法與對照 |

### 7.2 選型階段的依據

完整推理在 [`2026-08-15_vlm-model-survey.html`](2026-08-15_vlm-model-survey.html)（⚠ 部分內容已被
後續實測修正）。關鍵一手來源：

- <https://huggingface.co/Qwen/Qwen3.5-4B> — 模型卡
- <https://unsloth.ai/docs/models/gemma-4/train> — Gemma 4 微調指引（官方支援 QLoRA、vision 微調限 E2B／E4B）
- <https://ai.google.dev/gemma/docs/core/model_card_4> — Gemma 4 模型卡
- <https://github.com/unslothai/unsloth/issues/3899> — **vision 模型 GGUF 匯出的已知缺陷**
- <https://github.com/ollama/ollama/issues/14730> — Ollama 端的部署地雷
- <https://github.com/QwenLM/Qwen3.6>、<https://github.com/qwenlm/qwen3-vl> — 世代與尺寸確認

> ⚠ 上述連結為 **2026-08-15 查證**。版本號是知識庫裡最先過期的資訊，半年內的版本宣稱請重查。

### 7.3 選型結論（為什麼是 Qwen3.5-4B）

被**尺寸**決定的：Qwen3.6 最小 27B、Qwen3.7 無開放權重、Qwen3.8 只有 27B；27B 在 Q4 約 17 GB，
超過本機 16 GB 顯卡。**Qwen3.5 是唯一有 ≤10B 尺寸的 Qwen 世代。**
判準是「**有官方 notebook ＋ 較有機會成功**」，不是成本。

---

## 8. 從零復現的步驟

### 8.1 本機：建資料集

```powershell
cd D:\it_project\github_sync\zip-vlm\linkedin-zip-challenge
uv sync
uv run python -m src.core.vl_models.dataset_builder --count 8000 --name main_6x6
uv run python -m src.core.vl_models.dataset_builder --name main_6x6 --check   # 驗 SHA-256
```

⚠ 產出**不會**與 `69c753e1…0fbf` 相同（§2.1）。要重現**那一輪的數字**，必須用**那一包**資料，
不是重建一包。

### 8.2 上傳

把 `datasets/vl/main_6x6` 打包成 tar，放進 Google Drive 的 `colab_finetune/`，並記下 SHA-256。

### 8.3 Colab

1. VS Code → **Select Kernel → Colab → L4（付費）**。
2. 開 `notebooks/p4c_finetune_8000.ipynb`，**由上往下逐節執行**。
3. 第 3 節會比對 SHA-256，不符就 assert 失敗——**這是防呆，不要跳過**。
4. 第 7 節 dry run 通過（`weights unchanged: True`）再往下。
5. 第 8 節開跑，1.56 小時。
6. 第 10 節存 adapter、第 11 節跑 held-out 推論。
7. **`Runtime → Disconnect and delete runtime`**——閒置與訓練同費率。

### 8.4 回本機算分

```powershell
uv run python -m src.core.vl_models.score_predictions <path>\p4c_holdout_predictions.jsonl
```

### 8.5 已知的環境陷阱

| 陷阱 | 處理 |
|---|---|
| `drive.mount()` 回 `credentials-propagation ... Bad Request` | 同一個 session 在瀏覽器也開一個 Colab 分頁，再重跑那格 |
| VS Code 對遠端 Colab kernel 的 Interrupt 不可靠 | 用 Colab 網頁分頁 → Runtime → Interrupt execution。**絕對不要按 Terminate／Disconnect**，那不是優雅關閉，開著的檔案不會上傳 |
| VS Code 不顯示 cell 編號 | 指路用 markdown 標題，不要用索引 |
| Drive FUSE 只在**關檔**時上傳 | 寫本機 `/content`，每批用 `shutil.copy` 覆蓋到 Drive |

---

## 9. 這一輪暴露的缺陷與教訓

### 9.1 held-out 差點與訓練集重疊
見 §3.2。**seed 差 1 ＝ 同一批資料位移一格。**

### 9.2 dry run 的計時投影完全不能用
5 步裡第一步吃掉全部編譯與 autotune 成本（推算約 170 秒），平均下來爆掉 6.5 倍。
**修法：丟掉第一步再平均，或把那個數字標成「上限」而非「預估」。**

### 9.3 batch size 往上調反而虧
圖是變動尺寸的，micro-batch 會被 padding 補到最長者。抽 600 張實測：

```
 micro-batch  padded tokens/sample    waste
           1                   621    0.0%
           2                   767   18.9%   ← 本輪採用
           4                   902   30.6%
           8                   993   37.2%
```

batch 2 已經浪費 18.9%，往上調要多付 18–29%，而 MFU 只有約 42%、賺得回來的有限。
**反而 `batch=1, accum=8` 可以把 padding 歸零**（代價是 GEMM 變瘦），值得用 lr=0 短跑實測。
標準解是 `group_by_length=True`，但與 `skip_prepare_dataset=True` ＋ lazy transform 能否共存**未驗證**。

### 9.4 ★ 評估用 batch 1 逐筆生成 — 本輪最大的成本錯誤
推論 3.0 CU > 訓練 2.4 CU，本末倒置。

機制：batch 1 自回歸解碼要把 9.16 GB 權重**每個 token 讀一遍**，L4 的 300 GB/s 給出
30.5 ms/token 的地板；輸出約 300 token ⇒ 理論 9.2 s，實際 34.5 s，**只跑到 roofline 的 27%**。
差額是每 token 約 80 ms 的固定開銷：344 個**未 merge** 的 LoRA adapter（每 token 多 688 次 kernel 發動）、
HF `generate` 迴圈的 Python 開銷、沒有 CUDA graph。**GPU 大部分時間在發呆。**

**修法**：① 批次生成 ② 推論前 merge LoRA ③ 或直接走匯出後的本機部署。

> 2026-08-29 的實測佐證：同一個模型 merge 後匯成 GGUF、由本機 Ollama 服務，
> **每張圖 3–6.5 秒**（Colab 上是 34.5 秒）。③ 的效果是可觀的。

### 9.5 ★ checkpoint 只留下最後兩個（本報告新發現）
`save_total_limit = 2` 會刪掉舊的 checkpoint。實際從 Drive 撈回來的只有
**`checkpoint-800` 與 `checkpoint-975`**，200／400／600 早就被自動刪除了。

**影響**：交接文件 §6 提議的那個實驗——「拿 checkpoint-200 對同一批 held-out 測，
若也滿分則代表 1,600 筆就夠、本輪 4/5 的訓練量是白付的」——**已經做不成**。
要回答那個問題只能重訓，而重訓在飽和的評估集上又量不出東西（§10）。

**教訓**：`save_total_limit` 是為了 Drive 空間，但它同時決定了**事後還能問哪些問題**。
如果 checkpoint 有分析價值，要嘛不設限，要嘛在刪除前把要留的另存一份。

### 9.6 一個文件錯誤
P4a notebook 的 markdown 寫「訓練沒有 `<think>` 區塊」，P4c 沿用了。實測是錯的（§5.1）。

---

## 10. 這些數字證明什麼、不證明什麼

**✅ 證明的**：微調把未微調時唯一的瓶頸（牆，F1 0.438）徹底解決，
在**我們自己的 renderer 畫出來的、模型沒看過的**盤面上端到端 200/200。

**❌ 這一輪不證明的**：看得懂 LinkedIn 的真實截圖。訓練與評估都在同一個 renderer 上，
**這是本人 2026-08-22 明示接受的取捨**（原本規劃的 P3「收 30–50 張真實截圖手工標註」已定案不做）。

> ★ **2026-08-29 補充：這個保留現在有反面證據了。** 模型匯出到本機後，順手在既有的
> **六張真實 LinkedIn 截圖**上量了一次（用 `finetune` prompt，無 few-shot，所以沒有洩題）：
> 逐格 **1.000**、號碼召回 **1.000**、牆 F1 **0.972**、端到端 **5/6**（未微調是 0.947／0.917／0.438／2/6），
> 而且**兩張 7×7 全對**——推翻了「訓練全是 6×6，看到 7×7 可能答成 6×6」的預期。
> 唯一的失敗是多幻覺一道牆造成無解，**被 `solvable` 旗標抓到，靜默錯誤 0 筆**。
> **但 n=6，撐不起「已驗證真實截圖」的結論**，完整討論見
> [`2026-08-29_vl-p4d-export-and-integration.md`](2026-08-29_vl-p4d-export-and-integration.md) §6。

**⚠ 評估集已經飽和。** 全部 1.000 ⇒ **這把尺再也量不出差異**。
後續任何改動——視覺層消融、CoD 變體、更少的訓練資料、batch 調整、**換一個模型家族**——
在這個集合上都會是 1.000，無法比較。要重獲鑑別力**只能把合成資料變難**：
更多視覺雜訊、多種 renderer 風格、模擬截圖的壓縮與縮放失真、更大盤面。
`render_puzzle.py` 與 `dataset_builder.py` 都吃參數，改動範圍不大。

**★ 一個仍然有用的副產品**：出題器先畫 Hamiltonian path 再挖題，所以**真實盤面必定有解**
⇒ **預測盤面無解就一定是讀錯了，不需要 ground truth 也知道**。這是可以直接出貨的信心旗標。
`score_predictions.py` 的 `solvable` / `solvable_but_wrong` 就是量這個。

而且**牆的兩種錯誤不對稱**：

| 錯誤 | 集合關係 | 後果 |
|---|---|---|
| 多幻覺牆（偽陽性） | 預測 ⊇ 真實 | 解出來的路**仍然合法**；風險是過度受限導致「無解」——**看得見的失敗** |
| 漏讀牆（偽陰性） | 預測 ⊉ 真實 | 解出來的路**可能穿牆，而且沒有任何跡象**——**靜默的錯誤答案** |

⇒ **出貨時 recall 比 precision 更要命。**

---

## 11. 產物位置

| 東西 | 位置 |
|---|---|
| LoRA adapter（148 MB，688 張量） | Google Drive `colab_finetune/p4c_qwen35_4b_zip_lora`；**2026-08-29 已撈回本機** `models/colab_finetune/`（不進版控） |
| checkpoint | 只剩 **800／975**（§9.5） |
| 200 筆原始預測 | [`artifacts/vl-p4c/p4c_holdout_predictions.jsonl`](artifacts/vl-p4c/p4c_holdout_predictions.jsonl) |
| 算分結果 | [`artifacts/vl-p4c/p4c_holdout_predictions_scored.json`](artifacts/vl-p4c/p4c_holdout_predictions_scored.json) |
| 訓練資料集 | 本機 `datasets/vl/main_6x6/`（不進版控）＋ Drive 的 tar |
| 執行存證（含 loss 表與全部輸出） | [`../../notebooks/p4c_finetune_8000.ipynb`](../../notebooks/p4c_finetune_8000.ipynb) |

2026-08-29 撈回本機後的完整性驗證（**兩項都對得上**）：

```
訓練資料 tar     MATCH  69c753e16d030ecba7a7da505f046b747e275ec7ca956c0bada80a6d7ce70fbf
200 筆預測      MATCH  fc96cb30770ba2226012fa7da5fa63732afd6b7c3001df2d0e843ef103694e22
adapter 張量    688（visual 96 對 / language 248 對，全 F32）→ 與訓練當下印出的 96/96、248/248 吻合
```
