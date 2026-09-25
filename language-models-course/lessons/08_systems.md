# 第 08 課：GPU 與系統——算術強度、FlashAttention、混合精度、平行化

> 前置：第 04、07 課 ｜ 實驗：[`notebooks/08_systems.ipynb`](../notebooks/08_systems.ipynb) ｜ 預估時間：2.5 小時

> 本課是 CS336 的重心之一（第 5–8 講、作業 2），但那些內容需要 GPU 才能實作（Triton kernel、NCCL 通訊）。這裡在 CPU 上做得到的部分是：量出 roofline 的形狀、驗證 FlashAttention 演算法的正確性、看數值格式的精度、模擬資料平行。GPU 的實作細節請看延伸閱讀。

## 這一課要回答的問題

- 為什麼「FLOPs 一樣」的運算，花的時間可以差幾十倍？
- FlashAttention 沒有少算任何東西，為什麼更快、更省記憶體？
- bf16、fp16、fp8 差在哪裡？為什麼 fp16 訓練要「loss scaling」而 bf16 不用？
- 一個模型放不進一張卡、或一張卡太慢時，有哪些切法？各自要付出什麼通訊成本？

---

## 1. 白話版

### 1.1 算得快，搬得慢

GPU 有上萬個計算單元，但資料放在「大而慢」的主記憶體（HBM）裡，要先搬到「小而快」的晶片上記憶體（SRAM）才能算。很多運算（例如把兩個向量逐元素相加）每搬一個數字只算一次，時間都花在搬運上——這叫**記憶體受限**。大的矩陣乘法每個數字會被重複用很多次，才能讓計算單元忙起來——這叫**計算受限**。

### 1.2 FlashAttention：不要把大表寫出去

一般的 attention 會先算出 $T \times T$ 的分數表，寫回主記憶體，再讀回來做 softmax、再寫、再讀來乘 $V$。$T = 8{,}000$ 時這張表有 6,400 萬格，每個 head 一張。FlashAttention 把 $Q$、$K$、$V$ 切成小塊，**每一塊在晶片上算完 softmax 與加權和就丟掉**，只留下答案。訣竅是 softmax 可以「邊讀邊算」（online softmax）：看到更大的數字時，把之前的累計值乘一個修正係數。

### 1.3 少用幾個位元

每個數字用 32 位元存，精確但佔空間、算得慢。用 16 位元（bf16）可以快一倍、省一半記憶體，代價是精度只有約 3 位有效數字。技巧在於：權重的「正本」保留 32 位元，計算用 16 位元，累加用 32 位元。

### 1.4 很多張卡一起算

- **資料平行**：每張卡一份完整的模型，各算一部分資料，最後把梯度平均。
- **切模型狀態**（ZeRO／FSDP）：每張卡只保管一部分的權重、梯度、最佳化器狀態，要用時再向別人借。
- **切每個矩陣**（tensor parallel）、**切層**（pipeline parallel）、**切序列**（context parallel）。

每一種都在「記憶體」、「計算」、「卡與卡之間的通訊」之間取捨。

---

## 2. 正式版

### 2.1 Roofline 模型

**算術強度**（arithmetic intensity）＝ 一個運算的 FLOPs ÷ 它搬動的位元組數。可達到的效能：

$$\text{FLOP/s} = \min\left(\text{峰值 FLOP/s},\ \text{頻寬} \times I\right) \tag{8.1}$$

轉折點在 $I^* = \text{峰值} / \text{頻寬}$。H100 SXM：約 989 TFLOPS（bf16 dense）÷ 3.35 TB/s ≈ 295 FLOP/byte（NVIDIA 規格頁，查證 2026-09-25）。

| 運算 | FLOPs | 搬動的 bytes（bf16） | 強度 |
|---|---|---|---|
| 向量相加 $y = a + b$，長度 $n$ | $n$ | $6n$ | $1/6$ |
| 矩陣乘法 $n \times n$ 乘 $n \times n$ | $2n^3$ | $6n^2$ | $n/3$ |
| 矩陣–向量（推論的 decode） | $2n^2$ | $\approx 2n^2$ | $\approx 1$ |

所以 LayerNorm、GELU、softmax、殘差相加這類逐元素運算幾乎永遠記憶體受限；**kernel fusion**（把一串逐元素運算合成一個 kernel，中間值不寫回 HBM）是加速它們的主要方法——`torch.compile` 與手寫的 Triton kernel 都在做這件事。推論的 decode 階段（每步只處理一個 token）是矩陣–向量乘法，強度約 1，完全記憶體受限（第 09 課）。

### 2.2 Online softmax

$\mathrm{softmax}(x)_i = e^{x_i - m} / \sum_j e^{x_j - m}$，$m = \max_j x_j$（減去最大值避免溢位）。分塊讀入時維護目前的最大值 $m$ 與累計和 $\ell$：

$$m' = \max(m, \max_{j \in \text{新塊}} x_j), \qquad \ell' = \ell\, e^{m - m'} + \sum_{j \in \text{新塊}} e^{x_j - m'} \tag{8.2}$$

讀完所有塊時 $\ell$ 就是正確的分母（Milakov & Gimelshein 2018）。

### 2.3 FlashAttention

對一塊 query $Q_b$，依序處理 key／value 的塊 $K_j, V_j$，維護每列的 $m$、$\ell$ 與未正規化的輸出 $O$：

$$S = Q_b K_j^\top / \sqrt{d}, \quad m' = \max(m, \mathrm{rowmax}(S)), \quad P = e^{S - m'}, \quad \ell' = \ell\,e^{m - m'} + \mathrm{rowsum}(P), \quad O' = O\,e^{m - m'} + P V_j \tag{8.3}$$

最後輸出 $O / \ell$，並存下每列的 $\log$-sum-exp $m + \log \ell$ 供 backward 使用。

- **結果完全相同**（不是近似），只是運算順序不同。
- **記憶體**：不存 $T \times T$ 的 $S$、$P$，從 $O(T^2)$ 降到 $O(T)$。
- **backward**：從 $Q$、$K$ 與存下的 log-sum-exp **重算** $S$、$P$——多算一次，但省下大量記憶體讀寫，整體更快。
- FlashAttention-2 改善平行化與工作分配；FlashAttention-3 針對 H100 的非同步執行與 FP8。nanochat 在支援的 GPU 上用 FA3，否則退回 PyTorch 的 `scaled_dot_product_attention`。

### 2.4 數值格式

| 格式 | 符號／指數／尾數 | 範圍（約） | 相對精度（約） | 用途 |
|---|---|---|---|---|
| fp32 | 1／8／23 | $10^{\pm 38}$ | $10^{-7}$ | 主權重、最佳化器狀態、累加 |
| bf16 | 1／8／7 | $10^{\pm 38}$ | $4 \times 10^{-3}$ | 現代訓練的預設計算格式 |
| fp16 | 1／5／10 | $6 \times 10^{-5}$ ～ $6.5 \times 10^{4}$ | $5 \times 10^{-4}$ | 舊 GPU；需要 loss scaling |
| fp8 e4m3／e5m2 | 1／4／3、1／5／2 | 很小 | 很粗 | H100 起的矩陣乘法；需要每張量的縮放 |

- **bf16 與 fp32 的指數位數相同**，範圍一樣大，只是精度低；所以不會上溢或下溢，不需要 loss scaling。
- **fp16 範圍小**：很小的梯度會變成 0（下溢）。**loss scaling**（Micikevicius 等人 2017）：先把 loss 乘一個大數再 backward，梯度一起放大，更新前再除回來。
- **混合精度**：權重的 fp32 正本給最佳化器用，forward／backward 用 bf16；矩陣乘法的累加在硬體內部用 fp32。nanochat 不用 `autocast`，而是明確指定一個全域的 `COMPUTE_DTYPE`：權重存 fp32，自訂的 `Linear` 在 forward 時轉成計算精度；speedrun 的矩陣乘法用 fp8。

### 2.5 資料平行（DP／DDP）

$n$ 張卡各有一份模型，各處理 $B/n$ 筆資料，梯度做 all-reduce（平均）：

$$\frac{1}{n}\sum_{k=1}^{n} \nabla \frac{1}{B/n}\sum_{i \in \text{shard}_k} \ell_i = \nabla \frac{1}{B}\sum_{i=1}^{B}\ell_i \tag{8.4}$$

**數學上和單卡大 batch 完全相同**（Lab 08 §5 驗證）。代價是每步的通訊：ring all-reduce 每張卡約傳送 $2 \times$ 參數大小的資料，可與 backward 重疊。

### 2.6 切分模型狀態：ZeRO 與 FSDP

第 07 課 §2.2：每個參數 16 bytes。ZeRO 讓 $n$ 張卡分攤：stage 1 分最佳化器狀態、stage 2 加上梯度（用 reduce-scatter 取代 all-reduce）、stage 3 加上權重（每層用之前 all-gather）。PyTorch 的 FSDP 相當於 stage 3。
nanochat 不用 PyTorch 的 DDP，而是在自己的 `MuonAdamW` 裡做 ZeRO-2 式的分攤：梯度 reduce-scatter、每張卡只更新並保存自己那一片的最佳化器狀態，再 all-gather 更新後的權重（查證 2026-09-25，`nanochat/optim.py` 的說明）。

### 2.7 模型平行

- **Tensor parallel**（Megatron-LM，Shoeybi 等人 2019）：把 MLP 的第一個矩陣按欄切、第二個按列切，每層只需要一次 all-reduce；attention 按 head 切。通訊頻繁，需要節點內的高速互連（NVLink）。
- **Pipeline parallel**：不同的卡負責不同的層，batch 切成 $m$ 個 micro-batch 流水線式地前進。$p$ 段流水線的閒置（bubble）比例約

$$\text{bubble} \approx \frac{p - 1}{m + p - 1} \tag{8.5}$$

所以 micro-batch 要遠多於段數。

- **Context／sequence parallel**：長序列把 $T$ 切給多張卡，attention 用 ring 的方式交換 $K$、$V$（Ring Attention）。
- **Expert parallel**：MoE 模型把不同的 expert 放在不同的卡（第 15 課）。

大規模訓練把這些組合起來（Llama 3 的技術報告稱為 4D 平行：tensor、context、pipeline、資料平行）。經驗法則：**tensor parallel 留在節點內，pipeline 與資料平行跨節點**。

### 2.8 用算力換記憶體：activation checkpointing

只存每個區塊的輸入，backward 時重跑該區塊的 forward 取回中間值。activation 記憶體從「每層所有中間值」降到「每層一個張量」，代價約多一次 forward（總算力 +33%）。可以選擇性地只重算便宜的部分（Korthikanti 等人 2022）。

---

## 3. 對照程式碼

| 概念 | 位置（`src/lm_course/kernels.py`） |
|---|---|
| (8.1) 實測：矩陣乘法的 FLOP/s 與強度、逐元素運算的頻寬 | `matmul_throughput`、`elementwise_bandwidth`、`time_it` |
| (8.2) online softmax | `online_softmax` |
| (8.3) FlashAttention 的 forward（純 PyTorch、分塊、causal） | `tiled_attention`（回傳輸出與 log-sum-exp） |
| naive attention 的分數表大小 | `attention_score_bytes` |
| (8.4) 資料平行 | `data_parallel_gradients` |
| 第 07 課的記憶體帳 | `scaling.zero_memory_per_gpu`、`scaling.saved_activation_bytes` |

測試：`test_online_softmax_is_exact`、`test_tiled_attention_is_exact`（含 log-sum-exp）、`test_data_parallel_average_equals_full_batch_gradient`。

## 4. 常見誤解

- **「GPU 越快，訓練就越快」**：記憶體受限的運算只跟頻寬有關；小模型、小 batch 的訓練常常用不到峰值的一成。
- **「FlashAttention 是近似的 attention」**：它是精確的，只是換了計算順序與存放位置。
- **「bf16 比 fp16 精確」**：相反，bf16 的尾數更少、更不精確；它的優點是範圍大。
- **「資料平行會改變訓練的數學」**：在 mean loss 與等大分片下完全不會（式 8.4）；會改變的是「全域 batch size 變大」這件事本身，可能需要調學習率。

## 5. 練習

**想一想**

1. 計算 $n = 4096$ 的 bf16 矩陣乘法的算術強度。H100 上它是計算受限還是記憶體受限？$n = 64$ 呢？
2. 從 (8.2) 證明：讀完所有塊後 $\ell = \sum_j e^{x_j - m}$，$m$ 是全域最大值。
3. 在 fp16 裡，$10^{-8}$ 會變成什麼？loss scale 取 $2^{16}$ 時呢？
4. $p = 8$ 段流水線、$m = 32$ 個 micro-batch，bubble 佔多少？要壓到 5% 以下需要多少 micro-batch？
5. 7B 模型在 8 張卡上做 tensor parallel（8 路），每一層的 MLP 需要什麼通訊？

**動手改**（在 `08_systems.ipynb`）

6. 把 roofline 實驗改用 bf16，CPU 上會比較快嗎？為什麼？
7. 把 `tiled_attention` 的塊大小從 16 掃到 256，時間怎麼變？和 PyTorch 的 fused kernel 比差多少？
8. 模擬 fp16 訓練：把梯度轉成 fp16 再轉回來，有多少比例變成 0？加上 loss scaling 呢？

## 6. 延伸閱讀

- Dao 等人（2022），FlashAttention：<https://arxiv.org/abs/2205.14135>；FlashAttention-2：<https://arxiv.org/abs/2307.08691>；FlashAttention-3：<https://arxiv.org/abs/2407.08608>
- Milakov & Gimelshein（2018），online softmax：<https://arxiv.org/abs/1805.02867>；Rabe & Staats（2021），〈Self-attention Does Not Need $O(n^2)$ Memory〉：<https://arxiv.org/abs/2112.05682>
- Micikevicius 等人（2017），〈Mixed Precision Training〉：<https://arxiv.org/abs/1710.03740>
- Rajbhandari 等人（2019），ZeRO：<https://arxiv.org/abs/1910.02054>
- Shoeybi 等人（2019），Megatron-LM：<https://arxiv.org/abs/1909.08053>
- Liu 等人（2023），Ring Attention：<https://arxiv.org/abs/2310.01889>
- CS336 第 5 講（GPU）、第 6 講（kernel、Triton）、第 7–8 講（平行化）、作業 2（自己寫 FlashAttention-2 的 Triton 版本、分散式訓練）：<https://cs336.stanford.edu/>、<https://github.com/stanford-cs336/assignment2-systems>
- NVIDIA H100 規格：<https://www.nvidia.com/en-us/data-center/h100/>
