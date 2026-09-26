# 第 07 課：算力、記憶體與 scaling laws

> 前置：第 06 課 ｜ 實驗：[`notebooks/07_scaling.ipynb`](../notebooks/07_scaling.ipynb) ｜ 預估時間：2.5 小時

## 這一課要回答的問題

- 訓練一個模型要多少 FLOPs？「$6ND$」從哪裡來？
- 訓練時記憶體花在哪裡？一個 7B 模型為什麼一張 80 GB 的 GPU 放不下？
- 給定算力預算，模型該多大、資料該多少？Kaplan 與 Chinchilla 的結論為什麼不同？
- 為什麼 2024 年後的小模型都「過度訓練」（遠超過 20 token／參數）？

---

## 1. 白話版

### 1.1 先算帳再花錢

訓練大模型要花幾百萬美元，不可能每種設定都試一次。好消息是：**loss 隨模型大小、資料量、算力的變化非常規律**——畫在 log-log 圖上幾乎是直線。所以可以先用一堆小模型做實驗，量出這條線，再外推到大模型。這就是 scaling laws。

### 1.2 大模型還是多資料？

算力固定時，有兩種花法：做一個大模型、只看少量資料；或做一個小模型、看大量資料。太大的模型還沒學完就停了，太小的模型再怎麼看資料也裝不下。中間有一個最佳點。

- 2020 年 Kaplan 等人的結論：大部分預算給模型大小。GPT-3（175B）只訓練了 3,000 億 token。
- 2022 年 DeepMind 的 Chinchilla：模型和資料應該**等比例**成長，大約每個參數配 20 個 token。70B 的 Chinchilla 用 1.4 兆 token，贏過 4 倍大的 Gopher。

### 1.3 但訓練不是全部

模型訓練一次，卻要被使用幾十億次。推論的成本只和模型大小有關，所以現在的做法常是：**故意用比「最省訓練算力」更小的模型，餵遠多於 20 倍的資料**——訓練多花一點，換來每次使用都便宜。

---

## 2. 正式版

### 2.1 FLOPs

矩陣乘法 $(m \times k)(k \times n)$ 要 $2mkn$ 次浮點運算（乘與加各一）。一個有 $N$ 個「參與矩陣乘法的參數」的模型，每個 token：

- forward：每個參數一次乘加，$2N$；
- backward：對輸入的梯度 $2N$、對權重的梯度 $2N$，共 $4N$。

$$C_{\text{train}} \approx 6ND \tag{7.1}$$

$D$ 是訓練 token 數。attention 還有一項和 context 長度 $T$ 有關（$QK^\top$ 與 $AV$ 兩個矩陣乘法，forward $4dT$、加上 backward 共 $12dT$，每層每 token）：

$$C_{\text{train}} \approx \left(6N + 12\,L\,d\,T\right)D \tag{7.2}$$

（PaLM 附錄 B 的算法，nanochat 的 `estimate_flops` 也這樣算；因為 causal mask，實際只需一半，但通常照全額計。）對 $L$ 層、$12Ld^2$ 參數的模型，attention 項與 $6N$ 的比約為 $T / (6d)$：$T$ 比 $6d$ 小很多時可以忽略，長 context 時不行。

**MFU**（model FLOPs utilization）：

$$\text{MFU} = \frac{6N \times \text{每秒處理的 token 數}}{\text{硬體峰值 FLOP/s}} \tag{7.3}$$

以 H100 SXM 為例，NVIDIA 標示的 BF16 峰值是 1,979 TFLOPS（含結構化稀疏；dense 約一半，989 TFLOPS）。好的大規模訓練 MFU 約 40–50%。

### 2.2 記憶體

**模型狀態**（每個參數）：

| 設定 | 權重 | 梯度 | 最佳化器 | 合計 |
|---|---|---|---|---|
| fp32 ＋ AdamW | 4 | 4 | 8（$m$、$v$） | 16 bytes |
| 混合精度 ＋ AdamW | 2（bf16） | 2 | 12（fp32 主權重＋$m$＋$v$） | 16 bytes |

所以 7B 模型光是模型狀態就要 $7 \times 10^9 \times 16 = 112$ GB，一張 80 GB 的 GPU 放不下——還沒算 activation。

**Activation**：forward 時要存下供 backward 使用的中間值，量級約 $B \times T \times d \times L \times (\text{常數})$，再加上 attention 分數的 $B \times H \times T^2$（FlashAttention 不存它，第 08 課）。和 batch、context、深度都成正比，常常比模型狀態還大。對策：**activation checkpointing**（backward 時重算，約多花 1/3 的 forward 算力）、縮小 micro-batch 搭配梯度累積。

**分散式時的分攤**（ZeRO，Rajbhandari 等人 2019）：資料平行的 $n$ 張卡上，stage 1 分攤最佳化器狀態、stage 2 再分攤梯度、stage 3 再分攤權重（等同 PyTorch 的 FSDP），每張卡的模型狀態從 $16N$ 降到最低 $16N/n$（第 08 課）。

### 2.3 Kaplan 等人（2020）的 scaling laws

用非 embedding 參數 $N$、token 數 $D$、算力 $C$，在「只有一個因素受限」時：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N},\ \alpha_N \approx 0.076; \qquad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D},\ \alpha_D \approx 0.095; \qquad L(C_{\min}) \propto C_{\min}^{-0.050} \tag{7.4}$$

並推出最佳模型大小 $N_{\text{opt}} \propto C^{0.73}$：算力加 10 倍，模型要大 5.5 倍、資料只要多 1.8 倍。

### 2.4 Chinchilla（Hoffmann 等人 2022）

訓練 400 多個模型（7,000 萬到 160 億參數、50 億到 5,000 億 token），用三種方法估計最佳配置：

1. 固定模型大小、變化訓練長度，取每個算力的最低 loss；
2. **IsoFLOP**：固定算力 $C$，變化模型大小（$D = C / 6N$），找 loss 最低的 $N$（§2.5）；
3. 直接擬合參數化的 loss：

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}, \quad E = 1.69,\ A = 406.4,\ B = 410.7,\ \alpha = 0.34,\ \beta = 0.28 \tag{7.5}$$

三種方法都得到 $N_{\text{opt}} \propto C^{a}$、$D_{\text{opt}} \propto C^{b}$，$a \approx b \approx 0.5$：**模型與資料等比例成長**，經驗值約 20 token／參數：

$$N_{\text{opt}} \approx \sqrt{\frac{C}{6 \times 20}}, \qquad D_{\text{opt}} \approx 20\,N_{\text{opt}} \tag{7.6}$$

(7.5) 的 $E$ 是「不可約的 loss」——第 01 課式 (1.3) 的語言本身的熵（在這份資料與 tokenizer 下）。

**為什麼和 Kaplan 不同？** Chinchilla 論文自己的猜測是：Kaplan 對所有模型用同樣的訓練長度與學習率排程，小模型的學習率沒有降完。後續研究給了更完整的解釋：

- Pearce & Song（2024）：主要原因是 Kaplan 只算**非 embedding 參數**，而且實驗規模小；在這兩個條件下模擬 Chinchilla 的研究，會得到接近 Kaplan 的係數。
- Porian 等人（2024）：重現 Kaplan 的結果後找出三個因素——**最後一層（LM head）的算力沒算進去、暖身長度、沒有隨規模調整最佳化器的超參數**；修正後與 Chinchilla 吻合。他們也發現，和 Chinchilla 的猜測相反，仔細的學習率遞減並不是關鍵。

結論：用**全部參數與全部算力**來計，Chinchilla 的係數站得住。

### 2.5 IsoFLOP 怎麼做

1. 選幾個算力預算 $C_1 < C_2 < \dots$；
2. 每個 $C_i$ 訓練一系列不同大小的模型，每個都用完整的學習率排程（訓練 $D = C_i / 6N$ 個 token）；
3. 對每個 $C_i$，在 $\log N$ 上擬合一條拋物線，取最低點 $N^*(C_i)$；
4. 在 log-log 上擬合 $N^*(C) = kC^a$。

Lab 07 在 CPU 上用 $3 \times 10^{12}$–$3 \times 10^{13}$ FLOPs 的預算（比 GPT-3 的約 $3 \times 10^{23}$ 小 10 個數量級）做一次。

### 2.6 算力最佳以外的考量

- **推論成本**：Llama 3 的 8B 模型訓練了約 15 兆 token，每參數約 1,900 個 token，遠超過 20。訓練時「浪費」的算力，換來部署時便宜。
- **資料有限**：高品質資料不夠時，重複 epoch（約 4 次以內影響很小，Muennighoff 等人 2023）或用合成資料。
- **nanochat 的選擇**：`--target-param-data-ratio` 預設 12（它自己的 scaling 實驗結果，參數的計數方式與 Chinchilla 不同），speedrun 為了更快達標用 8（查證 2026-09-25）。
- **超參數外推**：大模型不能逐一調參。μP（Yang 等人 2022）讓最佳學習率在不同寬度間轉移；nanochat 用較簡單的 $1/\sqrt{d}$ 縮放。

### 2.7 loss 可以預測，能力呢？

loss 隨規模平滑下降，但下游任務的「正確率」有時看起來是突然出現（emergent）。Schaeffer 等人（2023）指出，很多「突然」來自指標本身不連續（例如完全正確才算分）；改用連續的指標，曲線多半也是平滑的。

---

## 3. 對照程式碼

| 概念 | 位置（`src/lm_course/scaling.py`） |
|---|---|
| 參與矩陣乘法的參數 $N$ | `matmul_parameters` |
| (7.1)(7.2) | `training_flops_per_token`；用 PyTorch 的 FlopCounterMode 實測：`measured_training_flops` |
| (7.6) | `chinchilla_optimal` |
| §2.2 模型狀態、ZeRO、KV cache | `BYTES_PER_PARAM`、`zero_memory_per_gpu`、`kv_cache_bytes` |
| activation 實測 | `saved_activation_bytes`（用 `saved_tensors_hooks` 加總 autograd 存下的張量） |
| 擬合 | `fit_power_law`、`fit_isoflop_minimum` |

測試：`test_six_n_matches_the_flop_counter`、`test_attention_term_is_twelve_d_t_per_token_and_layer`、`test_activation_memory_grows_linearly_with_batch`、`test_power_law_and_isoflop_fits_recover_known_curves`。

## 4. 常見誤解

- **「Chinchilla 說每個模型都該訓練 20 倍 token」**：那是「給定訓練算力、要最低 loss」的答案；考慮推論成本時最佳點不同。
- **「scaling laws 是自然定律」**：它們是在特定資料、架構、tokenizer、訓練設定下的經驗擬合；換了任何一項，係數都會變。
- **「FLOPs 決定訓練時間」**：還要看 MFU；小模型、小 batch、記憶體受限的運算都會讓實際速度遠低於峰值（第 08 課）。
- **「參數量就是記憶體用量」**：訓練時模型狀態是參數量的 16 倍 bytes，再加上 activation。

## 5. 練習

**想一想**

1. 為什麼 backward 的 FLOPs 大約是 forward 的 2 倍？
2. GPT-3（175B，3,000 億 token）用了多少訓練 FLOPs？照 (7.6)，同樣的算力應該訓練多大的模型、多少 token？
3. 一個 $d = 4096$、$T = 8192$ 的模型，attention 項佔 (7.2) 的幾成？
4. 7B 模型在 8 張 80 GB GPU 上用 ZeRO stage 1、2、3，每張卡的模型狀態各是多少？
5. 用 (7.5) 算 $N = 70\text{B}$、$D = 1.4\text{T}$ 的預測 loss，和 $N = 280\text{B}$、$D = 300\text{B}$ 比較。

**動手改**（在 `07_scaling.ipynb`）

6. 加一個更大的預算，擬合出的 $N^*(C)$ 指數會怎麼變？
7. 把參數量改成「不含 LM head」來算 $6ND$，IsoFLOP 的最佳點會怎麼移動？（Kaplan 與 Chinchilla 差異的來源之一。）
8. 量不同 context 長度下的 activation 記憶體，和 $T$ 是線性還是平方關係？

## 6. 延伸閱讀

- Kaplan 等人（2020），〈Scaling Laws for Neural Language Models〉：<https://arxiv.org/abs/2001.08361>
- Hoffmann 等人（2022），〈Training Compute-Optimal Large Language Models〉（Chinchilla）：<https://arxiv.org/abs/2203.15556>
- Porian 等人（2024），〈Resolving Discrepancies in Compute-Optimal Scaling of Language Models〉：<https://arxiv.org/abs/2406.19146>；Pearce & Song（2024）：<https://arxiv.org/abs/2406.12907>
- Chowdhery 等人（2022），PaLM（附錄 B 的 FLOPs 與 MFU）：<https://arxiv.org/abs/2204.02311>
- Rajbhandari 等人（2019），ZeRO：<https://arxiv.org/abs/1910.02054>
- Korthikanti 等人（2022），activation 記憶體與重算：<https://arxiv.org/abs/2205.05198>
- Yang 等人（2022），μP：<https://arxiv.org/abs/2203.03466>
- Schaeffer 等人（2023），〈Are Emergent Abilities of Large Language Models a Mirage?〉：<https://arxiv.org/abs/2304.15004>
- CS336 第 2 講（資源計算）、第 9 與 11 講（scaling laws）、作業 3：<https://cs336.stanford.edu/>
- nanochat 的 `runs/scaling_laws.sh` 與〈miniseries v1〉：<https://github.com/karpathy/nanochat/discussions/420>
