# 第 09 課：推論——KV cache、批次、speculative decoding、量化

> 前置：第 05–08 課 ｜ 實驗：[`notebooks/09_inference.ipynb`](../notebooks/09_inference.ipynb)（需要 Lab 06 的模型） ｜ 預估時間：2 小時

## 這一課要回答的問題

- 生成文字為什麼慢？KV cache 省下了什麼、又花掉了什麼？
- 為什麼處理 prompt（prefill）很快、一個字一個字吐（decode）很慢？為什麼伺服器要把很多使用者的請求湊成一批？
- speculative decoding 讓小模型「先猜」，為什麼結果的分布和大模型**完全一樣**？
- 把權重壓成 8 位元、4 位元，會掉多少品質？為什麼要「每一列各自一個縮放係數」？

---

## 1. 白話版

### 1.1 不要重算已經算過的東西

生成時每一步只多一個字。但如果每一步都把整段文字重新丟進模型，第 1,000 步就要重算前面 999 個字——而它們的結果根本沒變。**KV cache** 把每一層每個字的 key、value 存起來，新的字只需要算自己，再去「查詢」存起來的 key／value。

代價是記憶體：每個字、每一層都要存兩個向量。context 長、同時服務的使用者多時，cache 比模型本身還大。

### 1.2 兩種工作：讀 prompt 與吐字

- **prefill**：整段 prompt 一次丟進去，大矩陣乘法，GPU 很忙——計算受限。
- **decode**：每次只有一個新字，但要把**全部權重**從記憶體讀一遍才能算出它——記憶體受限，GPU 大多在等資料。

所以伺服器會把很多使用者的 decode 湊成一批：讀一次權重，同時服務幾十個人。

### 1.3 先讓小模型猜

大模型一次只能吐一個字，但它「檢查」一串字和吐一個字花的時間差不多（都是讀一次權重）。**speculative decoding**：讓一個小模型快速猜 4 個字，大模型一次檢查這 4 個——同意的就收下，不同意的地方換成大模型自己的選擇。巧妙的接受規則讓最後的結果和「大模型自己一個一個生成」在機率上完全一樣。

### 1.4 用更少的位元存權重

decode 慢是因為要讀權重；權重變小，讀得就快。把每個 32 位元的數字壓成 8 位元或 4 位元的整數，再配一個縮放係數。每一列（每個輸出神經元）各自一個係數，才不會被少數特別大的數字拖累。

---

## 2. 正式版

### 2.1 KV cache

不用 cache 時，生成第 $t$ 個 token 要對長度 $t$ 的序列做一次 forward，生成 $T$ 個 token 共處理 $O(T^2)$ 個 token 位置。用 cache 時每步只處理 1 個新 token，attention 讀取 $t$ 個已存的 $K$、$V$：矩陣乘法的計算降到 $O(T)$，attention 仍是 $O(T^2)$ 但每步只是一個向量對矩陣。

**大小**：

$$\text{KV bytes} = 2 \times L \times H_{kv} \times d_h \times T \times B \times \text{bytes per value} \tag{9.1}$$

例：$L = 32$、$d = 4096$、32 個 head（$d_h = 128$）的 7B 級模型，bf16 時每個 token 需要 $2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ bytes ≈ 0.5 MB；4,096 token 的 context 就是 2 GB／條序列。改成 GQA 的 8 組 KV head，縮小 4 倍。
縮小 cache 的方法：GQA／MQA（第 04 課）、DeepSeek-V2 的 MLA（把 $K$、$V$ 壓成低維的潛在向量再存）、滑動視窗、把 cache 量化。

### 2.2 prefill 與 decode 的算術強度

decode 時 batch 為 $B$：每層是 $(B \times d)(d \times d')$ 的乘法，FLOPs $2Bdd'$、讀取權重 $2dd'$ bytes（bf16），強度約 $B$ FLOP/byte。H100 的轉折點約 295（第 08 課），所以 batch 要上百才會變成計算受限。batch 1 時，每個 token 的延遲下限大約是

$$t_{\text{token}} \gtrsim \frac{\text{權重 bytes} + \text{KV cache bytes}}{\text{記憶體頻寬}} \tag{9.2}$$

7B 模型 bf16 約 14 GB，在 3.35 TB/s 下約 4 ms／token——不管算力多強。prefill 則是 $(T \times d)$ 的矩陣乘法，$T$ 大時就是計算受限。

**伺服器的做法**：continuous batching（請求隨時加入、結束就離開批次）；PagedAttention（vLLM）把 KV cache 切成固定大小的「頁」管理，減少碎片、讓共用前綴的請求共享頁面。

### 2.3 Speculative decoding

目標模型 $p$、草稿模型 $q$。一輪：

1. 草稿模型依序抽 $x_1, \dots, x_k \sim q$；
2. 目標模型**一次** forward 得到 $p(\cdot \mid \text{prefix}, x_{<i})$，$i = 1..k+1$；
3. 依序對 $i = 1..k$：以機率 $\min\left(1, \frac{p(x_i)}{q(x_i)}\right)$ 接受 $x_i$；第一次拒絕時，改從

$$p_{\text{res}}(x) = \frac{\max\left(0,\ p(x) - q(x)\right)}{\sum_{x'}\max\left(0,\ p(x') - q(x')\right)} \tag{9.3}$$

抽一個 token，結束這一輪；全部接受則從 $p(\cdot \mid \dots, x_k)$ 再抽一個（免費的第 $k+1$ 個）。

**為什麼分布不變**：對第一個位置，輸出為 $x$ 的機率是「草稿抽到 $x$ 且被接受」加上「被拒絕後從殘差抽到 $x$」：

$$q(x)\min\left(1, \tfrac{p(x)}{q(x)}\right) + \Big(1 - \sum_{x'}\min(p(x'), q(x'))\Big)\,p_{\text{res}}(x) = \min(p(x), q(x)) + \max(0, p(x) - q(x)) = p(x)$$

（第二項的係數正好等於殘差的正規化常數。）之後的位置同理，條件在前面已接受的 token 上。

**效益**：若每個草稿 token 被接受的機率約為 $\alpha$（$= \sum_x \min(p, q)$ 的期望），每次呼叫目標模型平均產生

$$\frac{1 - \alpha^{k+1}}{1 - \alpha} \tag{9.4}$$

個 token（Leviathan 等人 2022）。草稿越像目標、越便宜，加速越多。變形：Medusa（在目標模型上加幾個預測頭）、EAGLE（用目標模型的特徵做草稿）；DeepSeek-V3 訓練時的多 token 預測也可拿來做推論加速。

### 2.4 量化

**Absmax、每列一個係數**（對權重 $W$ 的第 $r$ 列）：

$$s_r = \frac{\max_j |W_{rj}|}{2^{b-1} - 1}, \qquad Q_{rj} = \mathrm{round}\left(\frac{W_{rj}}{s_r}\right), \qquad \hat W_{rj} = s_r Q_{rj} \tag{9.5}$$

誤差最多半個刻度 $s_r / 2$。每少一個位元，刻度加倍、平方誤差變 4 倍。**為什麼要每列一個係數**：若整個矩陣共用一個 $s$，只要有一個特別大的權重，其他數字就只能用到很少幾個刻度。

更進一步的方法：

- **LLM.int8()**（Dettmers 等人 2022）：activation 裡有少數維度的數值特別大（outlier features），把它們留在 16 位元、其餘用 8 位元。
- **GPTQ**（Frantar 等人 2022）：一次量化一欄，用二階資訊把誤差補償到還沒量化的欄位；4 位元幾乎無損。
- **AWQ**（Lin 等人 2023）：依 activation 的大小找出重要的權重通道並保護它們。
- 本機推論常見 4 位元的權重量化（例如 llama.cpp 的 GGUF 格式）；KV cache 也可以量化。

### 2.5 nanochat 的 Engine

nanochat 的 `engine.py` 把以上兩件事放在一起：預先配置的 KV cache、批次取樣（同一個 prompt 抽多個樣本，第 12 課的 RL 要用）、以及一個工具使用的狀態機——模型吐出 `<|python_start|>` 時開始收集程式碼，吐出 `<|python_end|>` 時執行，把結果以 `<|output_start|>…<|output_end|>` 強制插回序列（第 11 課）。

---

## 3. 對照程式碼

| 概念 | 位置 |
|---|---|
| (9.1) KV cache | `model.KVCache`（每層預先配置，存 RoPE 之後的 $K$）；`GPT.new_kv_cache`；公式 `scaling.kv_cache_bytes` |
| prefill ＋ decode 的生成迴圈、同一 prompt 的批次取樣 | `sampling.generate(..., use_cache=True, num_samples=...)` |
| (9.3) speculative decoding | `sampling.speculative_step`、`sampling.speculative_generate`、`SpeculativeStats` |
| (9.5) 量化 | `quantization.quantize_absmax`、`fake_quantize`、`quantize_model`、`weight_bytes` |
| 工具使用的推論引擎 | `chat.ChatEngine`（第 11 課） |

測試：`test_kv_cache_matches_full_forward`（prefill、分段、逐一 decode 三種用法）、`test_generation_with_and_without_cache_is_identical`、`test_speculative_sampling_reproduces_the_target_distribution`（2 萬次抽樣，第一與第二個位置的分布都等於目標分布；接受率等於 $\sum\min(p, q)$）、`test_absmax_quantization_error_is_at_most_half_a_step`。

## 4. 常見誤解

- **「KV cache 讓 attention 變成線性」**：它省掉的是重算舊 token 的投影與 MLP；每步的 attention 仍要讀全部的舊 $K$、$V$。
- **「speculative decoding 是近似的」**：用上述接受規則時，輸出分布與目標模型完全相同；它只改變速度。（貪婪版本則保證和目標模型的 greedy 輸出相同。）
- **「量化只是省記憶體」**：decode 是記憶體受限的，權重變小會直接變快。
- **「4 位元一定比 8 位元差很多」**：好的方法（GPTQ、AWQ）在大模型上的 4 位元損失很小；小模型比較敏感（Lab 09 §4）。

## 5. 練習

**想一想**

1. 用 (9.1) 算 GPT-2 small（MHA，fp32）在 1,024 token 時的 KV cache 大小，和它的權重比較。
2. 為什麼 decode 的算術強度約等於 batch size？prefill 的呢？
3. 驗證 §2.3 的推導：先證明 $\sum_x \max(0, p(x) - q(x)) = 1 - \sum_x \min(p(x), q(x))$。
4. $\alpha = 0.8$、$k = 4$ 時，(9.4) 是多少？$k \to \infty$ 的極限呢？
5. 8 位元、每列一個係數時，一個 $4096 \times 4096$ 的矩陣佔多少 bytes（係數用 fp16）？

**動手改**（在 `09_inference.ipynb`）

6. 把草稿模型換成更小（1 層）或更大（和目標一樣大），接受率與「每次呼叫目標模型得到的 token 數」怎麼變？
7. 用 temperature 0（greedy）做 speculative decoding，接受率是多少？
8. 量化時不量化 LM head（`include_head=False`），4 位元的 loss 改善多少？

## 6. 延伸閱讀

- Leviathan 等人（2022），〈Fast Inference from Transformers via Speculative Decoding〉：<https://arxiv.org/abs/2211.17192>；Chen 等人（2023），speculative sampling：<https://arxiv.org/abs/2302.01318>
- Kwon 等人（2023），PagedAttention（vLLM）：<https://arxiv.org/abs/2309.06180>
- DeepSeek-AI（2024），DeepSeek-V2（MLA）：<https://arxiv.org/abs/2405.04434>
- Dettmers 等人（2022），LLM.int8()：<https://arxiv.org/abs/2208.07339>；Frantar 等人（2022），GPTQ：<https://arxiv.org/abs/2210.17323>；Lin 等人（2023），AWQ：<https://arxiv.org/abs/2306.00978>
- Cai 等人（2024），Medusa：<https://arxiv.org/abs/2401.10774>；Li 等人（2024），EAGLE：<https://arxiv.org/abs/2401.15077>
- CS336 第 10 講（推論，Percy Liang）：<https://cs336.stanford.edu/>
- nanochat 的 `nanochat/engine.py`（KV cache、工具使用）：<https://github.com/karpathy/nanochat>
