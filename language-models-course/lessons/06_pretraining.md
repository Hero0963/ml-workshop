# 第 06 課：預訓練一個小 GPT——訓練迴圈、最佳化器與現代架構

> 前置：第 03–05 課 ｜ 實驗：[`notebooks/06_pretraining.ipynb`](../notebooks/06_pretraining.ipynb) ｜ 預估時間：3 小時（實驗本身約 45 分鐘；**第 09、11、12、13 課的實驗都用這裡存下的模型**）

## 這一課要回答的問題

- 預訓練的迴圈到底長什麼樣？資料怎麼切、loss 怎麼算、怎麼知道一開始就對了？
- AdamW 為什麼是預設？weight decay 為什麼要「解耦」？學習率為什麼要先暖身再遞減？
- nanochat 為什麼用 Muon？「把更新正交化」是什麼意思？
- 從 GPT-2 到 2026 年，Transformer 的配方改了哪些地方？各自的理由是什麼？

---

## 1. 白話版

### 1.1 迴圈

預訓練就是一直重複四件事：

1. 從一大串 token 裡隨機剪幾段（每段 256 個）；
2. 讓模型對每個位置猜下一個 token；
3. 算猜錯的程度（交叉熵）；
4. 往「讓 loss 變小」的方向微調所有權重。

重複幾千次，模型就從亂說話變成會寫故事。一開始的 loss 應該等於「完全亂猜」的 loss：4,096 個 token 平均猜，$\ln 4096 \approx 8.3$。如果一開始就遠高於它，初始化有問題。

### 1.2 每一步走多遠

學習率是「每一步往下坡走多遠」。一開始權重是亂的、梯度也很亂，先小步走（**暖身**）；中段大步走；最後小步走、慢慢停在谷底（**遞減**）。

### 1.3 AdamW：每個參數有自己的步伐

Adam 對每個參數記住「最近的梯度平均」與「最近的梯度有多大」，梯度一直很大的參數步伐自動縮小。AdamW 再加一個「每步把權重往 0 拉一點點」（weight decay），而且這一拉和梯度的大小無關。

### 1.4 Muon：讓每個方向都走一樣遠

一個權重矩陣的梯度，往往被少數幾個「大方向」主導，其他方向的訊號被淹沒。Muon 把更新矩陣「正交化」：保留每個方向，但把每個方向的步伐拉成一樣大。實驗上，同樣的算力能訓練得更好（nanochat、modded-nanogpt 的速度紀錄都用它）。

---

## 2. 正式版

### 2.1 資料

把所有故事串成一條 token 流，每篇前面放 `<|bos|>`。每一步從流中均勻抽 $B$ 個起點，取長度 $T+1$ 的視窗 $w$，輸入 $x = w_{0:T}$、目標 $y = w_{1:T+1}$。一步看過 $B \times T$ 個 token。
本課程：$B = 32$、$T = 256$、4,927,886 個訓練 token，1,500 步 ≈ 1,230 萬 token ≈ 2.5 個 epoch。資料重複幾次通常沒關係：Muennighoff 等人（2023）發現在資料受限時，重複到約 4 個 epoch 和全新資料幾乎一樣好。

### 2.2 Adam 與 AdamW

梯度 $g_t$，Adam（Kingma & Ba 2014）：

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2, \quad \theta_t = \theta_{t-1} - \eta\,\frac{m_t / (1-\beta_1^t)}{\sqrt{v_t / (1-\beta_2^t)} + \epsilon} \tag{6.1}$$

**AdamW**（Loshchilov & Hutter 2017）把 weight decay 從梯度裡拿出來、直接作用在權重上：

$$\theta_t \leftarrow \theta_t - \eta\lambda\,\theta_{t-1} \tag{6.2}$$

若把 L2 項加進梯度（「Adam＋L2」），它會被 $\sqrt{v_t}$ 除掉，梯度大的參數幾乎不受正則化——解耦後每個參數都以相同比例衰減。慣例：**只對矩陣做 weight decay**，不對 bias、norm 的增益做。LLM 常用 $\beta_2 = 0.95$（GPT-3），比預設 0.999 更快忘記舊的梯度大小，訓練較穩。

### 2.3 學習率排程

以乘數 $s(t) \in [0, 1]$ 乘上基礎學習率。暖身 $W$ 步之後：

**cosine**（GPT-3、CS336 作業 1）：

$$s(t) = s_{\min} + (1 - s_{\min})\cdot\tfrac{1}{2}\left(1 + \cos\frac{\pi (t - W)}{T_{\text{total}} - W}\right) \tag{6.3}$$

**WSD**（warmup–stable–decay；MiniCPM 2024 的名稱，nanochat 稱最後一段為 warmdown）：

$$s(t) = \begin{cases} 1 & W \le t < T_{\text{total}} - D \\ (T_{\text{total}} - t)/D & t \ge T_{\text{total}} - D \end{cases} \tag{6.4}$$

WSD 的好處：穩定段可以任意延長，想要一個「訓練完成」的模型時再接一段遞減；scaling law 實驗可以從同一條穩定段分支出不同長度的訓練。nanochat 目前用 $D = 0.65\,T_{\text{total}}$（查證 2026-09-25，`base_train.py` 的預設）。

**暖身為什麼需要**：一開始 $v_t$ 只看過幾個梯度，估計不準，步伐可能過大；post-norm 的架構尤其依賴暖身（第 04 課 §2.6）。

### 2.4 梯度裁剪

$$g \leftarrow g\cdot\min\left(1, \frac{c}{\|g\|_2}\right) \tag{6.5}$$

偶發的大梯度（壞批次、數值尖峰）不會把權重一次推太遠。常見 $c = 1$。

### 2.5 Muon

對一個權重矩陣 $W \in \mathbb{R}^{m \times n}$（只用在區塊內的矩陣；embedding、LM head、向量參數仍用 AdamW）：

$$M_t = \mu M_{t-1} + G_t, \qquad O_t = \mathrm{NS}_5\left(G_t + \mu M_t\right), \qquad W_t = W_{t-1} - \eta\sqrt{\max(1, m/n)}\;O_t \tag{6.6}$$

（第二式裡的 $G_t + \mu M_t$ 是 Nesterov 動量。）$\mathrm{NS}_5$ 是 5 步的 Newton–Schulz 迭代：先把 $X$ 除以 Frobenius 範數（讓所有奇異值 $\le 1$），再反覆做

$$X \leftarrow aX + b(XX^\top)X + c(XX^\top)^2X, \qquad (a, b, c) = (3.4445,\ -4.7750,\ 2.0315) \tag{6.7}$$

若 $X = U\Sigma V^\top$，每一步只作用在奇異值上：$\sigma \mapsto a\sigma + b\sigma^3 + c\sigma^5$。這個多項式在 0 附近斜率很大，把小的奇異值快速放大到 1 附近，於是 $\mathrm{NS}_5(G) \approx UV^\top$——**保留更新的方向、把每個方向的大小拉平**。係數刻意調得「不完全收斂」：不是太小的奇異值（正規化後大於約 0.02）最後落在約 0.7–1.2 之間而非精確的 1，但實驗上不影響，換來只需 5 步（Keller Jordan 2024 的部落格；nanochat 的註解說法是約 0.5–1.5）。

為什麼有效？一種說法：Adam 的更新矩陣條件數很大，幾乎是低秩的；正交化等於放大那些「罕見但重要的方向」。理論上，Muon 是在「譜範數」下的最陡下降（Bernstein & Newhouse 2024）。規模化：Moonshot 的 Moonlight（2025）加上 weight decay 與每參數的尺度調整後，報告在 compute-optimal 訓練下約有 AdamW 的 2 倍計算效率。nanochat 的版本又換成 Polar Express 係數、加了 NorMuon 的變異數正規化（見延伸閱讀）；本課程用 Keller Jordan 的原始版本。

### 2.6 從 GPT-2 到 nanochat：配方的演進

| 零件 | GPT-2（2019） | Llama 類（2023+） | nanochat（2026-09） | 理由 |
|---|---|---|---|---|
| 正規化 | LayerNorm（有增益與 bias） | RMSNorm | RMSNorm，**沒有可學參數** | 少算一步、少一些參數，品質不變 |
| 位置 | 可學的絕對位置 | RoPE | RoPE（base 100,000） | 相對位置、長度外推較好 |
| MLP | GELU，$4d$ | SwiGLU，$\tfrac{8}{3}d$ | **ReLU²**，$4d$ | ReLU² 便宜且不輸（Primer） |
| bias | 有 | 無 | 無 | 幾乎無用，拿掉讓計算與分散式更單純 |
| embedding | 綁定 | 不綁 | 不綁；LM head 以極小值（std 0.001）初始化 | 輸入與輸出的最佳表示不必相同 |
| attention | MHA | GQA | GQA（支援）、**QK-norm**、滑動視窗 | 推論省記憶體；穩定分數大小 |
| 輸出 | 直接 softmax | 直接 softmax | **logit soft-cap**：$z \leftarrow 15\tanh(z/15)$ | 限制 logit 大小，訓練更穩（Gemma 2：注意力 50、輸出 30） |
| 初始化 | 全部 $\mathcal{N}(0, 0.02)$，residual 投影縮小 | 類似 | **寫回 residual 的投影初始化為 0** | 每個區塊一開始是恆等映射 |
| 最佳化器 | Adam | AdamW | **Muon**（矩陣）＋ AdamW（其餘） | §2.5 |
| 精度 | fp32／fp16 | bf16 混合精度 | bf16，speedrun 用 **fp8** | 第 08 課 |

nanochat 還有一批較新的小技巧（value embeddings、每層可學的 residual 係數、把前一個 token 的 embedding「抹」進來的 smear、只看局部的滑動視窗層…），多來自 modded-nanogpt 的速度競賽。本課程的 `nanochat_style_config` 只實作表中的架構開關（RMSNorm 無參數、RoPE、ReLU²、無 bias、不綁 embedding、QK-norm、soft-cap）；初始化沿用 GPT-2 的方式，最佳化器則在 `build_optimizer` 另外選。

**nanochat 的單一旋鈕**：模型寬度 $d = 64 \times$ 層數、head 維度 128，學習率依 $1/\sqrt{d}$ 調整，訓練 token 數由「資料／參數比」決定（`base_train.py` 預設 12，speedrun 用 8；Chinchilla 的經驗值約 20，第 07 課）。使用者只需選 `--depth`，其餘自動算出。

### 2.7 怎麼讀 loss 曲線

- **一開始**：應在 $\ln|V|$ 附近（本課 8.32）。
- **很快掉到 unigram 程度**：模型先學會「哪些 token 常見」，接著學 bigram、再學更長的結構。
- **train 與 val 的差距**：資料重複多次後，差距變大代表開始背訓練資料。
- **尖峰**：學習率太大或數值問題；裁剪與 QK-norm、soft-cap 都是在抑制它。
- **最後的急降**：WSD／cosine 的遞減段，loss 往往明顯再掉一截——「退火」讓權重停在谷底。

---

## 3. 對照程式碼

| 概念 | 位置 |
|---|---|
| §2.1 資料 | `training.tokenize_stories`、`training.random_batch`；課程共用的切分與快取 `artifacts.course_tokens` |
| 訓練迴圈（累積梯度、裁剪、排程、定期驗證） | `training.train` |
| 驗證 loss 與 bits-per-byte | `training.evaluate` |
| (6.1)(6.2) AdamW（矩陣才 decay） | `optim.build_optimizer(model, "adamw")`、`optim.split_parameters` |
| (6.3)(6.4) 排程 | `optim.cosine_schedule`、`optim.wsd_schedule` |
| (6.6)(6.7) Muon | `optim.newton_schulz_orthogonalize`、`optim.Muon`、`build_optimizer(model, "muon")` |
| §2.6 三組架構 | `model.gpt2_config`、`llama_style_config`、`nanochat_style_config`；課程的 base model：`artifacts.base_model_config` |
| 命令列版本 | `scripts/pretrain.py` |

測試：`test_newton_schulz_flattens_the_spectrum_and_keeps_the_directions`、`test_training_memorizes_a_repeated_sequence`、`test_build_optimizer_steps_every_parameter`。

## 4. 常見誤解

- **「loss 還在降就繼續訓練」**：要看**驗證** loss；而且在固定算力下，「更大的模型少訓練一點」可能比「小模型訓練更久」好（第 07 課）。
- **「weight decay 就是 L2 正則化」**：在 Adam 裡兩者不同（§2.2）。
- **「Muon 可以用在所有參數」**：它是為矩陣設計的；embedding 與輸出層的最佳化動態不同，仍用 AdamW（Keller Jordan 的建議，nanochat 照做）。
- **「新架構每一項都是大進步」**：多數改動單獨看效果很小，而且在小模型上量不出來；它們的價值在於大規模下的穩定與效率。Lab 06 §5 的小規模比較要謹慎解讀。

## 5. 練習

**想一想**

1. 為什麼初始 loss 應該接近 $\ln|V|$？如果 LM head 用 $\mathcal{N}(0, 1)$ 初始化會怎樣？（提示：nanochat 用 0.001。）
2. 從 (6.1) 說明：只有 $m_t$ 的 bias correction 而沒有 $v_t$ 的，第一步的步伐會是多大？
3. 證明 NS 迭代只改變奇異值、不改變奇異向量：若 $X = U\Sigma V^\top$，則 $(XX^\top)X = U\Sigma^3V^\top$。
4. 畫出 $p(s) = 3.4445s - 4.7750s^3 + 2.0315s^5$ 在 $[0, 1]$ 上的圖，迭代 5 次後 $s = 0.01$ 會變成多少？
5. WSD 相對於 cosine 的優點是什麼？什麼情況下 cosine 比較方便？

**動手改**（在 `06_pretraining.ipynb`）

6. 把學習率乘 3 或除以 3，loss 曲線怎麼變？
7. 拿掉梯度裁剪（`grad_clip=None`），有沒有出現尖峰？
8. 用 `scripts/pretrain.py --steps 3000` 訓練更久，bits-per-byte 還能降多少？

## 6. 延伸閱讀

- Kingma & Ba（2014），Adam：<https://arxiv.org/abs/1412.6980>；Loshchilov & Hutter（2017），AdamW：<https://arxiv.org/abs/1711.05101>
- Keller Jordan（2024-12），〈Muon: An optimizer for hidden layers in neural networks〉：<https://kellerjordan.github.io/posts/muon/>
- Bernstein & Newhouse（2024），〈Old Optimizer, New Norm: An Anthology〉：<https://arxiv.org/abs/2409.20325>
- Liu 等人（2025），〈Muon is Scalable for LLM Training〉（Moonlight）：<https://arxiv.org/abs/2502.16982>
- Amsel 等人（2025），Polar Express：<https://arxiv.org/abs/2505.16932>；NorMuon（2025）：<https://arxiv.org/abs/2510.05491>
- Hu 等人（2024），MiniCPM（WSD 排程）：<https://arxiv.org/abs/2404.06395>
- Muennighoff 等人（2023），〈Scaling Data-Constrained Language Models〉：<https://arxiv.org/abs/2305.16264>
- Eldan & Li（2023），TinyStories：<https://arxiv.org/abs/2305.07759>
- nanochat 的 `nanochat/gpt.py`、`nanochat/optim.py`、`scripts/base_train.py`（MIT）：<https://github.com/karpathy/nanochat>；〈Beating GPT-2 for <<$100: the nanochat journey〉（2026-02）：<https://github.com/karpathy/nanochat/discussions/481>
- CS336 作業 1（實作 AdamW、cosine 排程、訓練迴圈）：<https://github.com/stanford-cs336/assignment1-basics>
