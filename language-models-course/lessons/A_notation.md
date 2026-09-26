# 附錄 A：符號與公式速查

> 撰寫日期：2026-09-25。公式編號對應各課講義。

## 符號

| 符號 | 意義 | 本課程的預設值 |
|---|---|---|
| $V$、$\lvert V\rvert$ | 詞彙表、詞彙量 | 4,096（GPT-2：50,257） |
| $x_{1:T}$ | token 序列，長度 $T$ | context 256 |
| $d$（`d_model`） | 隱藏維度 | 256 |
| $L$（`n_layer`） | 層數 | 4 |
| $H$、$H_{kv}$ | query head 數、key／value head 數 | 4、4 |
| $d_h$ | 每個 head 的維度 $= d / H$ | 64 |
| $d_{ff}$ | MLP 的中間寬度 | $4d$（SwiGLU：$\approx\tfrac{8}{3}d$） |
| $B$ | batch size（序列數） | 32 |
| $N$ | 參數量（第 07 課：參與矩陣乘法的參數） | 約 420 萬（全部 520 萬） |
| $D$ | 訓練 token 數 | 1,230 萬 |
| $C$ | 訓練 FLOPs | $\approx 6ND$ |
| $\eta$、$\lambda$ | 學習率、weight decay | |
| $\tau$ | 溫度（取樣或對比學習） | |
| $\beta$ | KL 係數（RLHF、DPO） | |
| $\pi_\theta$、$\pi_{\text{ref}}$ | 策略（語言模型）、參考模型 | |
| $\sigma(z)$ | sigmoid $1 / (1 + e^{-z})$ | |

## 語言模型與評估（第 01 課）

| 公式 | 編號 |
|---|---|
| $p(x_{1:T}) = \prod_t p(x_t \mid x_{<t})$ | (1.1) |
| $\mathcal{L} = -\frac{1}{T}\sum_t \log p_\theta(x_t \mid x_{<t})$ | (1.2) |
| $H(p, q) = H(p) + \mathrm{KL}(p \,\|\, q)$ | (1.3) |
| perplexity $= e^{\mathcal{L}}$；bpb $= \dfrac{\mathcal{L}_{\text{nats/token}}}{\ln 2 \cdot \overline{\text{bytes/token}}}$ | (1.4) |
| 加 k 平滑 $\dfrac{c(h, x) + k}{c(h) + k\lvert V\rvert}$ | (1.7) |
| Witten–Bell 插值 $\lambda(h) = \dfrac{c(h)}{c(h) + u(h)}$ | (1.8)(1.9) |

## word2vec（第 02 課）

| 公式 | 編號 |
|---|---|
| SGNS：$\log\sigma(u_o^\top v_c) + \sum_{k=1}^{K}\mathbb{E}_{w_k \sim P_n}\log\sigma(-u_{w_k}^\top v_c)$ | (2.3) |
| $P_n(w) \propto U(w)^{3/4}$ | (2.4) |
| 丟棄機率 $1 - \sqrt{t / f(w)}$ | (2.5) |
| 類比 $\arg\max_x \cos(v_x, v_b - v_a + v_c)$ | (2.6) |
| 最佳解 $v_w^\top u_c = \mathrm{PMI}(w, c) - \log K$ | (2.8) |

## Transformer（第 04–06 課）

| 公式 | 編號 |
|---|---|
| $\mathrm{softmax}(QK^\top / \sqrt{d_k} + M)V$ | (4.1) |
| RoPE：$\langle R_m q, R_n k\rangle = \langle q, R_{n-m}k\rangle$ | (4.5) |
| pre-norm：$x \leftarrow x + f(\mathrm{Norm}(x))$ | (4.6) |
| RMSNorm $\gamma \odot x / \sqrt{\overline{x^2} + \epsilon}$ | (4.8) |
| 每層參數 $\approx 12d^2$ | (4.11) |
| GPT-2 residual 初始化 std $0.02/\sqrt{2L}$ | (5.2) |
| temperature $p_j \propto e^{z_j/\tau}$；top-p：累積機率 $\ge p$ 的最小集合 | (5.4)(5.5) |
| AdamW：$\theta \leftarrow \theta - \eta\left(\hat m / (\sqrt{\hat v} + \epsilon) + \lambda\theta\right)$ | (6.1)(6.2) |
| Muon：$W \leftarrow W - \eta\sqrt{\max(1, m/n)}\,\mathrm{NS}_5(\text{Nesterov momentum})$ | (6.6) |

## 規模與系統（第 07–09 課）

| 公式 | 編號 |
|---|---|
| $C \approx 6ND$；含 attention：$(6N + 12LdT)D$ | (7.1)(7.2) |
| 模型狀態 16 bytes／參數（混合精度 AdamW） | §7.2.2 |
| $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$ | (7.5) |
| $N_{\text{opt}} \approx \sqrt{C/120}$，$D_{\text{opt}} \approx 20N_{\text{opt}}$ | (7.6) |
| roofline：$\min(\text{peak}, \text{bandwidth} \times I)$ | (8.1) |
| online softmax：$\ell' = \ell e^{m - m'} + \sum e^{x_j - m'}$ | (8.2) |
| pipeline bubble $(p - 1)/(m + p - 1)$ | (8.5) |
| KV cache $= 2LH_{kv}d_hTB \times$ bytes | (9.1) |
| speculative 接受 $\min(1, p/q)$，殘差 $\propto \max(0, p - q)$ | (9.3) |
| absmax 量化 $s = \max\lvert W_{r,:}\rvert / (2^{b-1} - 1)$ | (9.5) |

## 資料（第 10 課）

| 公式 | 編號 |
|---|---|
| Jaccard $\lvert A \cap B\rvert / \lvert A \cup B\rvert$ | (10.1) |
| $\Pr[m_h(A) = m_h(B)] = J(A, B)$ | (10.2) |
| LSH：$1 - (1 - s^r)^b$ | (10.3) |

## 後訓練與 embedding（第 11–13 課）

| 公式 | 編號 |
|---|---|
| SFT：$-\sum_t m_t \log p(x_t \mid x_{<t}) / \sum_t m_t$ | (11.1) |
| policy gradient $\mathbb{E}[(r - b)\nabla\log\pi]$ | (12.3) |
| Bradley–Terry $\sigma(r_w - r_l)$ | (12.4) |
| PPO-clip $\min(\rho A, \mathrm{clip}(\rho, 1 \pm \epsilon)A)$ | (12.5) |
| DPO $-\log\sigma\left(\beta\log\frac{\pi(y_w)}{\pi_{\text{ref}}(y_w)} - \beta\log\frac{\pi(y_l)}{\pi_{\text{ref}}(y_l)}\right)$ | (12.8) |
| GRPO advantage $(r_i - \bar r)/\mathrm{std}(r)$（nanochat：$r_i - \bar r$） | (12.9) |
| 每個 token 的 KL 估計 $\frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$ | (12.10) |
| pass@k $= 1 - \binom{n-c}{k}/\binom{n}{k}$ | (12.11) |
| InfoNCE $-\log\dfrac{e^{s(q_i, d_i)/\tau}}{\sum_j e^{s(q_i, d_j)/\tau}}$ | (13.1) |
| BM25 | (13.4) |
| 置中正確率 $(\text{acc} - \text{acc}_{\text{rand}})/(1 - \text{acc}_{\text{rand}})$ | (14.2) |

## 常用數字

| 項目 | 數值 | 來源 |
|---|---|---|
| GPT-2 small 參數量 | 124,439,808 | `tests/test_model.py` |
| GPT-2 詞彙量、context | 50,257、1,024 | GPT-2 論文 |
| Chinchilla 經驗比例 | 約 20 token／參數 | Hoffmann 等人 2022 |
| H100 SXM BF16 峰值（dense）、HBM 頻寬 | 約 989 TFLOPS、3.35 TB/s | NVIDIA 規格頁（2026-09-25） |
| $\ln 4096$（本課程初始 loss） | 8.318 | |
| $1 \text{ nat}$ | 1.443 bits | |
