# 第 12 課：偏好與強化學習——RLHF、DPO、GRPO

> 前置：第 11 課 ｜ 實驗：[`notebooks/12_rl.ipynb`](../notebooks/12_rl.ipynb)（需要 Lab 11 的模型） ｜ 預估時間：3 小時（推導較多）

## 這一課要回答的問題

- SFT 之後為什麼還需要強化學習？「獎勵」從哪裡來？
- RLHF 的三個步驟是什麼？PPO 的「clip」在防什麼？
- DPO 怎麼做到「不需要獎勵模型、也不需要抽樣」？
- DeepSeek-R1 用的 GRPO 是什麼？nanochat 為什麼說它的「GRPO」其實更像 REINFORCE？
- RL 到底讓模型學會新東西，還是只是讓它更常答對本來就會的題目？
- 為什麼獎勵越高，結果有時候反而越糟？

---

## 1. 白話版

### 1.1 模仿 vs. 被打分

SFT 是「照著範本抄」。但很多時候我們寫不出完美的範本，卻**分得出好壞**：兩個回答放在一起，人很容易說哪個比較好；一題數學的答案對不對，程式可以直接檢查。強化學習就是利用這種「打分數」的訊號：模型自己寫幾個答案，得高分的寫法以後多寫一點，得低分的少寫一點。

### 1.2 三種常見做法

- **RLHF**：先請人比較很多對答案，訓練一個「評分模型」模仿人的偏好；再讓語言模型在評分模型的指導下練習（ChatGPT 的做法）。
- **DPO**：跳過評分模型，直接拿「人比較喜歡的答案」和「比較不喜歡的答案」，把前者的機率往上推、後者往下壓。
- **GRPO／可驗證的獎勵**：數學、程式這種有標準答案的題目，不需要人：同一題寫 16 個答案，比「這一組的平均」好的就加強，差的就減弱。

### 1.3 別偏離太遠

如果只追求高分，模型可能找到評分的漏洞（例如評分模型喜歡長答案，它就寫得又長又空）。所以通常會加一條繩子：**不要離原本的模型太遠**（KL 懲罰）。

---

## 2. 正式版

### 2.1 把語言模型看成策略

prompt $x$、回答 $y = (y_1, \dots, y_n)$。語言模型就是策略 $\pi_\theta(y \mid x) = \prod_t \pi_\theta(y_t \mid x, y_{<t})$。給定獎勵 $r(x, y)$，常見的目標是

$$\max_\theta\ \mathbb{E}_{x,\ y \sim \pi_\theta}\left[r(x, y)\right] - \beta\,\mathrm{KL}\left(\pi_\theta(\cdot \mid x)\,\|\,\pi_{\text{ref}}(\cdot \mid x)\right) \tag{12.1}$$

$\pi_{\text{ref}}$ 通常是 SFT 模型。

### 2.2 Policy gradient（REINFORCE）

用 $\nabla \pi = \pi \nabla \log \pi$：

$$\nabla_\theta\,\mathbb{E}_{y \sim \pi_\theta}[r] = \mathbb{E}_{y \sim \pi_\theta}\left[r(x, y)\,\nabla_\theta \log \pi_\theta(y \mid x)\right] = \mathbb{E}\left[r \sum_t \nabla_\theta \log \pi_\theta(y_t \mid x, y_{<t})\right] \tag{12.2}$$

任何不依賴 $y$ 的 baseline $b(x)$ 都可以減掉而不改變期望（因為 $\mathbb{E}[\nabla \log \pi] = 0$），但能大幅降低變異數：

$$\nabla J = \mathbb{E}\left[\left(r - b(x)\right)\nabla_\theta \log \pi_\theta(y \mid x)\right] \tag{12.3}$$

$A = r - b$ 稱為 advantage。實作：抽樣、算獎勵、把 loss 設成 $-A \log \pi_\theta(y \mid x)$ 做梯度下降。

### 2.3 RLHF：InstructGPT 的三步

1. **SFT**（第 11 課）。
2. **獎勵模型**：對同一個 prompt 的兩個回答，標註者選出較好的 $y_w$ 與較差的 $y_l$。用 Bradley–Terry 模型

$$P(y_w \succ y_l \mid x) = \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \tag{12.4}$$

最大化它的 log-likelihood 來訓練 $r_\phi$（通常是 SFT 模型換掉輸出層）。

3. **PPO**：以 $r_\phi$ 減去 KL 懲罰當獎勵，最佳化 (12.1)。PPO（Schulman 等人 2017）用重要性比值 $\rho_t = \pi_\theta(y_t \mid \cdot) / \pi_{\text{old}}(y_t \mid \cdot)$，並把它夾在 $[1 - \epsilon, 1 + \epsilon]$：

$$\mathcal{L}_{\text{PPO}} = -\mathbb{E}_t\left[\min\left(\rho_t A_t,\ \mathrm{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon) A_t\right)\right] \tag{12.5}$$

同一批樣本要更新好幾次時，clip 讓策略一次不會改變太多（信賴域的簡化版）。PPO 還需要一個 value 網路估計 $A_t$——對 LLM 來說，這等於再養一個同樣大的模型。

### 2.4 DPO：把獎勵模型「解」掉

(12.1) 有封閉解（對每個 $x$，在所有分布上最大化「期望獎勵減 KL」）：

$$\pi^*(y \mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y \mid x)\,\exp\left(\frac{r(x, y)}{\beta}\right) \tag{12.6}$$

反過來寫，獎勵可以用最佳策略表示：

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x) \tag{12.7}$$

代入 (12.4)，$\beta \log Z(x)$ 在相減時消掉。於是直接對策略做最大概似（Rafailov 等人 2023）：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right] \tag{12.8}$$

標題說的「你的語言模型其實是一個獎勵模型」就是 (12.7)。優點：不需要獎勵模型、訓練時不需要抽樣（離線、穩定、便宜）。缺點：只從固定的偏好資料學，看不到自己新產生的答案。KTO（Ethayarajh 等人 2024）等變形放寬了「必須成對」的要求。

### 2.5 GRPO 與可驗證的獎勵

DeepSeekMath（Shao 等人 2024）提出 GRPO：對每個 prompt 抽 $G$ 個回答，用**組內的平均**當 baseline，免掉 value 網路：

$$A_i = \frac{r_i - \mathrm{mean}(r_1, \dots, r_G)}{\mathrm{std}(r_1, \dots, r_G)} \tag{12.9}$$

再用 (12.5) 的 clip 目標，加上對參考模型的 KL 懲罰。KL 不是精確算出來的，而是在模型自己抽出的每個 token 上估計（Schulman 的「k3」估計量）：

$$\hat{\mathrm{KL}}_t = \frac{\pi_{\text{ref}}(y_t \mid \cdot)}{\pi_\theta(y_t \mid \cdot)} - \log \frac{\pi_{\text{ref}}(y_t \mid \cdot)}{\pi_\theta(y_t \mid \cdot)} - 1 \tag{12.10}$$

因為 $u - \log u - 1 \ge 0$，每一項都非負；在 $y_t \sim \pi_\theta$ 時期望值正好是 $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$（練習 6）。

DeepSeek-R1（2025）用 GRPO 搭配**規則式獎勵**（答案對不對、格式對不對）直接在 base model 上訓練（R1-Zero），模型自發地寫出越來越長的推理過程。這類方法稱為 **RLVR**（reinforcement learning with verifiable rewards），也是 CS336 作業 5 的主題。

後續的修正：

- **Dr. GRPO**（Liu 等人 2025）：除以 std 與「除以回答長度」會引入偏差（例如讓錯誤的長答案受罰較輕），建議拿掉。
- **DAPO**（Yu 等人 2025）：把 loss 在**所有 token** 上平均而不是先在每個序列內平均；上下 clip 範圍不對稱；丟掉整組都對或都錯（advantage 全為 0）的題目。

**nanochat 的簡化版**（`scripts/chat_rl.py` 的說明，查證 2026-09-25）：拿掉 KL 與參考模型；每批樣本只更新一次（on-policy），所以不需要比值與 clip；token 層級的平均（DAPO 式）；advantage 只用 $r - \mathrm{mean}$ 不除 std。作者自己說：這已經很接近 REINFORCE。在我們的程式裡，`policy_gradient_loss(logprobs, adv, mask)` 不給 `old_logprobs` 時，梯度正好等於 REINFORCE（`tests/test_chat_and_rl.py::test_on_policy_loss_gradient_is_reinforce` 驗證）。

### 2.6 RL 學到了什麼？

**pass@k**：對一題抽 $k$ 個答案，至少一個對的機率。用 $n \ge k$ 個樣本、其中 $c$ 個正確，不偏的估計是（Chen 等人 2021）

$$\text{pass@}k = 1 - \binom{n - c}{k}\Big/\binom{n}{k} \tag{12.11}$$

Yue 等人（2025）比較 RLVR 前後的模型：RL 大幅提高 pass@1，但在 $k$ 很大時，**base model 的 pass@k 反而比較高**——RL 主要是讓模型更穩定地走向它本來就找得到的正確路徑（分布變尖），而不是學到全新的解法。這個結論仍有爭論，但它提醒我們：同時看 pass@1 與 pass@k。Lab 12 會在小模型上量這兩個數字。

反過來說，pass@k 也是 RL 之前的**健康檢查**：如果模型抽很多次都幾乎做不到，組內的獎勵就全是 0，RL 沒有訊號可學。Lab 12 §7 記錄了一個在小模型上失敗的例子：3 位數加法的 step-by-step 答案，greedy 八成對，但 temperature 1 只有兩成對：錯誤一半是某一步的個位數加法差 1（散在上百個不同的數字事實上），四成是格式整個跑掉。RL 的梯度方向雖然正確，訊號卻被雜訊淹沒。

### 2.7 獎勵被鑽漏洞

- 獎勵模型只是人類偏好的近似；最佳化得太用力，真正的品質會先升後降（Gao 等人 2022 的 overoptimization scaling laws）。KL 懲罰與提早停止是常見對策。
- 可驗證的獎勵比較難鑽，但仍要小心：答案的抽取規則（例如「取最後一個數字」）本身就可能被利用。
- Lab 12 有一個一眼就看得出來的例子：獎勵是「故事開頭有沒有提到主題字」，GRPO 之後模型把主題字塞進每一句（「the horse played with the horse」）——獎勵大漲，故事變差。加上 KL 懲罰 (12.10) 後，獎勵少漲一點，文字也比較像原本的模型。

---

## 3. 對照程式碼

| 概念 | 位置（`src/lm_course/rl.py`） |
|---|---|
| 每個回答 token 的 $\log\pi$ | `completion_logprobs`（同一 prompt 的多個回答一起算，含遮罩） |
| (12.9) 與 nanochat 的 $r - \mathrm{mean}$ | `group_advantages(rewards, normalize_std=...)` |
| (12.3)(12.5) | `policy_gradient_loss`（不給 `old_logprobs` 就是 on-policy 的 REINFORCE；token 層級平均） |
| (12.10) | `kl_penalty` |
| (12.8) | `dpo_loss` |
| (12.11) | `pass_at_k` |
| 獎勵：故事開頭有沒有提到主題（Lab 12 主任務） | `chat.mentions_topic`、`chat.story_request`、`chat.HELD_OUT_TOPICS` |
| 獎勵：加法答對與否（Lab 12 §7） | `chat.addition_reward` |
| 同一 prompt 批次抽樣（有 KV cache） | `sampling.generate(..., num_samples=G)` |

## 4. 常見誤解

- **「RLHF 讓模型學會新知識」**：它調整的是在既有能力中「選哪一種行為」；新知識主要來自預訓練。
- **「DPO 和 RLHF 等價」**：兩者最佳化同一個目標 (12.1) 的最佳解，但 DPO 只用固定的偏好資料（離線），RLHF 用模型自己的新樣本（線上），實務表現可以不同。
- **「GRPO 一定要有 KL 懲罰」**：原版有；DAPO、nanochat 等都拿掉了，改靠 on-policy 與小學習率控制漂移。
- **「整組都答錯的題目也能學」**：所有 $r_i$ 相同時 advantage 全為 0，沒有任何梯度——太難或太簡單的題目都浪費算力。

## 5. 練習

**想一想**

1. 證明 $\mathbb{E}_{y \sim \pi}[\nabla \log \pi(y)] = 0$，並由此說明 baseline 不改變 (12.3) 的期望。
2. 從 (12.1) 推出 (12.6)。提示：對每個 $x$ 做帶約束的最佳化，或把目標寫成 $-\beta\,\mathrm{KL}(\pi \,\|\, \pi^*) + \text{常數}$。
3. 驗證 (12.7) 代入 (12.4) 時 $Z(x)$ 會消掉。
4. $n = 16$ 個樣本中 $c = 2$ 個正確：pass@1、pass@4、pass@16 各是多少？
5. 一組 8 個回答的獎勵是 $(1, 0, 0, 0, 0, 0, 0, 0)$：用 (12.9) 與 nanochat 的 $r - \mathrm{mean}$，答對的那個 advantage 各是多少？
6. 證明 (12.10) 非負，且 $\mathbb{E}_{y_t \sim \pi_\theta}[\hat{\mathrm{KL}}_t] = \mathrm{KL}(\pi_\theta \,\|\, \pi_{\text{ref}})$。為什麼不直接用 $\log \pi_\theta - \log \pi_{\text{ref}}$（它的期望值也是 KL）？

**動手改**（在 `12_rl.ipynb`）

7. 把 advantage 改成除以 std（`normalize_std=True`），學習曲線有什麼不同？
8. 把每個主題的樣本數 $G$ 從 16 降到 4：「沒有訊號」的 prompt 變多少？學習變快還是變慢（以抽樣的總數計）？
9. KL 懲罰的 $\beta$ 試 0.03、0.3、1.0，畫出「held-out 獎勵 vs KL/token」的取捨曲線。
10. 修補獎勵：主題字出現超過 3 次就扣分，或乘上 $1 - \text{repetition}$。鑽漏洞的行為消失了嗎？模型又找到什麼新漏洞？
11. DPO 的 $\beta$ 從 0.1 改成 1.0，KL/token 怎麼變？再用 DPO 後的模型重新抽樣、組一批新的配對再訓練一輪（iterative DPO），和 GRPO 差多少？

## 6. 延伸閱讀

- Christiano 等人（2017），從人類偏好學習：<https://arxiv.org/abs/1706.03741>；Ziegler 等人（2019）：<https://arxiv.org/abs/1909.08593>；Stiennon 等人（2020）：<https://arxiv.org/abs/2009.01325>
- Ouyang 等人（2022），InstructGPT：<https://arxiv.org/abs/2203.02155>
- Schulman 等人（2017），PPO：<https://arxiv.org/abs/1707.06347>
- Rafailov 等人（2023），DPO：<https://arxiv.org/abs/2305.18290>；Ethayarajh 等人（2024），KTO：<https://arxiv.org/abs/2402.01306>
- Shao 等人（2024），DeepSeekMath（GRPO）：<https://arxiv.org/abs/2402.03300>；DeepSeek-AI（2025），DeepSeek-R1：<https://arxiv.org/abs/2501.12948>
- Yu 等人（2025），DAPO：<https://arxiv.org/abs/2503.14476>；Liu 等人（2025），Dr. GRPO：<https://arxiv.org/abs/2503.20783>
- Yue 等人（2025），〈Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?〉：<https://arxiv.org/abs/2504.13837>
- Gao 等人（2022），獎勵模型過度最佳化：<https://arxiv.org/abs/2210.10760>；Bai 等人（2022），Constitutional AI：<https://arxiv.org/abs/2212.08073>
- Lambert 等人（2024），Tülu 3（開源後訓練的完整配方）：<https://arxiv.org/abs/2411.15124>
- CS336 第 15–16 講與作業 5（SFT 與 GRPO 解數學題；選配的安全性與 DPO）：<https://cs336.stanford.edu/>
- nanochat 的 `scripts/chat_rl.py`：<https://github.com/karpathy/nanochat>
