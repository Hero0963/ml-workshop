# 強化學習講義 - Chap05：近端策略優化 (PPO) (Gemini 整理版)

本章節我們將學習**近端策略優化 (Proximal Policy Optimization, PPO)** 演算法。PPO 是當前強化學習領域最主流、最穩定的演算法之一，它由 OpenAI 提出，旨在解決傳統策略梯度方法更新步長難以確定、訓練不穩定的問題。PPO 作為 Actor-Critic 框架下的一種演算法，憑藉其出色的穩定性、易於實現和高效的數據利用率，成為了許多應用的首選，其中最著名的就是用於訓練 ChatGPT 的 RLHF 過程。

---

## 1. 從 On-Policy 到 Off-Policy：重要性採樣

*   **傳統策略梯度 (如 REINFORCE) 的困境**：身為 **On-Policy** 演算法，它要求產生數據的策略和要更新的策略必須是同一個。這意味著每更新一次網路，就需要用新網路去採集一批全新的數據，樣本效率極低。

*   **PPO 的解決方案：重要性採樣 (Importance Sampling)**
    *   **核心思想**：PPO 透過重要性採樣，巧妙地利用**舊策略 $\pi_{old}$** 收集的數據，來更新**當前的策略 $\pi_{new}$**，從而轉變為 **Off-Policy** 演算法，實現了數據的重複利用。
    *   **方法**：引入一個重要性比率 $r_t(\theta) = \frac{\pi_{new}(a|s)}{\pi_{old}(a|s)}$，我們可以修正新舊策略之間的分佈差異。
    *   **目標函數**：我們的優化目標，從而在數學上轉化為：
        $$
        J(\theta_{new}) = \mathbb{E}_{a_t \sim \pi_{\theta_{old}}} \left[ \frac{\pi_{\theta_{new}}(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \hat{A}_t \right]
        $$
        其中 $\hat{A}_t$ 是在 $t$ 時刻的優勢函數估計值。

*   **對應程式碼**：
    *   在 `train_action` 函數中，`prob_old` 就是 $\pi_{\theta_{old}}(a_t|s_t)$，`prob_new` 就是 $\pi_{\theta_{new}}(a_t|s_t)$。
    *   `ratio = prob_new / prob_old` 這行程式碼，計算的就是這個至關重要的重要性採樣比率。

---

## 2. PPO 的核心創新：裁剪代理目標 (Clipped Surrogate Objective)

*   **問題**：重要性採樣雖然提高了數據效率，但如果新舊策略差異過大，會導致 `ratio` 值劇烈波動，從而使訓練不穩定。

*   **PPO-Clip 的解決方案**：PPO 的作者提出了一種非常新穎且簡潔的目標函數，通過**裁剪 (Clipping)** 來限制策略的更新幅度。
    *   **目標函數**:
        $$
        L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
        $$
        其中，$\epsilon$ 是一個超參數，通常取 `0.1` 或 `0.2`。
    *   **直觀理解**：這個公式創造了一個「悲觀」的下界。
        1.  當優勢 $\hat{A}_t > 0$ 時（好動作），目標函數不鼓勵 `ratio` 超過 $1+\epsilon$。
        2.  當優勢 $\hat{A}_t < 0$ 時（壞動作），目標函數不鼓勵 `ratio` 低於 $1-\epsilon$。
        這相當於在舊策略周圍創建了一個隱式的「信任區域」，確保了每一次更新都是小步、穩定且安全的。

*   **對應程式碼**：
    *   `train_action` 函數中的這幾行，就是 PPO-Clip 的完美實現：
        ```python
        ratio = prob_new / prob_old
        surr1 = ratio * delta
        surr2 = ratio.clamp(0.8, 1.2) * delta
        loss = -torch.min(surr1, surr2).mean()
        ```

---

## 3. 優勢函數的精煉：GAE

*   **問題**：如何準確地估計優勢函數 $\hat{A}_t$？單步的 TD 誤差雖然穩定，但有偏差；而完整的蒙地卡羅回報雖然無偏，但方差極高。

*   **解決方案：廣義優勢估計 (Generalized Advantage Estimation, GAE)**
    *   GAE 通過引入參數 $\lambda$，對未來多步的 TD 誤差進行指數加權求和，從而在偏差和方差之間取得了巧妙的平衡。
    *   **數學公式**：
        $$
        \hat{A}^{GAE}_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}
        $$
        其中 $\delta_{t+l} = r_{t+l} + \gamma V(s_{t+l+1}) - V(s_{t+l})$ 是在未來第 `l` 步的 TD 誤差。

*   **對應程式碼**：
    *   `train_action` 函數中的第一個 `for` 迴圈，就是 GAE 的具體實現。

---

## 4. 處理連續動作空間

*   **核心思想**：PPO 的框架可以無縫從離散動作空間遷移到連續動作空間，唯一的變化在於策略的表示方式。
*   **策略表示**：
    *   **離散動作**：策略由一個**分類分佈 (Categorical Distribution)** 表示，網路輸出每個動作的機率。
    *   **連續動作**：策略由一個**高斯分佈 (Gaussian Distribution)** 表示，網路輸出該分佈的**均值 $\mu$** 和**標準差 $\sigma$**。
*   **概率計算**：
    *   對於連續動作，我們使用高斯分佈的**概率密度函數 (PDF)** 來計算 $\pi(a|s)$。
        $$
        \mathrm{pdf}(a|\mu, \sigma) = \frac{1}{\sigma \sqrt{2\pi}} \exp \left( -\frac{(a - \mu)^2}{2\sigma^2} \right)
        $$
*   **PPO 核心不變**：在計算出動作的概率密度後，後續的 `ratio` 計算、裁剪、以及目標函數都與離散情況完全相同，展現了 PPO 演算法的通用性。
*   **對應程式碼**：`02_conti_action.ipynb` 中的 `Model` 類和 `train_action` 函數中的概率計算部分。

---

## 5. PPO 的現代應用：RLHF

*   PPO 的穩定性和可靠性，使其成為訓練大型語言模型（LLM）的 **RLHF (Reinforcement Learning from Human Feedback)** 階段的核心演算法。
*   在 RLHF 中，LLM 本身是 **Actor**，它生成回答（**Action**），而一個預先訓練好的**獎勵模型 (Reward Model)** 則扮演了 **Critic** 的角色的一部分，為生成的回答打分（**Reward**）。PPO 則負責穩定地微調 LLM，使其生成的內容更符合人類的偏好。
