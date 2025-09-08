# 強化學習講義 - Chap07：深度確定性策略梯度 (DDPG) 與其改進 (TD3) (Gemini 整理版)

本章節將深入探討在連續控制領域中極為重要的兩種 Actor-Critic 演算法：**深度確定性策略梯度 (Deep Deterministic Policy Gradient, DDPG)** 及其後繼者 **雙生延遲 DDPG (Twin-Delayed DDPG, TD3)**。這兩種方法是解決諸如機器人手臂控制、自動駕駛等具有連續動作空間問題的基石。

---

## 1. DDPG (Deep Deterministic Policy Gradient)

DDPG 可以被視為 DQN 演算法在連續動作空間的成功延伸。它是一個 Off-Policy 的、基於 Actor-Critic 框架的演算法。

### 核心思想

1.  **確定性策略 (Deterministic Policy)**
    *   與 A2C/PPO 等輸出動作機率的**隨機策略**不同，DDPG 的 Actor 網路學習一個**確定性策略** $\mu_\theta(s)$，它將狀態 $s$ 直接映射到一個唯一的、確定的動作 $a$。
    *   **原因 (`explain why`)**: 對於連續動作空間（例如，力矩範圍 `[-2.0, 2.0]`），動作的可能性是無限的。如果要去計算所有可能動作的機率分佈，數學上需要進行積分，這在實務中是難以處理的。而直接輸一個確定的動作值，則完美地繞開了這個問題，使得更新更為直接高效。
    *   **程式碼實現**: 在 `01_ddpg.ipynb` 的 Actor `Model` 中，輸出層使用 `torch.nn.Tanh()` 激活函數，將動作值限定在 `[-1, 1]` 的範圍內，這就是確定性策略的典型實現。

2.  **Actor-Critic 框架**
    *   **Actor ($\\mu_\theta$)**: 策略網路，負責輸出最佳動作。其更新目標是最大化 Critic 給出的 Q 值。
    *   **Critic ($Q_\phi$)**: 價值網路，負責評估 Actor 輸出的動作的價值 $Q(s, a)$。它通過最小化 TD 誤差來學習，與 DQN 非常相似。

3.  **目標網路與軟更新 (Target Networks & Soft Updates)**
    *   **`model_delay` 的引入**: 為了穩定 TD 學習，DDPG 為 Actor 和 Critic 都創建了目標網路  $\mu'{\theta'}$ 和
     $Q'{\phi'}$。這些目標網路是主網路的延遲副本。
    *   **軟更新 (Soft Update)**: DDPG 不採用 DQN 的定期硬拷貝，而是使用 Polyak 平均進行軟更新，在每一步訓練後都小幅度地將主網路的權重混合到目標網路中：
        $$ 
        \theta' \leftarrow \tau\theta + (1-\tau)\theta' \\
        \phi' \leftarrow \tau\phi + (1-\tau)\phi' 
        $$ 
        其中 $\tau \ll 1$（例如 0.01）。這使得目標值的變化非常平滑，極大地提高了訓練的穩定性。

### DDPG 的挑戰：Q 值過高估計 (Overestimation Bias)

*   **問題 (`why 高估而不是低估`)**: DDPG 的 Actor 更新目標是最大化 Critic 的輸出。如果 Critic 由於函數近似誤差，偶然對某個不佳的動作給出了一個虛高的 Q 值，那麼 Actor 就會被錯誤地引導去學習這個動作。這種 Actor 不斷利用 Critic 誤差的行為，會形成一個惡性循環，導致 Q 值被系統性地、單向地高估。

---

## 2. TD3 (Twin-Delayed DDPG)

TD3 針對 DDPG 的 Q 值過高估計和訓練不穩定的問題，提出了三項關鍵改進，使其成為 DDPG 的一個更強大、更魯棒的替代品。

### 核心改進

1.  **裁剪的雙 Q 學習 (Clipped Double Q-Learning)**
    *   **思想**: 這是 TD3 的核心，即 "Twin" 的來源。它直接解決 Q 值過高估計問題。
    *   **實現**: TD3 學習**兩個獨立的 Critic 網路** ($Q_{\\phi_1}$ 和 $Q_{\\phi_2}$)，以及它們各自的目標網路 ($Q'_{\\phi_1}$ 和 $Q'_{\\phi_2}$)。在計算 TD-Target 時，取兩個目標網路中較小的那個 Q 值估計：
        $$ 
        y = r + \gamma \min_{i=1,2} Q'_{\\phi_i}(s', a') 
        $$ 
    *   **作用**: 單個 Critic 網路可能出錯，但兩個獨立的 Critic 網路**同時**對同一個動作產生過高估計的機率要小得多。取最小值可以得到一個更保守、更接近真實值的目標，從而有效抑制 Q 值的高估。
    *   **程式碼實現**: 在 `02_td3.ipynb` 中，明確定義了 `model_value1`, `model_value2` 以及它們各自的 `_delay` 版本。

2.  **延遲的策略更新 (Delayed Policy Updates)**
    *   **思想**: 這是 "Delayed" 的主要含義之一。TD3 認為在價值估計（Q 值）本身還不夠準確時，不應頻繁更新策略。
    *   **實現**: Actor 和目標網路的更新頻率，應低於 Critic 網路。通常的作法是「**每更新兩次 Critic，才更新一次 Actor 和目標網路**」。
    *   **作用**: 這給了 Critic 網路更充分的時間來收斂到一個更準確的價值估計，然後 Actor 再基於這個更可靠的評估來進行更新，使得策略的提升更穩健。

3.  **目標策略平滑化 (Target Policy Smoothing)**
    *   **思想**: 這是為了降低價值目標對特定動作值的過度依賴，使價值函數的學習更平滑。
    *   **實現**: 在計算 TD-Target 時，對目標 Actor $\\mu'$ 產生的動作 $a'$ 加入少量隨機噪聲，並將其裁剪到有效範圍內：
        $$ 
        a'(s') \leftarrow \text{clip}(\mu'(s') + \text{clip}(\epsilon, -c, c), a_{Low}, a_{High}), \quad \epsilon \sim \mathcal{N}(0, \sigma) 
        $$ 
    *   **作用**: 這使得價值目標對動作的微小變化不那麼敏感，有助於防止 Actor 利用 Critic 價值函數中的某些意外的尖峰，從而學習到一個更平滑的價值曲面。

---

## 3. 補充討論：離散動作空間的適用性

*   **核心問題**: DDPG/TD3 的確定性策略梯度，其數學基礎是**可微分的鏈式法則**，梯度需要從 Critic 平滑地傳播到 Actor。
*   **離散空間的挑戰**: 在離散空間中，動作選擇本質上是一個 `argmax` 操作，而 `argmax` 是**不可微分的**，這直接切斷了 DDPG 的梯度傳播路徑，使其無法直接應用。
*   **理論上的橋樑：Gumbel-Softmax**:
    *   Gumbel-Softmax 是一種數學技巧，它通過引入 Gumbel 噪聲和 Softmax 函數，為離散的 `argmax` 或分類採樣創建了一個可微分的連續代理。
    *   理論上可以利用此技術改造 DDPG，但這會顯著增加演算法的複雜度，且效果和效率通常不如原生為離散空間設計的演算法。
*   **結論**: 與其對 DDPG 進行複雜的改造，不如直接選擇更合適的工具。對於離散動作空間問題，**DQN 及其變體 (如 Rainbow)** 和 **PPO** 是更標準、更高效、更強大的選擇。
