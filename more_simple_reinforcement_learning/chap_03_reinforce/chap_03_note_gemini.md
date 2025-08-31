# 強化學習講義 - Chap03：策略梯度 (Policy Gradient) (Gemini 整理版)

本章節介紹了與價值學習 (Value-Based) 並列的另一大類強化學習方法：**策略學習 (Policy-Based)**。其核心思想不再是學習一個評分系統（Q-Table 或 Q-Network），而是直接學習一個**策略 (Policy)**，即一個從狀態直接映射到動作機率的模型。

- **價值學習 vs. 策略學習**:
  - **價值學習 (如 DQN)**: `狀態 -> 模型 -> 各動作的分數`，然後根據分數選擇動作（例如，選最高分）。
  - **策略學習 (如 REINFORCE)**: `狀態 -> 模型 -> 各動作的機率`，然後根據機率分佈來採樣動作。

---

## 1. REINFORCE 演算法

REINFORCE 是最基礎的策略梯度 (Policy Gradient) 演算法，它基於**蒙地卡羅採樣**來更新策略。

- **核心思想 (獎懲機制)**:
  1.  用當前的策略模型完整地玩一局遊戲，並記錄下整個軌跡（states, actions, rewards）。
  2.  計算這一局遊戲中，從**每一步**開始直到結束的**未來總折扣獎勵 $G_t$**。
  3.  如果 $G_t$ 是正的（好結果），就提高當初在該步所採取的動作的機率。
  4.  如果 $G_t$ 是負的（壞結果），就降低當初所採取的動作的機率。
- **目標函數**: 最大化期望回報 $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$。
- **梯度更新**:
  $$
  \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=0}^{T_i-1} G_{i,t} \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t})
  $$
- **缺點**: 由於蒙地卡羅採樣完全依賴於完整的遊戲軌跡，其**變異數 (variance) 非常高**，導致訓練過程很不穩定。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_03_reinforce/01_reinforce.ipynb`

---

## 2. REINFORCE with Baseline (帶基線的 REINFORCE)

為了緩解 REINFORCE 的高變異數問題，我們引入了**基線 (Baseline)**。基線的作用是提供一個對當前狀態價值的預期，使得獎勵訊號從一個絕對值變為一個相對值。

- **核心思想**:
  - 我們不再使用原始的未來總獎勵 $G_t$ 來更新策略。
  - 而是使用**優勢函數 (Advantage Function)** $A(s_t, a_t) = G_t - V(s_t)$。
    - $G_t$: **實際**得到的未來總獎勵（通過蒙地卡羅採樣）。
    - $V(s_t)$: **預期**能得到的未來總獎勵（由一個獨立的**價值網路**，即 `model_baseline`，來估計）。
- **獎懲機制變化**:
  - 如果 $G_t > V(s_t)$ (實際比預期好)，$A(s_t, a_t)$ 為正，鼓勵該動作。
  - 如果 $G_t < V(s_t)$ (實際比預期差)，$A(s_t, a_t)$ 為負，抑制該動作。
- **優點**: 大大降低了梯度的變異數，使得訓練**更穩定**、**更快速**。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_03_reinforce/02_baseline.ipynb`

---

## 3. Entropy Regularization (熵正規化)

- **問題**: 策略網路在訓練過程中，可能會過早地收斂到一個局部最優解，變得過於「自信」（例如，某個動作的機率趨近於 100%），從而停止探索其他可能更優的策略。
- **解決方案**: 在損失函數中加入**熵 (Entropy)** 作為一個正則項，以鼓勵探索。
- **核心思想**:
  - **熵** $H(\pi(\cdot|s))$ 衡量的是策略 $\pi$ 在狀態 `s` 下輸出機率分佈的**不確定性**。
    - 策略越確定（如 `[0.99, 0.01]`），熵越**低**。
    - 策略越不確定（如 `[0.5, 0.5]`），熵越**高**。
  - 我們希望**最大化熵**，即鼓勵策略保持一定的不確定性。
- **損失函數變化**:
  $$
  \text{Loss} = -(\text{Policy Gradient Loss}) - \beta \cdot H(\pi(\cdot|s))
  $$ 
  （在最小化 Loss 的框架下，減去熵項就等於最大化熵）
- **優點**:
  - **促進探索**: 防止策略過早收斂。
  - **提升穩定性**: 讓學習過程更平滑。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_03_reinforce/03_entropy.ipynb`
