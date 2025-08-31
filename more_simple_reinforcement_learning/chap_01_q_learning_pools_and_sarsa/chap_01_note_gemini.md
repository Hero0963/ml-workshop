# 強化學習講義 - Chap01：價值學習基礎 (Gemini 整理版)

本章節專注於強化學習的基礎：**表格型價值學習 (Tabular Value-Based Learning)**。核心思想是為環境中的每一個「狀態-動作」組合學習一個價值（分數），並儲存在一張大表 (Q-Table) 中。這適用於像「冰湖遊戲」這樣狀態與動作空間都有限的場景。

---

## 1. Q-Learning (Off-Policy)

Q-Learning 是一種經典的 **Off-Policy (異策略)** 時間差分 (TD) 學習方法。它的核心在於，它在更新 Q-Table 時，總是假設下一步會採取**最優**的動作，而不管智能體實際探索時選擇了哪個動作。這讓它在學習時比較「有野心」，專注於學習全局的最佳路徑。

- **更新公式**:
  $$
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \big]
  $$
- **關鍵點**: 更新時使用了 `max_a Q(s_{t+1}, a)`，即下一狀態所能帶來的**最大**期望價值。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_01_q_learning_pools_and_sarsa/01_QLearning.ipynb`

---

## 2. SARSA (On-Policy)

SARSA 是一種 **On-Policy (同策略)** 時間差分 (TD) 學習方法。它的更新方式比較「保守」，完全依賴於智能體**實際**採取的策略。它考慮了智能體在探索過程中可能會犯錯，因此學到的策略通常更安全。

- **名稱由來**: 其更新依賴於一個完整的五元組 `(S, A, R, S', A')`，即 (狀態, 動作, 獎勵, 下一狀態, **下一動作**)。
- **更新公式**:
  $$
  Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \big]
  $$
- **關鍵點**: 更新時使用了 `Q(s_{t+1}, a_{t+1})`，即在下一狀態**實際**採取的動作 `a_{t+1}` 所對應的 Q 值。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_01_q_learning_pools_and_sarsa/02_SARSA_no_pool.ipynb`

---

## 3. On-Policy vs Off-Policy (同策略 vs 異策略)

這是一個核心概念，用來區分演算法的學習方式。

- **On-Policy (同策略)**:
  - **比喻**: 自己親身實踐，從自己犯的錯誤中學習。
  - **特點**: 產生數據的策略和要評估改進的策略是**同一個**。它必須自己去探索環境，並為自己的行為負責。
  - **代表**: SARSA, PPO

- **Off-Policy (異策略)**:
  - **比喻**: 觀看世界冠軍的棋譜來學習，不管自己當前棋藝如何。
  - **特點**: 產生數據的策略（例如，帶有隨機探索的行為策略）和要學習的目標策略（例如，完全貪婪的最優策略）可以是**不同的**。這使得它可以利用過去的經驗（Replay Buffer）。
  - **代表**: Q-Learning, DQN, DDPG, SAC

---

## 4. n-Step TD 學習

Q-Learning 和 SARSA 都是每走一步就更新一次價值估計，這被稱為 **1-step TD**。**n-step TD** 則是一個更廣義的方法，它在更新當前狀態的價值時，會往前看 `n` 步的真實獎勵，並結合第 `n` 步之後的狀態價值估計。

- **n-step 回報 (Return)**:
  $$
  G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n}, a_{t+n})
  $$
- **優點**: 它是單步 TD (低變異數，但偏差較大) 和蒙地卡羅 (無偏差，但變異數高) 之間的一個權衡，通常能加速學習過程。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_01_q_learning_pools_and_sarsa/03_n_step_TD_learning.ipynb`
