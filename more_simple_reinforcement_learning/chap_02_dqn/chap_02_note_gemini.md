# 強化學習講義 - Chap02：深度 Q 網路 (DQN) 家族 (Gemini 整理版)

當狀態空間變得巨大（例如，遊戲畫面像素）或連續時，Q-Table 不再適用。**深度 Q 網路 (Deep Q-Network, DQN)** 應運而生，其核心思想是用一個神經網路來近似 Q 值函數 $Q_\theta(s,a)$，我們稱之為「Q 網路」。本章節將探討 DQN 及其重要的改進，這些改進共同構成了現代深度強化學習的基石。

---

## 1. 從 Q-Table 到 DQN

- **動機**: 解決 Q-Table 無法處理高維度或連續狀態空間的問題。
- **核心**: 使用神經網路 $Q_\theta(s, \cdot)$ 作為一個函數近似器。輸入是狀態 `s`，輸出是所有可能動作的 Q 值。
- **兩個關鍵穩定技巧**:
  1.  **經驗回放 (Experience Replay)**: 將智能體的經歷 `(s, a, r, s', done)` 存儲在一個數據池 (Replay Buffer) 中。訓練時，從中隨機採樣一個 mini-batch 來訓練網路。這打破了連續樣本之間的相關性，使訓練更穩定。
  2.  **目標網路 (Target Network)**: 見下一節。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_02_dqn/01_DQN.ipynb`

---

## 2. Target Network (雙模型延遲更新)

- **問題**: 在 DQN 中，計算目標 Q 值和要更新的 Q 值都依賴於同一個網路。這就像「自己追逐自己的尾巴」，目標不斷變動，導致訓練非常不穩定。
- **解決方案**: 使用兩個網路：
  - **主網路 (Online Network)** $Q_\theta$: 正常訓練，每一步都更新。
  - **目標網路 (Target Network)** $Q_{\theta^-}$: 參數**定期**從主網路複製而來，保持一段時間不變。在計算目標值時，使用這個固定的目標網路。
- **目標 Q 值計算**:
  $$ y = r + \gamma \max_{a'} Q_{\theta^-}(s', a') $$
- **優點**: 提供了一個穩定的學習目標，極大地提升了 DQN 的穩定性。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_02_dqn/02_DQN_model_delay.ipynb`

---

## 3. Double DQN (緩解最大化偏差)

- **問題**: DQN 中的 `max` 操作會傾向於選擇被**過高估計**的 Q 值，導致學習到的 Q 值系統性地偏高，這種現象稱為「最大化偏差」。
- **解決方案**: 將**動作選擇**和**價值評估**解耦。
  1.  使用**主網路** $Q_\theta$ 來選擇在下一狀態 `s'` 時的最佳動作 $a^*$。
  2.  使用**目標網路** $Q_{\theta^-}$ 來評估這個被選中動作 $a^*$ 的價值。
- **目標 Q 值計算**:
  $$ y = r + \gamma\, Q_{\theta^-}\bigl(s', \arg\max_{a'} Q_\theta(s',a')\bigr) $$
- **優點**: 有效地緩解了 Q 值的過高估計問題，使得學習更準確。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_02_dqn/04_double_DQN.ipynb`

---

## 4. Dueling DQN (V(s) + A(s,a) 分離)

- **動機**: 有時，一個狀態本身的價值比在該狀態下選擇哪個特定動作更重要。Dueling DQN 將 Q 值分解為兩部分來分別學習。
- **核心結構**:
  - **價值流 (Value Stream)**: 學習狀態價值函數 $V(s)$，評估「處於這個狀態有多好」。
  - **優勢流 (Advantage Stream)**: 學習優勢函數 $A(s,a)$，評估「在這個狀態下，選擇動作 `a` 相對於其他動作有多好」。
- **Q 值聚合**:
  $$ Q(s,a) = V(s) + \left( A(s,a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s,a') \right) $$
  （減去均值是為了穩定性）
- **優點**: 能夠更有效地學習狀態的內在價值，尤其是在動作對環境影響不大的情況下，能顯著加速學習。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_02_dqn/05_dueling_DQN.ipynb`

---

## 5. Prioritized Experience Replay (PER, 加權數據池)

- **動機**: 智能體的經歷並非同等重要。那些讓智能體感到「驚訝」的經歷（即 TD 誤差大的經歷）更值得學習。
- **核心思想**:
  - **優先級採樣**: 根據 TD 誤差 $|\delta|$ 的大小來賦予每個樣本一個優先級，優先級越高的樣本越容易被抽中。
  - **重要性採樣 (IS) 權重**: 為了修正因優先級採樣帶來的偏差，需要為每個樣本計算一個重要性權重 `w`，並在計算損失時應用它。
- **優點**: 大幅提升了數據利用效率，讓學習更快速、更有效。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_02_dqn/03_DQN_weight_replay_buffer.ipynb`

---

## 6. Noisy DQN (NoisyNet, 噪聲網路)

- **動機**: 傳統的 $\epsilon$-greedy 探索方式比較盲目，且很難確定最佳的 $\epsilon$ 值。NoisyNet 旨在讓網路**自己學會如何探索**。
- **核心思想**: 將傳統的線性層替換為**噪聲線性層 (NoisyLinear)**。在這些層的權重和偏置上加入可學習的參數化高斯噪聲。在訓練時，噪聲使得策略本身具有隨機性；在測試時，關閉噪聲即可得到確定性策略。
- **優點**: 提供了一種更智能、狀態依賴的探索機制，在許多任務上表現優於 $\epsilon$-greedy。
- **對應程式碼**: `more_simple_reinforcement_learning/chap_02_dqn/06_noise_DQN.ipynb`

---

### 總結: Rainbow DQN

這些改進並非互斥。將 Double DQN, Dueling DQN, PER, NoisyNet 等多種技術結合起來，就形成了著名的 **Rainbow DQN** 演算法，在 Atari 等標準測試環境中取得了非常出色的表現。
