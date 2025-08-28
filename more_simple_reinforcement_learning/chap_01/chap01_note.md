# 強化學習講義教材 - Chap01

## 1. 強化學習基礎

強化學習 (Reinforcement Learning, RL) 是讓智能體 (Agent) 在環境 (Environment) 中透過試誤 (Trial and Error)，逐步學會一個能最大化「累積獎勵」的策略 (Policy)。

核心元素：
- **Agent**：做決策的學習者
- **Environment**：與 Agent 互動的外部世界
- **State (s)**：環境當前的情況
- **Action (a)**：Agent 可採取的行動
- **Reward (r)**：執行動作後環境回饋的分數
- **Policy (π)**：從狀態選動作的規則
- **Q 值 (Q-function)**：表示在狀態 s 下採取動作 a 的期望累積回報

學習流程：
1. 初始化 Q-table (所有狀態-動作值)
2. Agent 依策略選擇動作
3. 獲得 reward 與下一狀態
4. 更新 Q 值
5. 重複直到收斂

---

## 2. Q-Learning (Off-Policy)

Q-Learning 是最經典的 off-policy TD 學習方法。更新公式：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ r_{t+1} + \gamma \max_a Q(s_{t+1}, a) - Q(s_t, a_t) \big]
$$

特點：
- 使用下一狀態的 **最大 Q 值** 來更新
- 不管實際上下一步選了什麼，都假設自己會選最佳動作
- 具體應用：FrozenLake 4x4，小規模環境可以直接用 Q-table

程式重點 (來自 `01_QLearning.ipynb`)：
- 初始化 Q-table 為零矩陣
- 使用 ε-greedy 策略探索
- 在每回合結束後更新 Q 值
- 記錄回合 reward，觀察收斂

---

## 3. SARSA (On-Policy)

SARSA 也是 TD 方法，但屬於 on-policy。更新公式：

$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \big]
$$

差異：
- Q-Learning 用「理想最佳動作」更新
- SARSA 用「實際選擇的下一動作」更新
- 更貼近實際策略行為，因此學到的策略通常更保守

程式重點 (來自 `02_SARSA_no_pool.ipynb`)：
- 在選擇動作後要先暫存下一個動作
- 更新 Q 值時使用這個實際動作
- 仍然使用 ε-greedy 策略，但更新方向不同

---

## 4. n-Step TD 學習

為了加快學習，可以使用多步 (n-step) TD 方法。基本想法是：
- 不只用一步的回報 (reward)，而是把未來 n 步的 reward 一次納入更新
- 可以在精確度與收斂速度之間取得平衡

n-step return：
$$
G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n}, a_{t+n})
$$

更新公式：
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \big[ G_t^{(n)} - Q(s_t, a_t) \big]
$$

程式重點 (來自 `03_n_step_TD_learning.ipynb`)：
- 使用一個 buffer 儲存最近 n 個 (s,a,r)
- 當收集滿 n 步後，就能計算一次回報
- 提升估計精確度，減少方差

---

## 5. On-Policy vs Off-Policy

- **On-Policy**：只能根據自己「實際」執行的動作來更新 (SARSA, PPO)
- **Off-Policy**：可以根據「理想策略」或「觀察數據」來更新 (Q-Learning, DQN, DDPG, TD3, SAC)

舉例：
- 玩遊戲時自己操作 → on-policy
- 看別人玩遊戲並學習最佳策略 → off-policy

---

## 6. 重點回顧

1. Q-Learning 與 SARSA 都是 TD 方法，只是更新方式不同
2. n-step TD 在 1-step TD 與 Monte Carlo 方法之間提供折衷方案
3. 折扣因子 γ 保證未來回報有限並能收斂
4. 學習率 α 與探索率 ε 對收斂速度和穩定性影響很大

---

## 7. 補充

**SARSA** = State → Action → Reward → next State → next Action

**TD 方法** (Temporal Difference)：  
當我們在互動過程中，可以利用「當下獎勵 + 對未來的估計」來修正 Q 值，不需要等整個 episode 結束。

例子：
- 冰湖 (FrozenLake) 遊戲中，若掉進洞 → 獎勵負面，Q 值下降。
- 若安全通過 → 獎勵為 0，但會加上「未來有可能到終點」的期望。

👉 TD 方法就是「邊走邊學，邊修正估計」。

**Q 表**：所有狀態下作所有動作的分數  
--> 訓練完之後，按照 Q 表執行  
--> 根據對應狀態選擇分數最高的動作
