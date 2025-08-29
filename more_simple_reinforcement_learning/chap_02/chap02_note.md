# 強化學習講義教材 - Chap02（DQN 系列）

---

## 1. DQN 動機與定義（從 Q-Table 到神經網路）
- **為什麼需要 DQN**：當狀態或動作空間太大（像影像或連續狀態）時，Q-Table 無法窮舉；DQN 用**神經網路近似 Q 值函數** $(Q_\theta(s,a)$，直接輸入狀態、輸出所有動作的 Q 值。
- **兩個穩定化關鍵**：
  1) **Experience Replay**：打亂相鄰樣本的相關性；  
  2) **Target Network**：用一個**延遲更新**的網路 $(Q_{\theta^-}$ 來產生目標，避免「邊學邊改目標」的震盪。
---

## 2. 基礎 DQN（流程與實作要點）
1. **互動蒐集** transition：$(s,a,r,s',\text{done})$ 存入 replay buffer。  
2. **取樣** mini-batch（sample 那段就是在取一個 **batch**）。  

---

## 3. 雙模型（Target Network，避免「自我追逐」）
- **概念**：同時維持 `online` $ Q_\theta $ 與 `target` $ Q_{\theta^-}$。訓練時用 `target` 產生 \(y\)，**較少更新**以穩定學習。  
以「`model` / `model_delay`」描述此事：`model_delay` 就是 target network（延遲更新）。

---

## 4. 加權數據池（Prioritized Experience Replay, PER）
- **動機**：不是每個 transition 都同等重要；**TD 誤差** \(|\delta|\) 大者更值得常被重播。  
- **抽樣機率**（Proportional PER）：$p_i=(|\delta_i|+\varepsilon)^\alpha,\; P(i)=\frac{p_i}{\sum_k p_k}$。  
- **重要性權重**（修正偏差）：$
w_i=\big(\frac{1}{N}\frac{1}{P(i)}\big)^\beta / \max_j w_j
$，常將 \(w_i\) 乘在 loss 上。  
- 為了緩解過擬合，**以削減 loss 的方式**實作 per-sample 調整」——這對應到 **importance sampling 的 loss 加權**。  

---

## 5. Double DQN（降低過高估計偏差）
- **問題**：在 DQN 的 \(\max_{a'}\) 會傾向**高估** Q 值。  
- **解法**：用 online 網路**選動作**，用 target 網路**評分**：  
$$
y = r + \gamma\, Q_{\theta^-}\bigl(s', \arg\max_{a'} Q_\theta(s',a')\bigr).
$$
。

---

## 6. Dueling DQN（回答：`fc_state`/`fc_action` 在做什麼？）
- **想法**：將 Q 拆成 **狀態價值**與**動作優勢**兩條分支：
  - `fc_state` 產生 \(V(s)\)，評估「這個**狀態**本身好不好」；  
  - `fc_action` 產生 \(A(s,a)\)，評估「在此狀態下**各動作**相對好多少」。  
- **聚合**（去不可辨識性）：  

---

## 7. Noise DQN（NoisyNet，學會「怎麼探索」）
- **目的**：用**可學的參數噪聲**取代 ε-greedy，讓探索強度隨訓練自動調整。  
- **Noisy 線性層**：\(w=\mu+\sigma\odot\varepsilon\)；常用**因子化高斯**減少參數量。  
- 參考：Fortunato et al., *Noisy Networks for Exploration*（2017）。

---

## 8. 能否組合？（Noise + Dueling + Double … → Rainbow）
- **可以**。實務上常把 Double、Dueling、PER、NoisyNet、n-step、分佈式 Q（如 C51）**一起用**，形成 **Rainbow DQN**，在 Atari 等基準上有顯著表現。[Hessel et al., *Rainbow*（2017/2018）]。

---

## 9. 補充

- **CartPole 輸入/輸出**：輸入 $x\in\mathbb{R}^4$（位置、速度、角度、角速度），輸出 2 維 Q 值（左/右）。  
  
- **`state_dict()`/儲存**：
   - 取權重：`model.state_dict()`；  
   - 存：`torch.save(model.state_dict(), "xxx.pth")`；  
   - 讀：`model.load_state_dict(torch.load("xxx.pth", weights_only=True))`；  
   - 若要**續訓練**，建議存 **checkpoint**（含 optimizer 等）。  
- **target 同步策略**：
   - **固定步數**硬更新（常見且穩定）；  
   - 或 **soft update**（Polyak）。「loss 收斂才同步」偏向啟發式，不如前兩者常見。  
