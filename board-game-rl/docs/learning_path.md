# Board Game RL: 學習與實作路線圖 (Learning Path)

本文件設計了從最基礎的完全資訊(Perfect Information)小遊戲，逐步進階到不完全資訊(Imperfect Information)複雜遊戲的強化學習（RL）研究路線。

每個階段都分為三大核心模組：**Theory (理論研究)**、**Implementation (Jupyter Notebook 實作)**、**Experiment (實驗與優化)**。

---

## Stage 1: 基礎暖身 —— 井字遊戲 (Tic-Tac-Toe)

作為 RL 的 "Hello World"，我們從 Tic-Tac-Toe 開始，理解最基礎的對抗演算法。

- **Theory (理論)**
  - **Minimax Algorithm & Alpha-Beta Pruning (剪枝)**：完美的數學解法，如何建立 Game Tree 並進行 Minimax 搜索。
  - **Q-Learning (Tabular RL)**：當狀態空間極小（$3^9$）時，如何使用查表法讓 AI 自己學會下棋。
  - **Deep Q-Network (DQN)**：將 Q-Table 替換成神經網路的基礎概念。
- **Implementation (實作)**
  - ~~使用現有的 `TicTacToeEnv` 介面~~ ✅
  - ~~實作 `MinimaxAgent` (Alpha-Beta) 確保能下出不敗棋局~~ ✅
  - ~~實作 Tabular Q-Learning 並達到不敗~~ ✅
  - ~~實作 `DQNAgent` 與 PyTorch (MLP + Replay Buffer + Target Network)~~ ✅
- **Experiment (實驗)**
  - ~~比較 DQN 與 Tabular Q-Learning 的勝率差距~~ ✅ (DQN vs Random 後手 ~2.5% 敗率，Q-Learning 零敗)
  - ~~調整 epsilon-greedy 策略觀察收斂速度~~ ✅

---

## Stage 2: 國際象棋突破 —— Chess (Stockfish & NNUE)

國際象棋狀態空間巨大，傳統搜索演算法碰壁，開始導入神經網路做盤面評估 (Evaluation)。

- **Theory (理論)**
  - **Stockfish NNUE (Efficiently Updatable Neural Network)**：
    - [Reference: Stockfish NNUE Architecture](https://www.chessprogramming.org/NNUE)
    - 學習為何在 CPU 上 NNUE 可以這麼快（Quantization, Incremental Updates）。
  - **Bitboards (位元棋盤)**：如何用 64-bit 整數極速表示棋盤狀態與生成合法步。
- **Implementation (實作 - `ipynb`)**
  - 實例探討：將棋盤狀態轉換為 NNUE 可以吃進去的 Sparse Feature (特徵編碼)。
  - 寫一個極簡版的 Evaluation 網路並進行效能測試。
- **Experiment (實驗)**
  - 評估單純 Alpha-Beta 剪枝 vs 加上 NNUE 評估函數的節點搜索深度與勝率提升。

---

## Stage 3: 深度強化學習巔峰 —— 圍棋 (Go / AlphaGo / AlphaZero)

圍棋擁有 $10^{170}$ 種變化，無法單靠搜索，需要依賴 Policy & Value Networks。

- **Theory (理論)**
  - **Monte Carlo Tree Search (MCTS)**：蒙地卡羅樹搜索的機制 (Selection, Expansion, Simulation, Backpropagation)。
  - **AlphaGo / AlphaZero 論文**：
    - *Mastering the game of Go with deep neural networks and tree search (Silver et al., 2016)*
    - *Mastering the game of Go without human knowledge (Silver et al., 2017)*
    - 學習如何將 MCTS 與 ResNet 神經網路結合，以及 PUCT (Predictor + UCB) 演算法。
- **Implementation (實作 - `ipynb`)**
  - 實作一個基礎的 Pure MCTS (無神經網路) 來玩 Tic-Tac-Toe 或縮小版的圍棋 (如 9x9 Board)。
  - 實作 AlphaZero 架構中的 Dual-head ResNet (同時輸出 Actor Policy 與 Critic Value)。
- **Experiment (實驗)**
  - 觀察 MCTS 在不同 Simulation 次數下的棋力變化。
  - (進階) 在小棋盤上跑幾輪 Self-play 訓練流程。

---

## Stage 4: 中國象棋 —— Xiangqi (Pikacat)

結合前面所學，挑戰具有獨特規則（如「憋馬腿」、「將帥不能碰面」）的中國象棋。

- **Theory (理論)**
  - **Pikacat Engine**：學習開源引擎如何結合 NNUE 與 Lazy-SMP 多執行緒搜索。
    - [Reference: PikaCat-OuO/ChineseChess](https://github.com/PikaCat-OuO/ChineseChess)
  - **No-Search RL (無搜索 AI)**：學習最新論文 *Mastering Chinese Chess AI (Xiangqi) Without Search* 如何單靠 Transformer 架構打敗高段位人類。
- **Implementation (實作 - `ipynb`)**
  - 中國象棋演算法的規則編寫挑戰 (Rule Engine)。
  - 探討 Transformer 如何用於序列化的棋盤決策機制。
- **Experiment (實驗)**
  - 比較 AlphaZero 架構 (使用 MCTS) 與純神經網路 (無搜索) 在推理速度與強度的 Trade-off。

---

## Stage 5: 不完全資訊博弈領域 —— 麻將 (Mahjong / Suphx / 天鳳)

麻將不僅是多人遊戲，還充滿了隨機性與隱藏資訊（摸牌、別人手牌未知），是 AI 領域的極大挑戰。

- **Theory (理論)**
  - **Imperfect Information Games**：不完全資訊賽局的挑戰與應對。
  - **Microsoft Suphx (Super Phoenix) 論文**：
    - *Suphx: Mastering Mahjong with Deep Reinforcement Learning*
    - 學習三個核心技術：**Global Reward Prediction** (全局獎勵預測)、**Oracle Guiding** (上帝視角引導訓練)、**Run-time Policy Adaptation** (運行時策略自適應)。
- **Implementation (實作 - `ipynb`)**
  - 將隱藏資訊進行特徵工程與狀態編碼 (State Encoding)。
  - 實作 Oracle Guiding 概念：先讓 AI 看得見所有人的牌進行訓練，再逐步遮蔽資訊 (Dropout)。
- **Experiment (實驗)**
  - 在簡化版麻將環境中，比較有/無 Oracle Guiding 訓練的速度與勝率差異。
  - 觀察 AI 是否能發展出「防守」概念 (減少放銃率)。

---

## 🎯 Next Steps
**Stage 1 已完成！** Alpha-Beta、Q-Learning (不敗)、DQN 全部實作完畢。
下一步進入 **Stage 2/3** — MCTS 與 AlphaZero。
