# Chap 04 - Actor-Critic (AC, A2C, A3C)

本章節深入探討了強化學習中一個非常重要且強大的框架——Actor-Critic (AC)。與前幾章的價值學習 (Q-Learning) 或策略學習 (REINFORCE) 不同，AC 框架結合了兩者的優點，透過一個「演員 (Actor)」來學習策略，並由一個「評論家 (Critic)」來評估策略的好壞，從而實現更穩定、更高效的學習。

我們將從基本的 Actor-Critic 概念出發，逐步演進到更高級的 A2C (Advantage Actor-Critic) 和 A3C (Asynchronous Advantage Actor-Critic) 演算法。

---

## 1. Actor-Critic (AC) 框架

AC 是一個通用的**框架**，其核心是包含兩個主要部分：

-   **`Actor` (演員)**: 一個策略網路，負責根據當前狀態 `state` 決定要採取的 `action`。它的目標是最大化長期回報。
-   **`Critic` (評論家)**: 一個價值網路，負責評估 `Actor` 所選擇的動作的好壞。它學習的是狀態價值函數 `V(s)` 或狀態-動作價值函數 `Q(s, a)`。

**工作流程**:
1.  `Actor` 根據當前 `state` 輸出一個 `action`。
2.  `Critic` 根據 `Actor` 的表現給出一個評分（例如，TD 誤差或 Advantage）。
3.  `Actor` 根據 `Critic` 的評分來更新自己的策略。如果評分是正面的，就增加該 `action` 的機率；反之則減少。

這種方式解決了 REINFORCE 演算法中梯度變異數過大的問題，因為 `Critic` 提供了一個相對穩定的基線 (baseline) 來進行評估。

---

## 2. Advantage Actor-Critic (A2C)

A2C 是 AC 框架下一個非常成功且流行的具體實現。它的主要貢獻是引入了 **Advantage (優勢)** 函數作為 `Critic` 的評分標準。

-   **Advantage Function**: `A(s, a) = Q(s, a) - V(s)`
    -   在實作中，我們通常用 `(r + γ * V(s')) - V(s)` 來估計 Advantage，其中 `r + γ * V(s')` 就是 TD-Target。
    -   這個值衡量了在狀態 `s` 下，採取動作 `a` 相對於「平均表現」有多好。

**優點**:
-   **降低變異數**: Advantage 作為一個相對值，而不是絕對的 Q-value，可以顯著降低策略梯度的變異數，使訓練過程更加穩定。
-   **同步更新**: A2C 採用同步更新策略，等待一個批次的數據收集完成後，統一進行計算和更新，這在現代 GPU 硬體上通常效率很高。

我們在 `02_baseline_a2c.ipynb` 中實現的就是一個基於 Advantage 概念的 Actor-Critic 模型。

---

## 3. Asynchronous Advantage Actor-Critic (A3C)

A3C 是 A2C 的前身，也是深度強化學習領域的重大突破之一。它與 A2C 的核心思想一致，都使用 Advantage，但實現方式是**異步 (Asynchronous)** 的。

**工作流程**:
-   A3C 會建立一個全域的中央網路 (Global Network)。
-   同時啟動多個並行的「工人 (Worker)」，每個 Worker 都有自己的環境和模型副本。
-   這些 Worker **獨立、並行**地與環境互動，並各自計算梯度。
-   梯度計算完成後，Worker 會**異步地**將其推送給中央網路來更新全域參數。

**優點**:
-   **打破數據相關性**: 來自不同 Worker 的數據具有天然的多樣性，這起到了類似 Replay Buffer 的作用，有效打破了數據之間的相關性，讓學習過程非常穩定。在當時，這使得在不使用 Replay Buffer 的情況下訓練大型網路成為可能。

---

## 4. 整合 Replay Buffer

在我們的最終版本 `04_a2c_replay_buffer` 中，我們將 Replay Buffer（數據池）整合進了基於 `02` 的 Actor-Critic 架構中。

**訓練流程**:
1.  **預填充**: 程式開始時，先運行數百局遊戲，將大量的初始經驗數據存入 Replay Buffer。
2.  **訓練與更新**:
    -   在每一個訓練輪次 (epoch) 中，先運行幾局新的遊戲，將最新的經驗數據也加入 Buffer，保持數據的新鮮度。
    -   然後，從 Buffer 中**隨機抽樣一個批次 (batch)** 的數據。
    -   使用這個批次的數據來訓練 `Actor` 和 `Critic` 模型。

這種「先填充，後訓練並持續更新」的模式，結合了 Replay Buffer 和 Actor-Critic 的優點，是一種非常強大且穩健的訓練策略。