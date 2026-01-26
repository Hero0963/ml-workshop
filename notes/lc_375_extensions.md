# LeetCode 375 延伸討論 (Extensions)

## 緣起
延續 leetcode 1039，此為區間 DP 題組。  
對範例的 **Minimax 策略樹 (Strategy Tree)** 結構感興趣，也嘗試繪製。

## 前置作業
- **題目**: [375. Guess Number Higher or Lower II](https://leetcode.com/problems/guess-number-higher-or-lower-ii/)
- **解法**: 可用 `dp` 解

## 補充：
原題為 `[1, n]`，我擴展成 `[left, right]` 。  
提供腳本輸出 **.md** file ，以 **Mermaid** 語法繪製 **Minimax 策略樹**。    
希望透過可視化，幫助我們理解決策過程。  
**範例輸出**：[Minimax Strategy Tree [5, 31]](./illustrations/lc_375/decision_tree_5_31.md)  

### Python 實作
我已將連續區間與離散集合的解法整合至同一個腳本中，包含 Mermaid 圖表生成功能。

**Script**: [scripts/lc_375_minimax_strategy.py](./scripts/lc_375_minimax_strategy.py)

#### 執行範例
```powershell
uv run scripts/lc_375_minimax_strategy.py
```

## 延伸思考：離散集合 (Discrete Set)
如果題目變形為：給定一個不重複的集合 `S = {v1, v2, ..., vn}` (如 `[1, 4, 10]`)，解法依然適用。

### Python 實作
因為索引 `i, j` 本身是連續的 (`0` 到 `len(arr)-1`)，所以演算法結構完全不變，只是把原本的 `k` (數值) 替換成 `arr[k]` (對應的值)。

**Script**: [scripts/lc_375_minimax_strategy.py](./scripts/lc_375_minimax_strategy.py)

## 關於最佳路徑的不唯一性 (Non-Uniqueness of Optimal Paths)
在研究過程中發現，對於某些區間，可能存在多個「最佳決策點 (Roots)」，它們都能產生相同的 Minimax Cost。

**腳本**: [scripts/lc_375_check_uniqueness.py](./scripts/lc_375_check_uniqueness.py)

例如集合 `[2, 4, 6, 8, 10]`：
- 選擇 `4` (Index 1) 或 `8` (Index 3) 作為分割點，最終的 Minimax Cost 都是 **12**。
- 單獨看連續區間 `[6..10]`，選擇 `7` 或 `9` 也是等價的。

這意味著 Minimax 策略樹的形狀並非唯一，取決於實作時遇到平手 (Tie) 的選擇策略（例如優先選 Index 小的或大的）。
