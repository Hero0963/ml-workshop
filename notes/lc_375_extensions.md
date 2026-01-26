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

```python

def calculate_dp(left: int, right: int) -> tuple:
    """
    Calculates the Minimax DP table for the 'Guess Number Higher or Lower II' problem.
    Optimized to O(N^2) using decision monotonicity.
    
    Args:
        left (int): The lower bound of the range.
        right (int): The upper bound of the range.
        
    Returns:
        tuple: (dp table, root table)
               dp[i][j]: min cost for range [i, j]
               root[i][j]: the optimal pivot k selected for range [i, j]
    """
    n = right
    size = n + 2
    dp = [[0] * size for _ in range(size)]
    root = [[0] * size for _ in range(size)]

    # Base case initialization
    for i in range(left, right + 1):
        root[i][i] = i

    # Iterate length from 2 to (right - left + 1)
    for length in range(2, right - left + 1 + 1):
        for i in range(left, right - length + 2):
            j = i + length - 1
            
            # Start search for k from the optimal k of the sub-problem [i, j-1]
            # to leverage monotonicity and reduce search space.
            start_k = root[i][j-1] if length > 2 else i
            end_k = j
            
            optimal_pivot = i
            local_min = float('inf')
            
            # Find optimal k that minimizes the maximum cost: k + max(left_cost, right_cost)
            # The cost function is convex-like, so we can stop once the cost starts increasing.
            for k in range(start_k, end_k + 1):
                left_c = dp[i][k-1] if k > i else 0
                right_c = dp[k+1][j] if k < j else 0
                val = k + max(left_c, right_c)
                
                if val < local_min:
                    local_min = val
                    optimal_pivot = k
                
                # Optimization: Once left_cost >= right_cost, the max term becomes left_cost.
                # Since left_cost increases with k, the total cost will only increase from here.
                if right_c <= left_c:
                    break
            
            dp[i][j] = local_min
            root[i][j] = optimal_pivot

    return dp, root

def generate_mermaid(left: int, right: int, dp: list, root: list, graph_lines: list, parent_id: str = None, edge_label: str = ""):
    """
    Recursively builds Mermaid graph syntax.
    Uses deterministic IDs based on path to ensure tree structure.
    """
    if left > right:
        return
        
    if left == right:
        node_id = f"{parent_id}_L{left}" if parent_id else f"L{left}"
        graph_lines.append(f'    {node_id}(("{left}"))')
        if parent_id:
            graph_lines.append(f"    {parent_id} -->|{edge_label}| {node_id}")
        return

    # Retrieve optimal k from pre-computed root table
    k = root[left][right]
    
    # Deterministic ID: "node" + value + path context
    current_id = f"{parent_id}_{k}" if parent_id else f"N{k}"
    
    graph_lines.append(f'    {current_id}(("{k}"))')
    if parent_id:
        graph_lines.append(f"    {parent_id} -->|{edge_label}| {current_id}")
        
    # Recurse
    generate_mermaid(left, k - 1, dp, root, graph_lines, current_id, "Lower")
    generate_mermaid(k + 1, right, dp, root, graph_lines, current_id, "Higher")

def _simple_run():
    # Configuration
    left = 5
    end = 31

    print(f"--- Calculating DP (O(N^2)) for range [{left}, {end}] ---")
    dp_table, root_table = calculate_dp(left, end)
    
    print("--- Generating Mermaid Tree ---")
    mermaid_lines = ["graph TD;"]
    
    generate_mermaid(left, end, dp_table, root_table, mermaid_lines)
    
    output_content = "```mermaid\n" + "\n".join(mermaid_lines) + "\n```"
    output_file = f"decision_tree_{left}_{end}.md"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("# Minimax Strategy Tree\n\n")
        f.write(f"Range: **[{left}, {end}]**\n\n")
        f.write(f"Minimax Cost: **{dp_table[left][end]}**\n\n")
        f.write(output_content)
        
    print(f"Mermaid tree saved to {output_file}")

if __name__ == "__main__":
    _simple_run()

```