def solve_continuous(left: int, right: int) -> tuple:
    """
    Calculates the Minimax DP table for the continuous version of 'Guess Number Higher or Lower II'.
    Optimized to O(N^2) using decision monotonicity.

    Args:
        left (int): The lower bound of the range.
        right (int): The upper bound of the range.

    Returns:
        tuple: (dp table, root table)
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
            start_k = root[i][j - 1] if length > 2 else i
            end_k = j

            optimal_pivot = i
            local_min = float("inf")

            for k in range(start_k, end_k + 1):
                left_c = dp[i][k - 1] if k > i else 0
                right_c = dp[k + 1][j] if k < j else 0
                val = k + max(left_c, right_c)

                if val < local_min:
                    local_min = val
                    optimal_pivot = k

                # Optimization
                if right_c <= left_c:
                    break

            dp[i][j] = local_min
            root[i][j] = optimal_pivot

    return dp, root


def solve_discrete(arr: list[int]) -> tuple:
    """
    Calculates the Minimax cost for a discrete set of numbers.
    Args:
        arr (list[int]): A list of distinct integers.
    Returns:
        tuple: (min_cost, dp_table, root_table)
    """
    arr.sort()
    n = len(arr)
    dp = [[0] * n for _ in range(n)]
    root = [[0] * n for _ in range(n)]

    # Base case initialization
    for i in range(n):
        root[i][i] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            local_min = float("inf")
            optimal_k_idx = -1

            for k in range(i, j + 1):
                cost_left = dp[i][k - 1] if k > i else 0
                cost_right = dp[k + 1][j] if k < j else 0

                cost = arr[k] + max(cost_left, cost_right)

                if cost < local_min:
                    local_min = cost
                    optimal_k_idx = k

            dp[i][j] = local_min
            root[i][j] = optimal_k_idx

    return dp[0][n - 1], dp, root


def generate_mermaid_continuous(
    left: int,
    right: int,
    root: list,
    graph_lines: list,
    parent_id: str = None,
    edge_label: str = "",
):
    if left > right:
        return

    if left == right:
        node_id = f"{parent_id}_L{left}" if parent_id else f"L{left}"
        graph_lines.append(f'    {node_id}(("{left}"))')
        if parent_id:
            graph_lines.append(f"    {parent_id} -->|{edge_label}| {node_id}")
        return

    k = root[left][right]
    current_id = f"{parent_id}_{k}" if parent_id else f"N{k}"

    graph_lines.append(f'    {current_id}(("{k}"))')
    if parent_id:
        graph_lines.append(f"    {parent_id} -->|{edge_label}| {current_id}")

    generate_mermaid_continuous(left, k - 1, root, graph_lines, current_id, "Lower")
    generate_mermaid_continuous(k + 1, right, root, graph_lines, current_id, "Higher")


def generate_mermaid_discrete(
    i: int,
    j: int,
    arr: list[int],
    root: list,
    graph_lines: list,
    parent_id: str = None,
    edge_label: str = "",
):
    if i > j:
        return

    if i == j:
        val = arr[i]
        node_id = f"{parent_id}_L{val}" if parent_id else f"L{val}"
        graph_lines.append(f'    {node_id}(("{val}"))')
        if parent_id:
            graph_lines.append(f"    {parent_id} -->|{edge_label}| {node_id}")
        return

    k_idx = root[i][j]
    val = arr[k_idx]
    current_id = f"{parent_id}_{val}" if parent_id else f"N{val}"

    graph_lines.append(f'    {current_id}(("{val}"))')
    if parent_id:
        graph_lines.append(f"    {parent_id} -->|{edge_label}| {current_id}")

    generate_mermaid_discrete(i, k_idx - 1, arr, root, graph_lines, current_id, "Lower")
    generate_mermaid_discrete(
        k_idx + 1, j, arr, root, graph_lines, current_id, "Higher"
    )


def run_continuous_demo(left: int, right: int, generate_md: bool = False):
    print(f"--- Continuous Range [{left}, {right}] ---")
    dp_table, root_table = solve_continuous(left, right)
    print(f"Minimax Cost: {dp_table[left][right]}")

    if generate_md:
        mermaid_lines = ["graph TD;"]
        generate_mermaid_continuous(left, right, root_table, mermaid_lines)

        output_content = "```mermaid\n" + "\n".join(mermaid_lines) + "\n```"
        output_file = f"decision_tree_{left}_{right}.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Minimax Strategy Tree (Continuous)\n\n")
            f.write(f"Range: **[{left}, {right}]**\n\n")
            f.write(f"Minimax Cost: **{dp_table[left][right]}**\n\n")
            f.write(output_content)
        print(f"Saved to {output_file}")


def run_discrete_demo(arr: list[int], generate_md: bool = False):
    print(f"--- Discrete Set {arr} ---")
    min_cost, dp, root = solve_discrete(arr)
    print(f"Minimax Cost: {min_cost}")

    if generate_md:
        mermaid_lines = ["graph TD;"]
        n = len(arr)
        generate_mermaid_discrete(0, n - 1, arr, root, mermaid_lines)

        output_content = "```mermaid\n" + "\n".join(mermaid_lines) + "\n```"
        output_file = "decision_tree_discrete.md"

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("# Minimax Strategy Tree (Discrete)\n\n")
            f.write(f"Set: **{arr}**\n\n")
            f.write(f"Minimax Cost: **{min_cost}**\n\n")
            f.write(output_content)
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    # Continuous Example
    run_continuous_demo(5, 31, generate_md=False)

    # Discrete Example
    run_discrete_demo([1, 3, 5], generate_md=False)
    run_discrete_demo([2, 4, 6, 8, 10], generate_md=True)
