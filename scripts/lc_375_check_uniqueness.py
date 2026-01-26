def solve_check(arr: list[int]):
    arr.sort()
    n = len(arr)
    dp = [[0] * n for _ in range(n)]

    # We need to compute DP iteratively
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            # 1. Find the true min cost first
            local_min = float("inf")
            temp_costs = {}

            for k_idx in range(i, j + 1):
                val_k = arr[k_idx]
                cost_left = dp[i][k_idx - 1] if k_idx > i else 0
                cost_right = dp[k_idx + 1][j] if k_idx < j else 0

                # The cost function
                c = val_k + max(cost_left, cost_right)

                if c < local_min:
                    local_min = c

                temp_costs[arr[k_idx]] = c

            # Store min cost for future use
            dp[i][j] = local_min

            # 2. Check for multiple optimals
            optimals = []
            for val, c in temp_costs.items():
                if c == local_min:
                    optimals.append(val)

            optimals.sort()
            if len(optimals) > 1:
                print(
                    f"Range {arr[i:j+1]} (indices {i}-{j}) has multiple optimal roots: {optimals} (Cost: {local_min})"
                )

    return dp[0][n - 1]


if __name__ == "__main__":
    arr = [2, 4, 6, 8, 10]
    # arr = list(range(1, 11))
    print(f"Checking for multiple optimal paths in: {arr}")
    solve_check(arr)
