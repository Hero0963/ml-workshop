"""
井字遊戲狀態空間分析腳本 (Tic-Tac-Toe State Space Analysis)

這個腳本從空白棋盤出發，精確計算：
  1. 唯一合法盤面數（用 set 去重）
  2. 完整對弈路徑數（遊戲場次）
  3. 暴力 Minimax vs Alpha-Beta 實際各走過幾個節點（親手測量！）

執行方式：
  uv run python scripts/count_states.py
"""


# ──────────────────────────────────────────────
#  基礎工具函數
# ──────────────────────────────────────────────


def check_winner(board: list[list[int]]) -> int | None:
    """回傳 1 (X 贏)、-1 (O 贏)、0 (平手)、None (進行中)。"""
    for i in range(3):
        if abs(sum(board[i])) == 3:
            return 1 if sum(board[i]) > 0 else -1
        col = board[0][i] + board[1][i] + board[2][i]
        if abs(col) == 3:
            return 1 if col > 0 else -1
    d1 = board[0][0] + board[1][1] + board[2][2]
    if abs(d1) == 3:
        return 1 if d1 > 0 else -1
    d2 = board[0][2] + board[1][1] + board[2][0]
    if abs(d2) == 3:
        return 1 if d2 > 0 else -1
    if all(board[r][c] != 0 for r in range(3) for c in range(3)):
        return 0
    return None


def board_to_tuple(board: list[list[int]]) -> tuple:
    """將盤面轉為可 hash 的 tuple，用來去重。"""
    return tuple(board[r][c] for r in range(3) for c in range(3))


# ──────────────────────────────────────────────
#  Part 1：唯一盤面數 & 對弈路徑數
# ──────────────────────────────────────────────


def count_unique_states() -> tuple[int, int, int]:
    """
    用 DFS + set 計算：
    - unique_states: 去重後的唯一合法盤面數
    - terminal_count: 終局的唯一盤面數
    - game_paths: 完整對弈路徑數（從頭到尾的場次）
    """
    seen: set[tuple] = set()
    game_paths = 0

    def dfs(board: list[list[int]], is_x_turn: bool) -> None:
        nonlocal game_paths
        seen.add(board_to_tuple(board))

        winner = check_winner(board)
        if winner is not None:
            game_paths += 1
            return

        player = 1 if is_x_turn else -1
        has_move = False
        for r in range(3):
            for c in range(3):
                if board[r][c] == 0:
                    has_move = True
                    board[r][c] = player
                    dfs(board, not is_x_turn)
                    board[r][c] = 0
        if not has_move:
            game_paths += 1

    dfs([[0] * 3 for _ in range(3)], True)

    # 計算終局盤面數（直接用 set 裡面的元素驗證）
    terminal_count = sum(
        1
        for s in seen
        if check_winner([[s[r * 3 + c] for c in range(3)] for r in range(3)])
        is not None
    )

    return len(seen), terminal_count, game_paths


# ──────────────────────────────────────────────
#  Part 2：暴力 Minimax 節點計數
# ──────────────────────────────────────────────


def minimax(board: list[list[int]], is_maximizing: bool, counter: list[int]) -> float:
    """標準 Minimax，每次呼叫讓 counter[0] += 1 以計算節點數。"""
    counter[0] += 1
    winner = check_winner(board)
    if winner == 1:
        return 10.0
    if winner == -1:
        return -10.0
    if winner == 0:
        return 0.0

    scores = []
    player = 1 if is_maximizing else -1
    for r in range(3):
        for c in range(3):
            if board[r][c] == 0:
                board[r][c] = player
                scores.append(minimax(board, not is_maximizing, counter))
                board[r][c] = 0

    if not scores:
        return 0.0
    return max(scores) if is_maximizing else min(scores)


# ──────────────────────────────────────────────
#  Part 3：Alpha-Beta 節點計數
# ──────────────────────────────────────────────


def alphabeta(
    board: list[list[int]],
    is_maximizing: bool,
    alpha: float,
    beta: float,
    counter: list[int],
) -> float:
    """Alpha-Beta 剪枝版，每次呼叫讓 counter[0] += 1 以計算節點數。"""
    counter[0] += 1
    winner = check_winner(board)
    if winner == 1:
        return 10.0
    if winner == -1:
        return -10.0
    if winner == 0:
        return 0.0

    player = 1 if is_maximizing else -1
    best = -float("inf") if is_maximizing else float("inf")

    for r in range(3):
        for c in range(3):
            if board[r][c] == 0:
                board[r][c] = player
                score = alphabeta(board, not is_maximizing, alpha, beta, counter)
                board[r][c] = 0
                if is_maximizing:
                    best = max(best, score)
                    alpha = max(alpha, best)
                else:
                    best = min(best, score)
                    beta = min(beta, best)
                if alpha >= beta:
                    break  # 剪枝！

    return best if best not in [float("inf"), -float("inf")] else 0.0


# ──────────────────────────────────────────────
#  主程式
# ──────────────────────────────────────────────


def main() -> None:
    print("=" * 52)
    print("🔢 井字遊戲狀態空間分析（精確版）")
    print("=" * 52)

    print("\n⏳ 計算唯一盤面數中（需幾秒）...")
    unique, terminal, paths = count_unique_states()

    print(f"\n  理論上限（3^9）          : {3**9:>10,}")
    print(f"  唯一合法盤面數（去重後）  : {unique:>10,}")
    print(f"    ├─ 對弈中的盤面         : {unique - terminal:>10,}")
    print(f"    └─ 終局盤面             : {terminal:>10,}")
    print(f"\n  完整對弈路徑數（遊戲場次）: {paths:>10,}")

    print("\n" + "=" * 52)
    print("⚔️  Minimax vs Alpha-Beta 節點數實測（空盤面出發）")
    print("=" * 52)

    board = [[0] * 3 for _ in range(3)]

    mm_counter = [0]
    minimax(board, True, mm_counter)

    ab_counter = [0]
    alphabeta(board, True, -float("inf"), float("inf"), ab_counter)

    reduction = (1 - ab_counter[0] / mm_counter[0]) * 100

    print(f"\n  暴力 Minimax 走過的節點數 : {mm_counter[0]:>10,}")
    print(f"  Alpha-Beta 走過的節點數   : {ab_counter[0]:>10,}")
    print(f"  節省比例                  : {reduction:>9.1f}%")
    print()


if __name__ == "__main__":
    main()
