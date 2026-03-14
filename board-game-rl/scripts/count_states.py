"""
井字遊戲狀態空間分析腳本 (Tic-Tac-Toe State Space Analysis)

這個腳本從空白棋盤出發，對所有合法對弈路徑進行深度優先搜尋 (DFS)，
精確計算出井字遊戲中：
  1. 總合法盤面數（mid-game + terminal）
  2. 終局盤面數（有人贏，或平手）
  3. 完整對弈路徑數（從頭到尾的每一種下法）

執行方式：
  uv run python scripts/count_states.py
"""

from board_game_rl.agents.alphabeta_agent import AlphaBetaAgent


def check_winner(board: list[list[int]]) -> int | None:
    """
    傳回 1 (X 贏)、-1 (O 贏)、0 (平手)，或 None (遊戲仍在進行)。
    """
    # 橫列、直行
    for i in range(3):
        if abs(sum(board[i])) == 3:
            return 1 if sum(board[i]) > 0 else -1
        col_sum = board[0][i] + board[1][i] + board[2][i]
        if abs(col_sum) == 3:
            return 1 if col_sum > 0 else -1

    # 對角線
    d1 = board[0][0] + board[1][1] + board[2][2]
    if abs(d1) == 3:
        return 1 if d1 > 0 else -1
    d2 = board[0][2] + board[1][1] + board[2][0]
    if abs(d2) == 3:
        return 1 if d2 > 0 else -1

    # 平手（棋盤全滿但無人獲勝）
    if all(board[r][c] != 0 for r in range(3) for c in range(3)):
        return 0

    return None  # 遊戲還在進行


def count_all_states() -> tuple[int, int, int]:
    """
    使用 DFS 遍歷所有合法盤面，並回傳：
    (total_states, terminal_states, game_paths)
    """
    total_states = 0
    terminal_states = 0
    game_paths = 0

    def dfs(board: list[list[int]], is_x_turn: bool) -> None:
        nonlocal total_states, terminal_states, game_paths
        total_states += 1

        winner = check_winner(board)
        if winner is not None:
            terminal_states += 1
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
            game_paths += 1  # 棋盤滿了但上面 check_winner 應已經處理了

    empty = [[0] * 3 for _ in range(3)]
    dfs(empty, True)
    return total_states, terminal_states, game_paths


def main() -> None:
    print("=" * 50)
    print("🔢 井字遊戲狀態空間分析")
    print("=" * 50)

    total, terminal, paths = count_all_states()
    ongoing = total - terminal

    print(f"\n  理論上限（3^9）          : {3**9:>10,}")
    print(f"  實際合法盤面總數          : {total:>10,}")
    print(f"    ├─ 對弈中的盤面         : {ongoing:>10,}")
    print(f"    └─ 終局盤面             : {terminal:>10,}")
    print(f"\n  完整對弈路徑數（遊戲場次）: {paths:>10,}")

    print("\n" + "=" * 50)
    print("📌 結論")
    print("=" * 50)
    print(f"""
  Alpha-Beta 剪枝前（暴力 Minimax 最壞情況）  : ~{paths:,} 節點
  Alpha-Beta 剪枝後（最佳排序，估計值）        : ~1,000 – 6,000 節點
  節省比例                                     : >95%
""")

    # 彩蛋：讓 Alpha-Beta Agent 真的對第一步計算，並印出它的推理日誌
    print("=" * 50)
    print("🤖 Alpha-Beta 第一步選擇示範（空盤面）")
    print("=" * 50)
    import numpy as np

    agent = AlphaBetaAgent(player=1)
    empty_board = np.zeros((3, 3), dtype=int)
    action = agent.act(empty_board)
    r, c = action // 3, action % 3
    print(f"\n  ➔ 最終選擇：落子於 ({r}, {c})\n")


if __name__ == "__main__":
    main()
