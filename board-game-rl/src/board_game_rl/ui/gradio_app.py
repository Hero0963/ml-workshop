"""
Gradio Frontend for Board Game RL API.
"""

import gradio as gr

from board_game_rl.api.inference import get_optimal_move
from board_game_rl.games.tic_tac_toe.engine import TicTacToeEngine


def get_board_values(engine: TicTacToeEngine) -> list[str]:
    chars = {1: "X", -1: "O", 0: " "}
    return [chars[engine.board[i // 3][i % 3]] for i in range(9)]


def get_winner_msg(engine: TicTacToeEngine, human_player: int) -> str:
    if engine.winner == human_player:
        return "You Win!"
    if engine.winner == -human_player:
        return "Agent Wins!"
    if engine.winner == 0:
        return "It's a Draw!"
    return ""


def parse_action(
    engine: TicTacToeEngine,
    agent_type: str,
    human_player: int,
    cell_idx: int,
) -> list:
    if engine.winner is not None:
        return [engine, human_player, "Game over! Click New Game."] + get_board_values(
            engine
        )

    if engine.current_player != human_player:
        return [engine, human_player, "Not your turn!"] + get_board_values(engine)

    r, c = cell_idx // 3, cell_idx % 3
    valid = engine.step((r, c))
    if not valid:
        return [engine, human_player, "Invalid move!"] + get_board_values(engine)

    if engine.winner is not None:
        return [
            engine,
            human_player,
            get_winner_msg(engine, human_player),
        ] + get_board_values(engine)

    # Agent turn
    action = get_optimal_move(engine.board, engine.current_player, agent_type)
    if action != -1:
        engine.step((action // 3, action % 3))

    side = "X" if human_player == 1 else "O"
    msg = f"Your turn ({side})."
    if engine.winner is not None:
        msg = get_winner_msg(engine, human_player)

    return [engine, human_player, msg] + get_board_values(engine)


def reset_game(agent_type: str, side_choice: str) -> list:
    engine = TicTacToeEngine()
    human_player = 1 if side_choice == "先手 (X)" else -1

    if human_player == -1:
        action = get_optimal_move(engine.board, engine.current_player, agent_type)
        if action != -1:
            engine.step((action // 3, action % 3))

    side = "X" if human_player == 1 else "O"
    return [
        engine,
        human_player,
        f"New Game! You are {side}. Playing against: {agent_type}",
    ] + get_board_values(engine)


def make_handler(idx: int):
    def handler(engine: TicTacToeEngine, agent_type: str, human_player: int):
        return parse_action(engine, agent_type, human_player, idx)

    return handler


css = """
/* 強制按鈕為完美的正方形 */
.square-btn { 
    font-size: 3em !important; 
    height: 100px !important; 
    width: 100px !important; 
    min-width: 100px !important;
    max-width: 100px !important;
    padding: 0 !important;
    margin: 2px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* 棋盤列容器，確保水平置中且不換行 */
.row-container {
    display: flex !important;
    flex-direction: row !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 0 !important;
}

/* 棋盤整體容器，限制寬度以防變形 */
.board-container { 
    width: 320px !important; 
    margin: 0 auto !important; 
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    gap: 0 !important;
}

/* 修復 Gradio 預設的間距 */
.form { gap: 0 !important; }
"""


def create_tic_tac_toe_ui(agent_type_name: str) -> gr.Blocks:
    """封裝成函數，方便未來可以掛載到不同的 FastAPI endpoint 或 Tabs"""
    with gr.Blocks(css=css) as demo:
        gr.Markdown(
            f"# 🎮 井字遊戲 (Tic-Tac-Toe)\n\n### 目前對手：**{agent_type_name}**"
        )

        agent_state = gr.State(agent_type_name)
        engine_state = gr.State(TicTacToeEngine)
        human_player_state = gr.State(1)

        with gr.Row(elem_classes="row-container"):
            side_selector = gr.Radio(
                choices=["先手 (X)", "後手 (O)"],
                value="先手 (X)",
                label="選擇先後手",
            )

        status_text = gr.Textbox(
            label="遊戲狀態", value="選好先後手後按「開始」。", interactive=False
        )

        buttons = []
        with gr.Column(elem_classes="board-container"):
            for r in range(3):
                with gr.Row(elem_classes="row-container"):
                    for c in range(3):
                        btn = gr.Button(
                            " ", elem_classes="square-btn", variant="secondary"
                        )
                        buttons.append(btn)

        with gr.Row(elem_classes="row-container"):
            new_game_btn = gr.Button("🔄 開始 / 重新開始", variant="primary", scale=0)

        for i, btn in enumerate(buttons):
            btn.click(
                fn=make_handler(i),
                inputs=[engine_state, agent_state, human_player_state],
                outputs=[engine_state, human_player_state, status_text] + buttons,
            )

        new_game_btn.click(
            fn=reset_game,
            inputs=[agent_state, side_selector],
            outputs=[engine_state, human_player_state, status_text] + buttons,
        )
    return demo


# 啟動區域
if __name__ == "__main__":
    with gr.Blocks(title="Tic-Tac-Toe RL 訓練場", css=css) as combined_demo:
        gr.Markdown("# 🏆 Board Game RL 訓練場")
        with gr.Tab("🆚 Alpha-Beta (完美大師)"):
            create_tic_tac_toe_ui("Alpha-Beta Pruning (完美大師)")
        with gr.Tab("🆚 DQN (深度Q)"):
            create_tic_tac_toe_ui("DQN (深度Q網路)")
        with gr.Tab("🆚 Q-Learning (小Q)"):
            create_tic_tac_toe_ui("Q-Learning (從零學習的小Q)")
        with gr.Tab("🆚 Random (小白)"):
            create_tic_tac_toe_ui("Random Agent (亂走的小白)")

    combined_demo.launch(server_name="0.0.0.0", server_port=7860)
