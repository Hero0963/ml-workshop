"""
Gradio Frontend for Board Game RL API.
"""

import gradio as gr

from board_game_rl.api.inference import get_optimal_move
from board_game_rl.core.engine import TicTacToeEngine


def get_board_values(engine: TicTacToeEngine) -> list[str]:
    chars = {1: "X", -1: "O", 0: " "}
    return [chars[engine.board[i // 3][i % 3]] for i in range(9)]


def get_winner_msg(engine: TicTacToeEngine) -> str:
    if engine.winner == 1:
        return "You (X) Win!"
    if engine.winner == -1:
        return "Agent (O) Wins!"
    if engine.winner == 0:
        return "It's a Draw!"
    return ""


def parse_action(engine: TicTacToeEngine, cell_idx: int) -> list:
    if engine.winner is not None:
        return [engine, "Game over! Click New Game."] + get_board_values(engine)

    # Human turn (X, which is 1)
    r, c = cell_idx // 3, cell_idx % 3
    valid = engine.step((r, c))
    if not valid:
        return [engine, "Invalid move!"] + get_board_values(engine)

    if engine.winner is not None:
        return [engine, get_winner_msg(engine)] + get_board_values(engine)

    # Agent turn (O, which is -1)
    action = get_optimal_move(engine.board, engine.current_player)
    if action != -1:
        engine.step((action // 3, action % 3))

    msg = "Your turn (X)."
    if engine.winner is not None:
        msg = get_winner_msg(engine)

    return [engine, msg] + get_board_values(engine)


def reset_game() -> list:
    engine = TicTacToeEngine()
    return [engine, "New Game Started. Your turn (X)."] + [" "] * 9


# Factory for event handlers to capture the loop variable
def make_handler(idx: int):
    def handler(engine: TicTacToeEngine):
        return parse_action(engine, idx)

    return handler


css = """
.square-btn {
    font-size: 3em !important;
    height: 100px !important;
    width: 100px !important;
    min-width: 100px !important;
}
.board-container {
    max-width: 350px !important;
    margin: 0 auto !important;
    display: flex !important;
    flex-wrap: wrap !important;
    justify-content: center !important;
    gap: 10px !important;
}
"""

with gr.Blocks(title="Tic-Tac-Toe RL", css=css) as demo:
    gr.Markdown("# 🎮 Play Tic-Tac-Toe against the RL Agent!")

    engine_state = gr.State(TicTacToeEngine)
    status_text = gr.Textbox(
        label="Game Status", value="Game Started. Your turn (X).", interactive=False
    )

    buttons = []
    # Create 3x3 grid of buttons using CSS Flexbox container
    with gr.Column(elem_classes="board-container"):
        for r in range(3):
            with gr.Row():
                for c in range(3):
                    idx = r * 3 + c
                    btn = gr.Button(" ", elem_classes="square-btn", variant="secondary")
                    buttons.append(btn)

    with gr.Row():
        new_game_btn = gr.Button("New Game", variant="primary")

    # Wire up the events
    for i, btn in enumerate(buttons):
        btn.click(
            fn=make_handler(i),
            inputs=[engine_state],
            outputs=[engine_state, status_text] + buttons,
        )

    new_game_btn.click(
        fn=reset_game, inputs=[], outputs=[engine_state, status_text] + buttons
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
