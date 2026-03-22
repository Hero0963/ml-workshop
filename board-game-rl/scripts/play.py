"""
Play Tic-Tac-Toe in terminal against a Random Agent.
"""

import sys
import time
from pathlib import Path

# Add src to python path for relative imports to work
sys.path.append(str(Path(__file__).parent.parent / "src"))

from board_game_rl.games.tic_tac_toe.env import TicTacToeEnv
from board_game_rl.agents.random_agent import RandomAgent


def play_game():
    env = TicTacToeEnv(render_mode="human")
    agent = RandomAgent()

    observation, info = env.reset()
    env.render()

    terminated = False

    while not terminated:
        current_player = info["current_player"]
        legal_actions = info["legal_actions"]

        if current_player == 1:
            # Human turn (X)
            print(f"Your turn. Legal moves: {legal_actions}")
            while True:
                try:
                    action_str = input("Enter cell number (0-8): ")
                    action = int(action_str)
                    if action in legal_actions:
                        break
                    print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a number.")
        else:
            # Agent turn (O)
            print("Agent is thinking...")
            time.sleep(0.5)
            action = agent.act(observation, info)
            print(f"Agent played: {action}")

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        if terminated:
            engine = env.unwrapped.engine
            winner = engine.winner
            print("Game Over!")
            if winner == 1:
                print("Congratulations, you (X) win!")
            elif winner == -1:
                print("Agent (O) wins!")
            else:
                print("It's a draw!")
            break


if __name__ == "__main__":
    print("Welcome to Tic-Tac-Toe RL Interface!")
    print("Cells are numbered 0-8 from top-left to bottom-right:")
    print(" 0 | 1 | 2 ")
    print("-----------")
    print(" 3 | 4 | 5 ")
    print("-----------")
    print(" 6 | 7 | 8 ")
    print("=" * 30)

    while True:
        play_game()
        play_again = input("\nPlay again? (y/n): ")
        if play_again.lower() != "y":
            break
