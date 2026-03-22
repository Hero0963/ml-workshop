"""
Train Q-Learning Agent against Random Agent.
"""

import sys
from pathlib import Path
from tqdm import tqdm

# Add src to python path for relative imports to work
sys.path.append(str(Path(__file__).parent.parent / "src"))

from board_game_rl.games.tic_tac_toe.env import TicTacToeEnv
from board_game_rl.agents.q_learning_agent import QLearningAgent
from board_game_rl.agents.random_agent import RandomAgent


def train(episodes=20000):
    env = TicTacToeEnv(render_mode=None)
    # 開始訓練時，給比較高的探索率 (20% 機率亂下)
    q_agent = QLearningAgent(epsilon=0.2)
    random_agent = RandomAgent()

    wins, losses, draws = 0, 0, 0

    print(f"Starting training for {episodes} episodes...")

    for episode in tqdm(range(episodes)):
        observation, info = env.reset()
        terminated = False

        # 紀錄 Q-Agent 上一步的狀態與動作
        last_obs = None
        last_action = None

        while not terminated:
            current_player = info["current_player"]

            if current_player == 1:
                # Q-Agent 的回合 (X)

                # 如果這不是第一步，代表對手剛下完，我們現在來幫「上一步」打分數
                if last_obs is not None:
                    q_agent.learn(
                        old_observation=last_obs,
                        action=last_action,
                        reward=0.0,  # 遊戲還沒結束，當下獎勵是 0
                        next_observation=observation,
                        next_info=info,
                        done=False,
                    )

                action = q_agent.act(observation, info, is_training=True)
                last_obs = observation.copy()
                last_action = action

                observation, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    # Q-Agent 下完這步後遊戲結束
                    engine = env.unwrapped.engine
                    if engine.winner == 1:
                        final_reward = 1.0
                        wins += 1
                    elif engine.winner == -1:
                        final_reward = -1.0
                        losses += 1
                    else:
                        final_reward = 0.5  # 給平手一點小獎勵，鼓勵不敗
                        draws += 1

                    q_agent.learn(
                        old_observation=last_obs,
                        action=last_action,
                        reward=final_reward,
                        next_observation=observation,  # 這裡的 observation 已經不重要了
                        next_info=info,
                        done=True,
                    )
            else:
                # Random Agent 的回合 (O)
                action = random_agent.act(observation, info)
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    # 對手下完這步後遊戲結束 (代表 Q-Agent 輸了或平手)
                    engine = env.unwrapped.engine
                    if engine.winner == -1:
                        final_reward = -1.0
                        losses += 1
                    else:
                        final_reward = 0.5
                        draws += 1

                    q_agent.learn(
                        old_observation=last_obs,
                        action=last_action,
                        reward=final_reward,
                        next_observation=observation,
                        next_info=info,
                        done=True,
                    )

        # 隨著訓練變多，AI 越來越聰明，降低亂下的機率 (Exploitation 比例增加)
        if episode > 0 and episode % 1000 == 0:
            q_agent.epsilon = max(0.01, q_agent.epsilon * 0.95)

    print("\nTraining Complete!")
    print(
        f"Results over {episodes} episodes: Wins: {wins}, Losses: {losses}, Draws: {draws}"
    )

    # 儲存訓練成果 (Q-Table)
    model_path = Path(__file__).parent.parent / "models" / "q_table.json"
    q_agent.save_model(str(model_path))


if __name__ == "__main__":
    # 需要先安裝 tqdm: uv pip install tqdm
    train(20000)
