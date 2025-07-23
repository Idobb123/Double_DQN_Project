import torch
import numpy as np
from DQN import DQN, state_to_dqn_input
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import gymnasium as gym
import ale_py
import torch.nn as nn

gym.register_envs(ale_py)

class DQNAgent:
    def __init__(self, game_name, weights_path):
        # use dummy env to get the number of actions
        env = gym.make(game_name, render_mode=None)
        num_actions = env.action_space.n
        self.model = DQN(num_actions)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()

    def select_action(self, state):
        with torch.no_grad():
            temp_state = state_to_dqn_input(state)
            action_values = self.model(temp_state)
            # print(f"Action values: {action_values}")
            action = action_values.argmax().item()
            return action

    def play(self, game_name , episodes=3, disable_frame_skips=False):
        env = gym.make(game_name, render_mode='human')
        env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=1 if disable_frame_skips else 4)
        env = FrameStackObservation(env, stack_size=4)

        for ep in range(episodes):
            state = env.reset()[0]
            terminated, truncated = False, False
            total_reward = 0
            
            step_count = 0
            action = self.select_action(state)
            while (not terminated and not truncated):
                if step_count < 100:
                    action = env.action_space.sample()  # Random action for the first 100 steps
                else:  # Every 4 steps, select action
                    action = self.select_action(state)
                state, reward, terminated, truncated, _  = env.step(action)
                total_reward += reward
                step_count += 1
                # print(f"Action distribution: {action_count}")
            print(f"Episode {ep+1}: Total Reward = {total_reward}")


        env.close()

if __name__ == "__main__":
    game_name = 'MsPacmanNoFrameskip-v4'
    # game_name = 'ALE/Breakout-v5'
    sanitized_game_name = game_name.replace('/', '_')  # Sanitize game name for file paths

    # weights_path = f'output/weights/{sanitized_game_name}.pt'  # Adjust the path as needed
    weights_path = f'output/weights/{sanitized_game_name}_episode_400.pt'  # Adjust the path as needed

    agent = DQNAgent(game_name, weights_path)
    agent.play(game_name, episodes=3, disable_frame_skips=True)