import torch
import numpy as np
from DQN import DQN, state_to_dqn_input
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import gymnasium as gym
import ale_py

import torch.nn as nn

class DQNAgent:
    def __init__(self, num_actions, weights_path):
        self.model = DQN(num_actions)
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        self.model.eval()

    def select_action(self, state):
        with torch.no_grad():
            temp_state = state_to_dqn_input(state)
            action = self.model(temp_state).argmax().item()
            return action

    def play(self, gameName , episodes=3):
        gym.register_envs(ale_py)
        env = gym.make(gameName, render_mode='human')
        env = AtariPreprocessing(env, grayscale_obs=True)
        env = FrameStackObservation(env, stack_size=4)

        for ep in range(episodes):
            state = env.reset()[0]
            terminated, truncated = False, False
            total_reward = 0
            while (not terminated and not truncated):
                action = self.select_action(state)
                state, reward, terminated, truncated, _  = env.step(action)
                total_reward += reward
            print(f"Episode {ep+1}: Total Reward = {total_reward}")

        env.close()

if __name__ == "__main__":
    gameName = 'MsPacmanNoFrameskip-v4'
    weights_path = f'output/weights/{gameName}.pt'  # Adjust the path as needed
    num_actions = 9  # Example number of valid actions

    agent = DQNAgent(num_actions, weights_path)
    agent.play(gameName, episodes=10)  # Play with the trained agent