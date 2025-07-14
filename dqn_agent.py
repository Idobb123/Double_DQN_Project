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
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return int(torch.argmax(q_values, dim=1).item())

    def play(self, gameName , episodes=3):
        gym.register_envs(ale_py)
        env = gym.make(gameName, render_mode='human')
        env = AtariPreprocessing(env, grayscale_obs=True)
        env = FrameStackObservation(env, stack_size=4)

        for ep in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            while not done:
                state_tensor = state_to_dqn_input(state)
                action = self.select_action(state_tensor)
                state, reward, done, _ = env.step(action)
                total_reward += reward
            print(f"Episode {ep+1}: Total Reward = {total_reward}")

        env.close()

# Example usage:
# import gym
# env = gym.make('CartPole-v1')
# agent = DQNAgent(state_dim=4, action_dim=2, weights_path='dqn_weights.pt')
# agent.play(env, episodes=5, render=True)