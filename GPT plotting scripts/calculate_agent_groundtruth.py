import os
import torch
import numpy as np
from regularAgent import RegularAgent

def load_pt_files(directory='./output'):
    pt_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    dqn_models = [os.path.join(directory, f) for f in pt_files if '_dqn_' in f]
    ddqn_models = [os.path.join(directory, f) for f in pt_files if '_ddqn_' in f]
    return dqn_models, ddqn_models

def get_agents_groundtruth(config="lunarlander4-windy", episodes=100, max_steps=1000, gamma=0.99):
    dqn_models, ddqn_models = load_pt_files()
    # print total models found
    print(f"Total models found: {len(dqn_models) + len(ddqn_models)}")
    render = False
    agent = RegularAgent(config)


    dqn_discounted_rewards = []
    ddqn_discounted_rewards = []
    for model_path in dqn_models:
        mean_discounted_reward = agent.run(weights_path=model_path, episodes=episodes, max_steps=max_steps, render=render, gamma=gamma)
        dqn_discounted_rewards.append(mean_discounted_reward)
    for model_path in ddqn_models:
        mean_discounted_reward = agent.run(weights_path=model_path, episodes=episodes, max_steps=max_steps, render=render, gamma=gamma)
        ddqn_discounted_rewards.append(mean_discounted_reward)

    return np.mean(dqn_discounted_rewards), np.mean(ddqn_discounted_rewards)

if __name__ == "__main__":
    get_agents_groundtruth()
