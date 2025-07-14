import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
import ale_py
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from PIL import Image
import random

from ReplayMemory import ReplayMemory
from DQN import DQN, state_to_dqn_input
import time

gym.register_envs(ale_py)
output_dir = 'output'
weights_path = f"{output_dir}/weights"
plots_path = f"{output_dir}/plots"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQNtrainer():
    learning_rate_a = 0.00025
    discount_factor_g = 0.99
    network_sync_rate = 100
    replay_memory_size = 10000
    mini_batch_size = 32
    save_model_frequency = 100 # Save model every 100 episodes

    loss_fn = nn.MSELoss()
    optimizer = None
    num_actions = None
    total_steps = 0

    def train(self, episodes, gameName='BreakoutNoFrameskip-v4', render=False):
        # Environment setup
        env = gym.make(gameName, render_mode='human' if render else None)
        env = AtariPreprocessing(env, grayscale_obs=True)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = FrameStackObservation(env, stack_size=4)
        self.num_actions = env.action_space.n
        state, _ = env.reset()

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []

        policy_dqn = DQN(self.num_actions).to(device)
        target_dqn = DQN(self.num_actions).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        step_count = 0
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        for i in range(episodes):
            print(f"Episode {i+1}/{episodes} started.")
            state = env.reset()[0]
            terminated, truncated = False, False
            rewards_per_episode[i] = 0

            print(f"Initial state shape: {state.shape}")
            for i in range(4):  # Ensure we have 4 stacked frames
                state, _, terminated, truncated, _ = env.step(env.action_space.sample())
                image = Image.fromarray(state[i])
                image.show(title=f"Initial State - Episode {i+1}")
            return

            while not terminated and not truncated:
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        temp_state = state_to_dqn_input(state).to(device)
                        action = policy_dqn(temp_state).argmax().item()

                new_state, reward, terminated, truncated, _ = env.step(action)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                rewards_per_episode[i] += reward
                step_count += 1
                self.total_steps += 1

                if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode[:i+1]) > 0:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

            epsilon = max(epsilon - 1 / episodes, 0)
            epsilon_history.append(epsilon)

            if i % self.save_model_frequency == 0:
                self.save_model(policy_dqn, f"{weights_path}/{gameName}_episode_{i}.pt")
            
            print(f"Episode {i+1} steps: {env.info["l"]}")

        env.close()
        fileName = f"{weights_path}/{gameName}.pt"
        torch.save(policy_dqn.state_dict(), fileName)

        # plt.figure(1)
        # sum_rewards = np.zeros(episodes)
        # for x in range(episodes):
        #     sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        # plt.subplot(121)
        # plt.plot(sum_rewards)
        # plt.subplot(122)
        # plt.plot(epsilon_history)
        # plt.savefig(f"{plots_path}/{gameName}_plot.png")

        plt.figure(1)

        # Plot raw reward per episode
        plt.subplot(121)
        plt.plot(rewards_per_episode)
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Plot epsilon decay
        plt.subplot(122)
        plt.plot(epsilon_history)
        plt.title("Epsilon Decay")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")

        # Save the figure
        plotName = f"{plots_path}/{gameName}_plot.png"
        plt.savefig(plotName)

        return fileName

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        policy_q_values = []
        target_q_values = []

        for state, action, new_state, reward, terminated in mini_batch:
            state_tensor = state_to_dqn_input(state).to(device)
            new_state_tensor = state_to_dqn_input(new_state).to(device)

            if terminated:
                y = torch.tensor([reward], dtype=torch.float32, device=device)
            else:
                with torch.no_grad():
                    target_max_value = target_dqn(new_state_tensor).max()
                    y = torch.tensor([reward + self.discount_factor_g * target_max_value], device=device)

            current_q = policy_dqn(state_tensor)
            target_q = current_q.clone().detach()
            target_q[0][action] = y

            policy_q_values.append(current_q)
            target_q_values.append(target_q)

        loss = self.loss_fn(torch.stack(policy_q_values), torch.stack(target_q_values))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, model, file_name):
        torch.save(model.state_dict(), file_name)
        print(f"Model saved to {file_name}")

if __name__ == "__main__":
    start_time = time.time()
    pacman = DQNtrainer()
    # pacman.train(episodes=500, gameName='MsPacmanNoFrameskip-v4', render=False)
    pacman.train(episodes=500, render=False)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
