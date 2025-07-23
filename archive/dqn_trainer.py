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
    train_frequency = 4  # Frequency of training updates
    save_model_frequency = 10000 # Save model every 100 episodes

    model = None
    loss_fn = nn.MSELoss()
    loss_list = []
    optimizer = None

    num_actions = None
    total_steps = 0
    optimization_steps = 0

    def train(self, episodes, game_name='BreakoutNoFrameskip-v4', render=False, disable_frame_skips=False):
        # Environment setup
        env = gym.make(game_name, render_mode='human' if render else None)
        env = AtariPreprocessing(env, grayscale_obs=True, frame_skip=1 if disable_frame_skips else 4)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = FrameStackObservation(env, stack_size=4)
        self.num_actions = env.action_space.n
        state, _ = env.reset()
        game_name = game_name.replace('/', '_')  # Sanitize game name for file paths

        epsilon = 1
        memory = ReplayMemory(self.replay_memory_size)
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []

        policy_dqn = DQN(self.num_actions).to(device) if self.model is None else self.model
        target_dqn = DQN(self.num_actions).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        step_count = 0
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        for i in range(episodes):
            print(f"Episode {i+1}/{episodes} started.")
            state = env.reset()[0]
            terminated, truncated = False, False
            rewards_per_episode[i] = 0

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

                if len(memory) > self.mini_batch_size and self.total_steps % self.train_frequency == 0:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    if self.optimization_steps > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

            epsilon = max(epsilon - 1 / episodes, 0)
            epsilon_history.append(epsilon)

            if i % self.save_model_frequency == 0:
                self.save_model(policy_dqn, f"{weights_path}/{game_name}_episode_{i}.pt")

            # print total steps so far
            print(f"Total steps so far: {self.total_steps}")

        env.close()
        fileName = f"{weights_path}/{game_name}.pt"
        torch.save(policy_dqn.state_dict(), fileName)

        self.plot_results(rewards_per_episode, epsilon_history, game_name)

        print()
        print("=" * 50)
        print(f"Training completed. Total steps: {self.total_steps}, Optimization steps: {self.optimization_steps}")

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
        self.loss_list.append(loss.item()) 
        self.optimization_steps += 1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, model, file_name):
        torch.save(model.state_dict(), file_name)
        print(f"Model saved to {file_name}")

    def load_model(self, file_name):
        self.model.load_state_dict(torch.load(file_name, map_location=device))
        self.model.eval()

    def plot_results(self, rewards_per_episode, epsilon_history, game_name):
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
        plotName = f"{plots_path}/{game_name}_plot.png"
        plt.savefig(plotName)

        # plot enumarted loss_list
        plt.figure(2)
        plt.plot(self.loss_list)
        plt.title("Loss per Optimization Step")
        plt.xlabel("Optimization Step")
        plt.ylabel("Loss")
        plt.savefig(f"{plots_path}/{game_name}_loss_plot.png")

    def show_state_images(self, state):
        for i in range(4):  # Ensure we have 4 stacked frames
            image = Image.fromarray(state[i])
            image.show(title=f"Initial State - Episode {i+1}")
        return

if __name__ == "__main__":
    start_time = time.time()
    trainer = DQNtrainer()
    trainer.train(episodes=100000, game_name='ALE/Breakout-v5', render=False, disable_frame_skips=True)
    # trainer.train(episodes=500, game_name='MsPacmanNoFrameskip-v4', render=False)
    # trainer.train(episodes=500, render=False)
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
