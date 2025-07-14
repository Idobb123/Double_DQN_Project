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

gym.register_envs(ale_py)
output_dir = 'output'  # Directory to save the model and plots
weights_path = f"{output_dir}/weights"
plots_path = f"{output_dir}/plots"

class DQNtrainer():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.00025       # learning rate (alpha)
    discount_factor_g = 0.99        # discount rate (gamma)    
    network_sync_rate = 100         # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 10000      # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    # Actions for the Atari environment
    num_actions = None  # This will be set during training
    
    def train(self, episodes, gameName='BreakoutNoFrameskip-v4', render=False):
        # environment setup
        env = gym.make(gameName, render_mode='human' if render else None)
        env = AtariPreprocessing(env, grayscale_obs=True)
        env = FrameStackObservation(env, stack_size=4)
        self.num_actions = env.action_space.n
        state, _ = env.reset()
        state, reward, done, truncated, info = env.step(env.action_space.sample())

        # misc setup
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []

        # network setup
        policy_dqn = DQN(self.num_actions)
        target_dqn = DQN(self.num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        step_count=0
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        for i in range(episodes):
            print(f"Episode {i+1}/{episodes} started.")
            
            # init environment
            state = env.reset()[0]
            terminated, truncated = False, False
            rewards_per_episode[i] = 0

            while(not terminated and not truncated):
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        temp_state = state_to_dqn_input(state)
                        action = policy_dqn(temp_state).argmax().item()

                # Take action in the environment
                new_state,reward,terminated,truncated,_ = env.step(action)
                memory.append((state, action, new_state, reward, terminated)) 
                state = new_state
                rewards_per_episode[i] += reward
                step_count+=1

                # Check if enough experience has been collected
                if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode[:i+1]) > 0:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count=0

            # Decay epsilon
            epsilon = max(epsilon - 1/episodes, 0)
            epsilon_history.append(epsilon)

        # Close environment
        env.close()

        # Save policy
        fileName = f"{weights_path}/{gameName}.pt"
        torch.save(policy_dqn.state_dict(), fileName)

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plotName = f"{plots_path}/{gameName}_plot.png"
        plt.savefig(plotName)

        return fileName

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        poilcy_q_values = []
        target_q_values = []

        for state, action, new_state, reward, terminated in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                y = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    new_state = state_to_dqn_input(new_state)
                    target_max_value = target_dqn(new_state).max()  # Get the maximum Q value for the next state

                    y = torch.FloatTensor(
                        reward + self.discount_factor_g * target_max_value
                    )

            # Get the current set of Q values
            current_q_value = policy_dqn(state_to_dqn_input(state))
            poilcy_q_values.append(current_q_value)

            # Get the target set of Q values
            target_q = target_dqn(state_to_dqn_input(state))
            
            # Adjust the specific action to the target that was just calculated
            target_q[0][action] = y
            target_q_values.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(poilcy_q_values), torch.stack(target_q_values))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    pacman = DQNtrainer()
    pacman.train(episodes=10, render=False)  # Adjust the number of episodes as needed