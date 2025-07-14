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

# CUDA device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class DQLAgent():
    # Hyperparameters (adjustable)
    learning_rate_a = 0.00025       # learning rate (alpha)
    discount_factor_g = 0.99        # discount rate (gamma)    
    network_sync_rate = 1000         # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 100000      # size of replay memory
    mini_batch_size = 32            # size of the training data set sampled from the replay memory

    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.

    ACTIONS = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']
    
    # Train the FrozeLake environment
    def train(self, episodes, is_slippery=False):
        # env = gym.make('ALE/Pacman-v5', render_mode=None)
        env = gym.make('BreakoutNoFrameskip-v4', render_mode=None)
        # print(f"Number of actions: {env.action_space}")
        env = AtariPreprocessing(env, grayscale_obs=True)
        env = FrameStackObservation(env, stack_size=4)

        # Print the action space
        num_actions = env.action_space.n
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network
        policy_dqn = DQN(num_actions).to(device)
        target_dqn = DQN(num_actions).to(device)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Reset and take one random step
        state, _ = env.reset()
        state, reward, done, truncated, info = env.step(env.action_space.sample())

        # Initialize rewards per episode
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0

        for i in range(episodes):
            print(f"Episode {i+1}/{episodes} started.")
            
            state = env.reset()[0]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            truncated = False       # True when agent takes more than 200 actions    

            while(not terminated and not truncated):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action            
                    with torch.no_grad():
                        temp_state = state_to_dqn_input(state).to(device)
                        action = policy_dqn(temp_state).argmax().item()

                # Execute action
                new_state,reward,terminated,truncated,_ = env.step(action)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated)) 

                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1

            # Keep track of the rewards collected per episode.
            if reward == 1:
                rewards_per_episode[i] = 1

            # Check if enough experience has been collected and if at least 1 reward has been collected
            # if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
            if len(memory)>self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)        
                # print(f"Episode {i+1}/{episodes} completed. Total reward: {np.sum(rewards_per_episode)}")

                # Decay epsilon
                epsilon = max(epsilon - 1/episodes, 0)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count=0

        # Close environment
        env.close()

        # Save policy
        fileName = "breakout_dqn"

        torch.save(policy_dqn.state_dict(), f"{fileName}.pt")

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
        plt.savefig(f'{fileName}.png')

    def optimize(self, mini_batch, policy_dqn, target_dqn):

        current_q_list = []
        target_q_list = []

        for state, action, new_state, reward, terminated in mini_batch:
            

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.tensor([reward], dtype=torch.float32, device=device)
            else:
                # Calculate target q value 
                with torch.no_grad():
                    new_state = state_to_dqn_input(new_state).to(device)
                    target_max_value = target_dqn(new_state).max()  # Get the maximum Q value for the next state

                    # torch.tensor(data, dtype=*, device='cuda')
                    target = torch.tensor(
                        reward + self.discount_factor_g * target_max_value,
                        dtype=torch.float32,
                        device=device
                    )

            # Get the current set of Q values
            current_q = policy_dqn(state_to_dqn_input(state).to(device))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(state_to_dqn_input(state).to(device))
            
            # Adjust the specific action to the target that was just calculated
            target_q[0][action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == "__main__":
    pacman = DQLAgent()
    pacman.train(episodes=10000)  # Adjust the number of episodes as needed
