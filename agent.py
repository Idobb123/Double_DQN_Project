from abc import abstractmethod
import torch
import gymnasium
from dqn import DQN
from replaymemory import ReplayMemory
import yaml
import random
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Set the backend
import matplotlib.pyplot as plt
import time
import flappy_bird_gymnasium

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for compatibility

class Agent():
    suffix: str = ""

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.exploration_fraction = hyperparameters['exploration_fraction']  # fraction of total steps for exploration
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.episodes           = hyperparameters['episodes']               # number of episodes to train the agent
        self.train_freq         = hyperparameters['train_freq']               # how often to train the agent
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # how often to sync the target network with the policy network
        self.discount_factor    = hyperparameters['discount_factor']   
        self.learning_rate      = hyperparameters['learning_rate']          # learning rate for the optimizer
        self.first_layer_dim    = hyperparameters['first_layer_dim']       # size of the first hidden layer in the DQN
        self.second_layer_dim   = hyperparameters['second_layer_dim']      # size of the second hidden layer in the DQN
        self.max_reward         = hyperparameters['max_reward']                     # maximum reward for the environment
        self.eval_freq          = hyperparameters['eval_freq']

        self.loss_fn = torch.nn.MSELoss()    # loss function for training

    @abstractmethod
    def update_network(self, mini_batch, policy_dqn, target_dqn):
        """
        Perform a single optimization step using the given minibatch, policy network,
        and target network. Must be implemented by subclasses.
        """
        pass

    def train(self, render=False, total_steps=100_000):
        env = self.create_env(self.env_id, "human" if render else None)

        num_features = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_features, num_actions, self.first_layer_dim, self.second_layer_dim).to(device)
        target_dqn = DQN(num_features, num_actions, self.first_layer_dim, self.second_layer_dim).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        memory = ReplayMemory(self.replay_memory_size)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)  # optimizer for training

        rewards_per_episode = []
        epsilon_history = []
        step_rewards = []  # Track reward at each step for plotting
        epsilon = self.epsilon_init
        best_reward = float('-inf')

        episode_reward = 0
        episode_length = 0
        epsilon_slope = (self.epsilon_init - self.epsilon_min) / (total_steps * self.exploration_fraction)

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)  # Add batch
        for step in range(total_steps):
            # choose action
            if random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64).to(device)
            else:
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

            # take action
            new_state, reward, terminated, truncated, _ = env.step(action.item())
            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
            reward = torch.tensor(reward, dtype=torch.float32).to(device)

            # store transition in replay memory
            memory.append((state, action, new_state, reward, terminated))
            # update state, reward, epsilon
            episode_reward += reward
            state = new_state
            epsilon = max(self.epsilon_min, epsilon - epsilon_slope)
            epsilon_history.append(epsilon)
            step_rewards.append(episode_reward.item())  # Track cumulative episode reward at each step

            # optimization step
            if step % self.train_freq == 0 and len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.update_network(mini_batch, policy_dqn, target_dqn)

            # check if episode is done
            if terminated or truncated:
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32).to(device)
                rewards_per_episode.append(episode_reward)
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    print(f"{current_time} : NEW BEST : {best_reward}")
                    self.save_weights(policy_dqn)
                    # self.save_gif(policy_dqn, step)
                episode_reward = 0

            # print progress
            if step % self.eval_freq == 0:
                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"{current_time} : Step {step}, MeanReward: {np.mean(rewards_per_episode[-100:]):.4f}, Epsilon: {epsilon:.4f}")

            # sync networks
            if step % self.network_sync_rate == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())

            episode_length += 1

        env.close()
        self.save_graph(step_rewards, epsilon_history)
        return rewards_per_episode, epsilon_history

    def run(self, weights_path, episodes=1,max_steps=2000, render=True):
        # Create the environment
        env = self.create_env(self.env_id, render_mode="human" if render else None)

        # Load the trained model
        num_features = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []

        self.policy_dqn = DQN(num_features, num_actions, self.first_layer_dim, self.second_layer_dim).to(device)
        self.policy_dqn.load_state_dict(torch.load(weights_path, map_location=device))

        for episode in range(episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            episode_reward = 0
            reached_max_steps = True
            for step in range(max_steps):
                with torch.no_grad():
                    action = self.policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()
                new_state, reward, terminated, _, info = env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                state = new_state
                episode_reward += reward
                if terminated:
                    reached_max_steps = False
                    rewards_per_episode.append(episode_reward)
                    break

            if reached_max_steps:
                print("REACHED MAX STEPS")
            else:
                print("episode terminated early")

            print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}")

        print(f"Average Reward over {episodes} episodes: {np.mean(rewards_per_episode):.2f} +- {np.std(rewards_per_episode):.2f}")
        env.close()

    def create_env(self, env_id, render_mode=None):
        if env_id == "FlappyBird-v0":
            env = gymnasium.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
            return env
        env = gymnasium.make(env_id, render_mode=render_mode)
        return env

    def save_graph(self, step_rewards, epsilon_history):
        # Create dual-axis plot similar to visualize.py
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Calculate rolling mean rewards over steps (not episodes)
        window_size = 1000  # Rolling window over steps
        mean_step_rewards = []
        for i in range(len(step_rewards)):
            start_idx = max(0, i - window_size + 1)
            mean_step_rewards.append(np.mean(step_rewards[start_idx:i+1]))
        
        # Plot mean reward on primary y-axis (blue) vs steps
        color1 = 'tab:blue'
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Mean Episode Reward', color=color1)
        line1 = ax1.plot(range(len(mean_step_rewards)), mean_step_rewards, color=color1, linewidth=2, label='Mean Reward')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)
        
        # Plot epsilon decay on secondary y-axis (red) vs steps
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('Exploration Rate (Epsilon)', color=color2)
        line2 = ax2.plot(range(len(epsilon_history)), epsilon_history, color=color2, linewidth=2, label='Epsilon')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
        
        plt.title(f"Training Progress - {self.env_id}", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Show statistics (similar to visualize.py)
        print(f"\nTraining Statistics:")
        print(f"Total training steps: {len(step_rewards):,}")
        print(f"Initial mean reward: {mean_step_rewards[0]:.2f}")
        print(f"Final mean reward: {mean_step_rewards[-1]:.2f}")
        print(f"Best mean reward: {max(mean_step_rewards):.2f}")
        print(f"Initial epsilon: {epsilon_history[0]:.4f}")
        print(f"Final epsilon: {epsilon_history[-1]:.4f}")

        # Save plots
        fig.savefig(f"output/{self.env_id+self.suffix}_training_progress.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    def save_weights(self, policy_dqn):
        sanitized_env_id = self.env_id.replace('/', '_')
        torch.save(policy_dqn.state_dict(), f'output/{sanitized_env_id+self.suffix}.pt')

    def save_gif(self, policy_dqn, steps):
        env = gymnasium.make(self.env_id, render_mode="rgb_array")
        frames = []
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)

        for _ in range(500):
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()
            new_state, reward, terminated, _, info = env.step(action)
            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
            frames.append(env.render())
            state = new_state

            if terminated:
                break

        env.close()

        # Save frames as a GIF
        from imageio import mimsave
        mimsave(f'output/{self.env_id+self.suffix}_{steps}.gif', frames, fps=30)