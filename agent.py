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
from torch.utils.tensorboard import SummaryWriter
import json
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for compatibility

class Agent():
    """
    Base class for reinforcement learning agents.
    Contains common methods and properties for training and evaluating agents.
    """
    suffix: str = ""

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set
        self.hyperparameters = hyperparameters  # Store for later saving

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

    def train(self, render=False, total_steps=100_000, index=1):
        """
        Train the agent using the specified environment and hyperparameters.
        Args:
            render (bool): Whether to render the environment during training.
            total_steps (int): Total number of steps to train the agent.
            index (int): Index for saving logs and models.
        """
        env = self.create_env(self.env_id, "human" if render else None)

        # Create TensorBoard writer with informative indexed directory
        sanitized_env_id = self.env_id.replace('/', '_')
        algorithm_name = self.__class__.__name__.lower()
        log_dir = f"logs/{sanitized_env_id}_{algorithm_name}_{index:03d}{self.suffix}"
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logs will be saved to: {log_dir}")
        
        # Store index for use in saving methods
        self.current_index = index
        
        # log hyperparameters
        self.save_hyperparameters(log_dir, total_steps)

        # initialize networks and memory
        num_features = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_features, num_actions, self.first_layer_dim, self.second_layer_dim).to(device)
        target_dqn = DQN(num_features, num_actions, self.first_layer_dim, self.second_layer_dim).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        memory = ReplayMemory(self.replay_memory_size)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate) 

        # initialize metrics
        rewards_per_episode = []
        epsilon_history = []
        value_function_history = []
        eval_steps = []  # Track steps when value function was evaluated
        epsilon = self.epsilon_init
        best_reward = float('-inf')

        episode_reward = 0
        episode_length = 0
        episode_count = 0
        epsilon_slope = (self.epsilon_init - self.epsilon_min) / (total_steps * self.exploration_fraction)

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)  # Add batch
        for step in range(total_steps):
            # epsilon-greedy action selection
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

            # optimization step
            if step % self.train_freq == 0 and len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.update_network(mini_batch, policy_dqn, target_dqn)

            # check if episode is done
            if terminated or truncated:
                # Log episode metrics to TensorBoard
                writer.add_scalar('rollout/ep_rew_mean', np.mean(rewards_per_episode[-100:]) if rewards_per_episode else episode_reward, step)
                writer.add_scalar('rollout/ep_len_mean', episode_length, step)
                writer.add_scalar('rollout/exploration_rate', epsilon, step)
                
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32).to(device)
                rewards_per_episode.append(episode_reward)
                episode_count += 1
                if episode_reward > best_reward:
                    best_reward = episode_reward
                    current_time = time.strftime("%H:%M:%S", time.localtime())
                    print(f"{current_time} : NEW BEST : {best_reward}")
                    self.save_weights(policy_dqn)
                    # self.save_gif(policy_dqn, step)
                episode_reward = 0
                episode_length = 0

            # print progress and evaluate value function
            if step % self.eval_freq == 0:
                current_time = time.strftime("%H:%M:%S", time.localtime())
                avg_value = self.estimate_value_function(policy_dqn, eval_steps=1000) # Estimate value function for 1000 steps
                mean_reward = np.mean(rewards_per_episode[-100:]) if rewards_per_episode else 0.0 # Calculate mean reward over last 100 episodes
                value_function_history.append(avg_value) # Store value function data for plotting
                eval_steps.append(step)
                writer.add_scalar('evaluation/avg_value_function', avg_value, step) # Log to TensorBoard
                
                print(f"{current_time} : Step {step}, MeanReward: {mean_reward:.4f}, Epsilon: {epsilon:.4f}, AvgValue: {avg_value:.4f}")

            # sync networks
            if step % self.network_sync_rate == 0:
                target_dqn.load_state_dict(policy_dqn.state_dict())

            episode_length += 1

        # Close TensorBoard writer
        writer.close()
        env.close()
        return rewards_per_episode, epsilon_history

    def run(self, weights_path, episodes=1,max_steps=2000, render=True, debug=False, gamma=0.99, show_rewards=False):
        """
        Run the trained agent in the environment for a specified number of episodes.
        Args:
            weights_path (str): Path to the trained model weights.
            episodes (int): Number of episodes to run.
            max_steps (int): Maximum steps per episode.
            render (bool): Whether to render the environment.
            debug (bool): Whether to print debug information.
            gamma (float): Discount factor for future rewards.
            show_rewards (bool): Whether to print reward statistics.
        """
        # Create the environment
        env = self.create_env(self.env_id, render_mode="human" if render else None)

        # Load the trained model
        num_features = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewards_per_episode = []
        discounted_rewards_per_episode = []

        self.policy_dqn = DQN(num_features, num_actions, self.first_layer_dim, self.second_layer_dim).to(device)
        self.policy_dqn.load_state_dict(torch.load(weights_path, map_location=device))

        # Episode loop
        for episode in range(episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            episode_reward = 0
            discounted_reward = 0
            current_gamma = 1 
            reached_max_steps = True
            for step in range(max_steps):
                with torch.no_grad():
                    action = self.policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()
                new_state, reward, terminated, _, info = env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                state = new_state
                episode_reward += reward
                discounted_reward += reward * current_gamma
                current_gamma *= gamma
                if terminated:
                    reached_max_steps = False
                    break
            
            rewards_per_episode.append(episode_reward)
            discounted_rewards_per_episode.append(discounted_reward)

            # Print debug information
            if reached_max_steps:
                print ("reached max steps")
                if debug:
                    print("REACHED MAX STEPS")
            else:
                if debug:
                    print("episode terminated early")

            # Print episode statistics
            if debug:
                print(f"Episode {episode + 1}/{episodes}, Reward: {episode_reward}")

        # Print reward statistics
        if show_rewards:
            print(f"Average Reward over {episodes} episodes: {np.mean(rewards_per_episode):.2f} +- {np.std(rewards_per_episode):.2f}")
            print(f"Average Discounted Reward over {episodes} episodes: {np.mean(discounted_rewards_per_episode):.2f} +- {np.std(discounted_rewards_per_episode):.2f}")
        
        env.close()
        return np.mean(discounted_rewards_per_episode)

    def create_env(self, env_id, render_mode=None):
        """
        Create a Gym environment.
        Args:
            env_id (str): ID of the environment to create.
            render_mode (str): Render mode for the environment (e.g., "human").
        """
        if env_id == "FlappyBird-v0":
            env = gymnasium.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
            return env
        if env_id == "LunarLander-v2":
            env = gymnasium.make("LunarLander-v2", render_mode=render_mode, enable_wind=True, wind_power=15, turbulence_power=1.5)
            # env = gymnasium.make("LunarLander-v2", render_mode=render_mode)
            return env
        env = gymnasium.make(env_id, render_mode=render_mode)
        return env

    def save_weights(self, policy_dqn):
        """ 
        Save the trained model weights to a file.
        """
        sanitized_env_id = self.env_id.replace('/', '_')
        torch.save(policy_dqn.state_dict(), f'output/{sanitized_env_id+self.suffix}_{self.current_index:03d}.pt')

    ##### GPT GENERATED UTILS #####
    def estimate_value_function(self, policy_dqn, eval_steps=1000):
        """
        Estimate the value function by running the policy network for a fixed number of steps
        and computing the average max Q-values when selecting optimal actions.
        
        Args:
            policy_dqn: The policy network to evaluate
            eval_steps: Number of steps to run for evaluation (regardless of episodes)
            
        Returns:
            avg_value: Average value function (mean of max Q-values)
        """
        eval_env = self.create_env(self.env_id, render_mode=None)
        policy_dqn.eval()  # Set to evaluation mode
        
        all_max_q_values = []
        
        with torch.no_grad():
            state, _ = eval_env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)
            
            for step in range(eval_steps):
                # Get Q-values from policy network
                q_values = policy_dqn(state.unsqueeze(0)).squeeze()
                max_q_value = torch.max(q_values).item()
                all_max_q_values.append(max_q_value)
                
                # Select optimal action (greedy)
                action = q_values.argmax().item()
                
                # Take action
                new_state, reward, terminated, truncated, _ = eval_env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
                
                state = new_state
                
                # Reset environment if episode ended, but continue evaluation
                if terminated or truncated:
                    state, _ = eval_env.reset()
                    state = torch.tensor(state, dtype=torch.float32).to(device)
        
        eval_env.close()
        policy_dqn.train()  # Set back to training mode
        
        avg_value = np.mean(all_max_q_values) if all_max_q_values else 0.0
        
        return avg_value

    def save_hyperparameters(self, log_dir, total_steps):
        """Save hyperparameters and training config to JSON file"""
        # Create hyperparameters directory if it doesn't exist
        hyperparams_dir = os.path.join(log_dir, "hyperparameters")
        os.makedirs(hyperparams_dir, exist_ok=True)
        
        # Prepare hyperparameters data
        config_data = {
            "hyperparameter_set": self.hyperparameter_set,
            "training_config": {
                "total_steps": total_steps,
                "algorithm": self.__class__.__name__,
                "device": str(device),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            },
            "hyperparameters": self.hyperparameters
        }
        
        # Save to JSON file
        config_file = os.path.join(hyperparams_dir, "config.json")
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        # Also save a copy as YAML for easier reading
        yaml_file = os.path.join(hyperparams_dir, "config.yml")
        with open(yaml_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
        
        print(f"Hyperparameters saved to: {hyperparams_dir}")
        return config_file, yaml_file

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

        # Save frames as a GIF with index
        sanitized_env_id = self.env_id.replace('/', '_')
        from imageio import mimsave
        mimsave(f'output/{sanitized_env_id+self.suffix}_{steps}_{self.current_index:03d}.gif', frames, fps=30)