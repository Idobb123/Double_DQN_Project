import torch
import flappy_bird_gymnasium
import gymnasium
from dqn import DQN
from replaymemory import ReplayMemory
import yaml
import random
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")  # Force CPU for compatibility

class Agent():
    def __init__(self, hyperparameter_set):
        with open('flappyBird/hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.episodes           = hyperparameters['episodes']               # number of episodes to train the agent
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # how often to sync the target network with the policy network
        self.discount_factor    = hyperparameters['discount_factor']   
        self.learning_rate      = hyperparameters['learning_rate']          # learning rate for the optimizer
        self.hidden_layer_size = hyperparameters['hidden_layer_size']       # size of the hidden layer in the DQN
        self.max_reward = hyperparameters['max_reward']                     # maximum reward for the environment

        self.save_frequency = 2000
        self.loss_fn = torch.nn.MSELoss()    # loss function for training

    def train(self, render=False):
        env = gymnasium.make(self.env_id, render_mode="human" if render else None)

        num_features = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(num_features, num_actions, self.hidden_layer_size).to(device)
        target_dqn = DQN(num_features, num_actions, self.hidden_layer_size).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        step_count = 0

        memory = ReplayMemory(self.replay_memory_size)
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate)  # optimizer for training

        rewards_per_episode = []
        epsilon_history = []
        epsilon = self.epsilon_init
        best_reward = float('-inf')

        for episode in range(self.episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32).to(device)  # Add batch dimension

            episode_reward = 0
            while True and episode_reward < self.max_reward:

                if random.random() < epsilon:
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64).to(device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

                new_state, reward, terminated, _, info = env.step(action.item())

                new_state = torch.tensor(new_state, dtype=torch.float32).to(device)  # Add batch dimension
                reward = torch.tensor(reward, dtype=torch.float32).to(device)

                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                episode_reward += reward
                step_count += 1

                if terminated:
                    break

            if len(memory) > self.mini_batch_size:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                if step_count % self.network_sync_rate == 0:
                    target_dqn.load_state_dict(policy_dqn.state_dict())

            rewards_per_episode.append(episode_reward)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            epsilon_history.append(epsilon)

            if episode % 10 == 0:
                print(f"Episode {episode + 1}/{self.episodes}, Reward: {episode_reward.item()}, Epsilon: {epsilon:.4f}")

            if episode % self.save_frequency == 0:
                self.save_graph(rewards_per_episode, epsilon_history)
                self.save_weights(policy_dqn, episode)

            if episode_reward > best_reward:
                best_reward = episode_reward
                current_time = time.strftime("%H:%M:%S", time.localtime())
                print(f"{current_time} : New best reward: {best_reward} at episode {episode + 1}")

        # save the trained model
        sanitized_env_id = self.env_id.replace('/', '_')
        torch.save(policy_dqn.state_dict(), f'flappyBird/{sanitized_env_id}_episode_{self.episodes}.pt')

        self.save_graph(rewards_per_episode, epsilon_history)
        self.save_weights(policy_dqn, self.episodes)

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * self.discount_factor * target_dqn(new_states).max(dim=1)[0]

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases

    def run(self, weights_path):
        # Create the environment
        env = gymnasium.make(self.env_id, render_mode="human")

        # Load the trained model
        num_features = env.observation_space.shape[0]
        num_actions = env.action_space.n

        self.policy_dqn = DQN(num_features, num_actions, self.hidden_layer_size).to(device)
        self.policy_dqn.load_state_dict(torch.load(weights_path, map_location=device))

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32).to(device)
        while True:
            with torch.no_grad():
                action = self.policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()
            new_state, reward, terminated, _, info = env.step(action)
            new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
            state = new_state

            if terminated:
                break

        env.close()

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(f"output/{self.env_id}_training_progress.png")
        plt.close(fig)

    def save_weights(self, policy_dqn, current_episode):
        sanitized_env_id = self.env_id.replace('/', '_')
        torch.save(policy_dqn.state_dict(), f'output/{sanitized_env_id}_episode_{current_episode}.pt')

if __name__ == "__main__":
    # time the training
    import time
    start_time = time.time()
    agent = Agent("cartpole1")
    # agent.train(render=False)
    # end_time = time.time()
    # print(f"Training time: {end_time - start_time} seconds")


    agent.run("output/CartPole-v1_episode_90000.pt")