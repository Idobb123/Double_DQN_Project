import gymnasium as gym
from FrozenLakeAgent import FrozenLakeAgent
from BlackJackAgent import BlackjackAgent

# env = gym.make('FrozenLake-v1', render_mode="human", map_name="4x4", is_slippery=False)
# env = gym.make('Blackjack-v1', natural=False, sab=False, render_mode="human")
env = gym.make('ALE/Pacman-v5')

lr = 0.1
epsilon = 0.1
epsilon_decay = 0.002
final_epsilon = 0
discount_factor = 1
agent = BlackjackAgent(env, lr, epsilon, epsilon_decay, final_epsilon, discount_factor)
# agent = FrozenLakeAgent(env, lr, epsilon, epsilon_decay, final_epsilon, discount_factor)

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = agent.get_action(observation)
    # action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    next_observation, reward, terminated, truncated, info = env.step(action)

    # Update agent
    agent.update(observation, action, float(reward), terminated, next_observation)
    observation = next_observation
    agent.decay_epsilon()

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
agent.print()
