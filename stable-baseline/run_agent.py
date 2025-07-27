import gymnasium as gym
from stable_baselines3 import DQN

agent_type = "Double DQN"  # "DQN" or "Double DQN"
agent_type = "DQN"  # "DQN" or "Double DQN"
env_id = "CartPole-v1"
max_steps = 1000  # Maximum steps per episode


if agent_type == "Double DQN":
    model_path = "stable-baseline/double dqn/best_model/best_model.zip"
elif agent_type == "DQN":
    model_path = "stable-baseline/dqn/best_model/best_model.zip"

# === Load environment ===
env = gym.make(env_id, render_mode="human")  # render_mode="human" shows window
env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)

# === Load the saved model ===
model = DQN.load(model_path)  # Adjust the path if needed

# === Run the agent ===
obs, _ = env.reset()
for step in range(max_steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    if terminated:
        print("Agent terminated the episode.")
        obs, _ = env.reset()

print()
print("Testing completed.")

env.close()