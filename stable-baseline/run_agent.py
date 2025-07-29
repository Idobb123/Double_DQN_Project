import gymnasium as gym
from stable_baselines3 import DQN
from training.dqn_train import create_env

def run_agent(agent_type="DQN", env_id="CartPole-v1", max_steps=1000, model_path=None):
    """Run the specified agent type on the given environment."""
    if model_path is None:
        if agent_type == "Double DQN":
            model_path = "stable-baseline/double dqn/best_model/best_model.zip"
        elif agent_type == "DQN":
            model_path = "stable-baseline/dqn/best_model/best_model.zip"
    
    # === Load environment ===
    env = create_env(env_id, render=True)
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