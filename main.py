import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import torch
import numpy as np
from DQN import DQN
import ale_py

def state_to_dqn_input(state_np):
    """
    Convert a (84, 84, 4) numpy array into a torch tensor for DQN input.
    """
    print(f"Original state shape: {state_np.shape}")
    tensor = torch.tensor(state_np, dtype=torch.float32)
    # tensor = tensor.permute(2, 0, 1)  # (H, W, C) → (C, H, W)
    tensor = tensor.unsqueeze(0)     # Add batch dim → (1, C, H, W)
    return tensor

def play_trained_agent(model_path, episodes=1):
    gym.register_envs(ale_py)

    # Create env with render mode
    env = gym.make('MsPacmanNoFrameskip-v4', render_mode='human')
    env = AtariPreprocessing(env, grayscale_obs=True)
    env = FrameStackObservation(env, stack_size=4)

    # Determine number of actions
    num_actions = env.action_space.n

    # Load trained model
    model = DQN(num_actions)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            state_tensor = state_to_dqn_input(state)
            print(f"State shape: {state_tensor.shape}")

            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

        print(f"Episode {ep + 1} finished. Total reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    play_trained_agent("msPacMan_dqn.pt", episodes=3)
