import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3 import DQN
import ale_py

def train_and_play_pacman():
    # Create the raw environment
    gym.register_envs(ale_py)
    env = gym.make("MsPacmanNoFrameskip-v4", render_mode="human", obs_type="grayscale")

    # Preprocess: skip frames, scale, etc.
    env = AtariPreprocessing(
        env,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True
    )
    # Stack last 4 frames
    env = FrameStackObservation(env, stack_size=4)

    # Build the DQN agent
    model = DQN(
        policy="CnnPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        verbose=1
    )

    # Train the agent
    model.learn(total_timesteps=200_000)
    model.save("pacman_dqn_gymnasium")

    # Play forever
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if done or truncated:
            obs, info = env.reset()

if __name__ == "__main__":
    train_and_play_pacman()
