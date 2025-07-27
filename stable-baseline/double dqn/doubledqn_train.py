import time
from doubledqn import DoubleDQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
import gymnasium as gym

source_dir = "stable-baseline/double dqn/"

# model parameters
env_id = "CartPole-v1"
learning_rate = 1e-3
buffer_size = 100_000  # Increased buffer size for better performance
learning_starts = 1000
batch_size = 32
gamma = 0.99
train_freq = 4
target_update_interval = 250
exploration_fraction = 0.1
exploration_final_eps = 0.02
tensorboard_log = f"./{source_dir}/train_logs/"
verbose = 0  

# total timesteps for training
total_timesteps = 300_000

def train_double_dqn():
    env = gym.make(env_id)

    model = DoubleDQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
    )
    # set our Double-DQN enabled flag
    model.double_dqn_enabled = True

    # Callback: evaluate every 10k steps, save best model
    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{source_dir}/best_model",
        log_path=f"{source_dir}/logs/",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print()
    print(f"Mean reward: {mean_reward}, Std: {std_reward}".center( 80, "="))

if __name__ == "__main__":
    time_list = []
    iterations = 10
    for i in range(iterations):
        start = time.time()
        train_double_dqn()
        end = time.time()
        training_time = end - start
        time_list.append(training_time)

    print(f"Average training time over {iterations} iterations: {sum(time_list) / iterations:.2f} seconds".center(80, "="))