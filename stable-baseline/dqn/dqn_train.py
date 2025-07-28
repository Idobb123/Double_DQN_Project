# # Suppress TensorFlow messages BEFORE any imports
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING messages
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN to stop the messages entirely

import time
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import yaml
from torch import nn
import flappy_bird_gymnasium

source_dir = "stable-baseline/dqn/"
config_name = "flappybird1"  # Change this to select different configurations

# Load model parameters from YAML config file (same pattern as agent.py)
def load_hyperparameters(hyperparameter_set):
    """Load hyperparameters from YAML file, same pattern as agent.py"""
    config_path = f"{source_dir}/config.yml"
    with open(config_path, 'r') as file:
        all_hyperparameter_sets = yaml.safe_load(file)
        hyperparameters = all_hyperparameter_sets[hyperparameter_set]
    return hyperparameters

def create_env(env_id, render=False):
    """Create the environment using the specified env_id"""
    render_mode = "human" if render else None
    if env_id == "flappybird":
        env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)
        return env
    elif env_id == "qwop":
        pass
    return gym.make(env_id, render_mode=render_mode)

def create_model(env, params):
    # load hyperparameters
    learning_rate = params['learning_rate']
    buffer_size = params['buffer_size']
    learning_starts = params['learning_starts']
    batch_size = params['batch_size']
    gamma = params['gamma']
    train_freq = params['train_freq']
    target_update_interval = params['target_update_interval']
    exploration_fraction = params['exploration_fraction']
    exploration_final_eps = params['exploration_final_eps']
    verbose = params['verbose']
    tensorboard_log = f"./{source_dir}/train_logs/"
    network_layers = params['network_layers']
    network_activation = params['network_activation']

    policy_kwargs = {
        'net_arch': network_layers,
        'activation_fn': getattr(nn, network_activation)  # Convert string to activation function
    }

    model = DQN(
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
        policy_kwargs=policy_kwargs
    )
    
    return model

def train_dqn():
    params = load_hyperparameters(config_name)
    env = create_env(params['env_id'])
    model = create_model(env, params)

    # Callback: evaluate every 10k steps, save best model
    eval_env = create_env(params['env_id'])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{source_dir}/best_model",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )

    # Train the model
    model.learn(total_timesteps=params['total_timesteps'], callback=eval_callback)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print()
    print(f"Mean reward: {mean_reward}, Std: {std_reward}".center( 80, "="))

if __name__ == "__main__":
    start = time.time()
    train_dqn()
    end = time.time()
    training_time = end - start
    print(f"Training time: {training_time:.2f} seconds".center(80, "="))