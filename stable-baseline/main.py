from dqn.dqn_train import train_dqn
from visualize import plot_training, plot_averaged_training
import time
from run_agent import run_agent

# Environment names
# "CartPole-v1"
# "MountainCar-v0"
# "LunarLander-v2"
# "flappybird"

if __name__ == "__main__":
    output_dir = "mountaincar"  # Change this to select different output directories
    config_name = "car2"
    env_id = "MountainCar-v0"

    train_enabled = False
    plot_enabled = True
    play_enabled = False

    # train agent
    if train_enabled:
        start = time.time()
        train_dqn(config_name=config_name, output_path=f"stable_baseline/dqn/output/{output_dir}/{config_name}")
        end = time.time()
        print(f"Training time: {end - start:.2f} seconds".center(80, "="))

    # present training and evaluation results
    plot_name = f"DQN {config_name} training"
    data_path = f"./stable-baseline/dqn/output/{output_dir}/{config_name}"
    model_path = f"./{data_path}/best_model/best_model.zip"
    
    if plot_enabled:
        plot_training(plot_name, data_path, save_dir=data_path)
    
    if play_enabled:
        run_agent(agent_type="DQN", env_id=env_id, max_steps=10000, model_path=model_path)