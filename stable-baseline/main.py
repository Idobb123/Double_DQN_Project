from training.dqn_train import train_dqn, get_suffix, load_hyperparameters
from visualize import visualize_single_training_run
import time
from run_agent import run_agent

# Environment names
# "CartPole-v1"
# "MountainCar-v0"
# "LunarLander-v2"
# "flappybird"

if __name__ == "__main__":
    config_name = "lunarlander2"
    env_id = "LunarLander-v2"
    source_dir = f"stable-baseline/training"  # source directory of the project

    train_enabled = True  # THIS WILL OVERWRITE THE MODEL
    double_dqn_enabled = False  # Set to True to enable Double DQN, False for standard DQN
    plot_enabled = True
    play_enabled = False

    # train agent
    if train_enabled:
        start = time.time()
        train_dqn(config_name=config_name, double_dqn=double_dqn_enabled)
        end = time.time()
        print(f"Training time: {end - start:.2f} seconds".center(80, "="))

    # present training and evaluation results
    plot_name = f"{get_suffix(double_dqn_enabled)} {config_name}"
    data_path = f"{source_dir}/output/{env_id}/{config_name}/{get_suffix(double_dqn_enabled)}"
    model_path = f"./{data_path}/best_model/best_model.zip"
    
    if plot_enabled:
        print(f"Now viewing the training results in {data_path}".center(80, "="))
        visualize_single_training_run(plot_name, data_path, save_dir=data_path)

    if play_enabled:
        run_agent(agent_type="DQN", env_id=env_id, max_steps=10000, model_path=model_path)
        # run_agent(agent_type="DQN", env_id=env_id, max_steps=1000, model_path="./stable-baseline/training/output/LunarLander-v2/lunarlander1/lunarlander1_best_model")