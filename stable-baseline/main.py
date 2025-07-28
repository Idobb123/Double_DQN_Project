from dqn.dqn_train import train_dqn
from visualize import plot_training, plot_averaged_training
import time
from run_agent import run_agent

# Environment names
# "CartPole-v1"
# "MountainCar-v0"
# "LunarLander-v2"

if __name__ == "__main__":
    # start = time.time()
    # train_dqn()
    # end = time.time()
    # training_time = end - start
    # print(f"Training completed in {training_time:.2f} seconds.")

    plot_training("DQN", "./stable-baseline/dqn/train_logs/")
    # run_agent(agent_type="DQN", env_id="flappybird", max_steps=10000)
    # run_agent(agent_type="DQN", env_id="MountainCar-v0", max_steps=10000)
    # run_agent(agent_type="DQN", env_id="CartPole-v1", max_steps=1000)
