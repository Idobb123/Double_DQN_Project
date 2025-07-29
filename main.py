from regularAgent import RegularAgent
from doubleAgent import DoubleAgent
import time

# env names
# LunarLander-v2
# FlappyBird-v0
# QWOP-v0

if __name__ == "__main__":
    # time the training
    start_time = time.time()
    # print("just started")

    configuration_name = "flappybird2"

    # regularagent = RegularAgent(configuration_name)
    # regularagent.train(render=False, total_steps=2_000_000)
    # regularagent.run("output/FlappyBird-v0_dqn.pt", episodes=3, max_steps=100_000)

    doubleagent = DoubleAgent(configuration_name)
    doubleagent.train(render=False, total_steps=2_000_000)
    doubleagent.run("output/FlappyBird-v0_ddqn.pt", episodes=3, max_steps=100_000)

    end_time = time.time()
    print(f"Training time: {end_time - start_time} seconds")