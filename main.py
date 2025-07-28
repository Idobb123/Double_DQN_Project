from regularAgent import RegularAgent
from doubleAgent import DoubleAgent
import time

if __name__ == "__main__":
    # time the training
    # start_time = time.time()
    # print("just started")

    configuration_name = "lunarlander1"
    # regularagent = RegularAgent(configuration_name)
    # regularagent.train(render=False, total_steps=300_000)

    doubleagent = DoubleAgent(configuration_name)
    doubleagent.train(render=False, total_steps=300_000)
    # end_time = time.time()
    # print(f"Training time: {end_time - start_time} seconds")

    # regularagent.run("output/LunarLander-v2_dqn.pt")
