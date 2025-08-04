from regularAgent import RegularAgent
from doubleAgent import DoubleAgent
import time

# env names
# LunarLander-v2
# FlappyBird-v0
# QWOP-v0

if __name__ == "__main__":

    configuration_name = "lunarlander4-windy"

    regularagent = RegularAgent(configuration_name)
    doubleagent = DoubleAgent(configuration_name)
    for i in range(1, 7):
        regularagent.train(render=False, total_steps=200_000, index=i)
        doubleagent.train(render=False, total_steps=200_000, index=i)


    # regularagent.run("output/MountainCar-v0_dqn_002.pt", episodes=3, max_steps=10_000, render=True)
    # doubleagent.run("output/LunarLander-v2_ddqn_20250804_111653.pt", episodes=50, max_steps=10_000, render=False)
