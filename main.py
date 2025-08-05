from regularAgent import RegularAgent
from doubleAgent import DoubleAgent
import time

# env names
# LunarLander-v2
# FlappyBird-v0
# QWOP-v0

if __name__ == "__main__":

    configuration_name = "lunarlander4-windy"
    configuration_name = "flappybird1"
    training_steps = 2_000_000

    regularagent = RegularAgent(configuration_name)
    doubleagent = DoubleAgent(configuration_name)
    # for i in range(1, 2):
    #     regularagent.train(render=False, total_steps=training_steps, index=i)
    #     doubleagent.train(render=False, total_steps=training_steps, index=i)


    regularagent.run("output/FlappyBird-v0_ddqn_001.pt", episodes=50, max_steps=500_000, render=False, show_rewards=True)
    regularagent.run("output/FlappyBird-v0_dqn_001.pt", episodes=50, max_steps=500_000, render=False, show_rewards=True)
    # doubleagent.run("output/LunarLander-v2_ddqn_20250804_111653.pt", episodes=50, max_steps=10_000, render=False)
