from regularAgent import RegularAgent
from doubleAgent import DoubleAgent
import time

"""
First set the configuration you wish to run in 'hyperparameters.yml' file.
Then run this script to train the agent, with respect to the configuration.
"""

if __name__ == "__main__":
    # Set the configuration name and training steps
    configuration_name = "flappybird1"
    training_steps = 1_000_000

    regularagent = RegularAgent(configuration_name)
    doubleagent = DoubleAgent(configuration_name)

    regularagent.train(render=False, total_steps=training_steps, index=1)
    doubleagent.train(render=False, total_steps=training_steps, index=1)

    ##### You can run the trained agents #####
    # regularagent.run("output/FlappyBird-v0_ddqn_001.pt", episodes=50, max_steps=500_000, render=False, show_rewards=True)
    # regularagent.run("output/FlappyBird-v0_dqn_001.pt", episodes=50, max_steps=500_000, render=False, show_rewards=True)