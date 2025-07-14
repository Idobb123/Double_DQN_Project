import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
import torch
import numpy as np
from DQN import DQN
import ale_py

from dqn_trainer import DQNtrainer
from dqn_agent import DQNAgent

if __name__ == "__main__":
    gameName = 'BreakoutNoFrameskip-v4'
    weights_directory = 'output/weights'

    # Create the trainer and train the model
    trainer = DQNtrainer()
    weightsFileName = trainer.train(episodes=1000, gameName=gameName)  # Train the model
    model_path = f"{weights_directory}/{weightsFileName}"

    # Create the agent and play the game
    agent = DQNAgent(num_actions=trainer.num_actions, weights_path=model_path)
    agent.play(gameName, episodes=3)  # Play with the trained agent