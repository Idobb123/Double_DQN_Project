from regularAgent import RegularAgent
from doubleAgent import DoubleAgent
import time

if __name__ == "__main__":
    # time the training
    start_time = time.time()
    print("just started")

    env_name = "acrobot"
    doubleagent = DoubleAgent(env_name)
    # doubleagent.train(render=False)
    #
    #
    # end_time = time.time()
    # print(f"Training time: {end_time - start_time} seconds")


    doubleagent.run("output/Acrobot-v1_ddqn_episode_0.pt")