import numpy as np
import os
from drone_env import AirSimDroneEnv
import tensorflow as tf
from tensorflow.keras import layers
from custom_dqn import DQNAgent
import subprocess
from datetime import datetime
import time
from tensorboard_logger import configure, log_value
from ReplayMemory import ReplayMemory

### Tensorboard
logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
configure(logdir)


env = AirSimDroneEnv(step_length=1, max_steps=50, clock_speed=1)
image_shape = (144,256,1)
state_shape = env.observation_space.shape
num_actions = env.action_space.n
memory = ReplayMemory(state_shape, load_memory=True, length=10000)
agent = DQNAgent(state_shape, num_actions, replay_memory = memory, load_model=True, epsilon_decay=0.90)

action_map = [
    "+x",
    "-x",
    "-y",
    "+y",
    "+z",
    "-z",
    "0",
]

try:

    for episode in range(1000):
        state, info = env.reset()
        # state = np.reshape(state, [1, *state_shape])

        total_reward = 0
        step_count = 0
        # os.system("cls")

        while True:
            if(episode == 0 and step_count == 1):
                os.system('cls')
            state1 = np.reshape(state[0], [1, *image_shape])
            state2 = np.reshape(state[1], [1, *image_shape])
            action = agent.select_action(state1, state2, info)
            if(action == -1):
                print("ERROR IN ACTION SELECT")
                # os.system('cls')
                continue
            next_state, reward, done, info = env.step(action)
            print ("\033[A \033[A")
            print("Episode", episode+1, "Action- ",action_map[action], "Reward = ",  int(reward), "Step = ",info['step'])
            # next_state = np.reshape(next_state, [1, *state_shape])
            next_state = next_state
            # print(state.shape)

            # state = np.reshape(state, [1, *state_shape])
            # next_state = np.reshape(next_state, [1, *state_shape])
            # agent.replay_buffer.add((state, action, reward, next_state, done))
            # state = np.array([state])
            # next_state = np.array([next_state])
            # print(state.shape)
            # print(next_state.shape)
            memory.store_memory(
                {
                    'state': state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done,
                    "info": info
                }
            )
            agent.train(state, action, reward, next_state,info, done)

            total_reward += reward
            step_count += 1
            state = next_state
            # os.system('cls')
            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Exploration Rate:{agent.epsilon}")
        log_value('EpReward', total_reward,episode+1)
        log_value('Steps',step_count , episode+1)
        log_value('AvgEpReward',total_reward / step_count , episode+1)
    agent.save_model()
    memory.save_file()
    print("Saving model")
    subprocess.call("TASKKILL /F /IM Blocks.exe", shell=True)
    
# except:
#     print("Saving model")
#     agent.save_model()
#     memory.save_file()
except Exception as e:
    print(e)
#     print("Saving model")
    agent.save_model()
    memory.save_file()