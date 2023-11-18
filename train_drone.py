import numpy as np
import os
from drone_env import AirSimDroneEnv
import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback


print("Starting")
log_path = os.path.join('Training', 'logs')



log_path = os.path.join('Training', 'logs', 'Small_env_DDPG')
save_path = os.path.join('Training', 'SavedModels', 'Small_env_DDPG_1')
backup_path = os.path.join('Training', 'Backup', 'Small_env_DDPG_1')
env = AirSimDroneEnv(step_length=1, max_steps=50, clock_speed=1)
# print(env.action_space.shape)
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
env = Monitor(env, log_path)
# new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
# model = sb3_contrib.QRDQN.load(backup_path, env, verbose = 1, tensorboard_log = log_path)

#2188
# Recurrent PPO, Maskable PPO, QRDQN, TRPO

# model = sb3_contrib.RecurrentPPO('MlpLstmPolicy', env, verbose=1, tensorboard_log = log_path,device="cuda", learning_rate=0.1)
# model = sb3.A2C('MultiInputPolicy', env, verbose=1, device="cuda", learning_rate=0.1)
# model = sb3.PPO('MultiInputPolicy', env, verbose=1, tensorboard_log = log_path)
# model = sb3.PPO('MlpPolicy', env, verbose=1, tensorboard_log = log_path)

# model = sb3_contrib.TRPO('MultiInputPolicy', env, verbose=1, device="cuda", learning_rate=0.1)
# model = sb3_contrib.RecurrentPPO('MultiInputLstmPolicy', env, verbose=1, tensorboard_log = log_path,device="cuda", learning_rate=0.1)
# model = QRDQN('MultiInputPolicy', env, verbose=1, tensorboard_log = log_path,device="cuda", learning_rate=0.1, replay_buffer_kwargs={'handle_timeout_termination': False}, buffer_size=1000)
# model = A2C.load(os.path.join('Training', 'SavedModels', 'A2C_100_4_Camera_LR_0.1_MaxStep_3'), env, verbose=1, tensorboard_log = log_path, learning_rate=0.5)
# model = sb3.DDPG('MultiInputPolicy', env, verbose=1, device="cuda", learning_rate=0.2, buffer_size=1000, learning_starts=100)
# model = sb3.DQN('MultiInputPolicy', env, verbose=1, device="cuda", learning_rate=0.2, buffer_size = 1000)
# model = sb3.DQN('MlpPolicy', env, verbose=1, device="cuda", learning_rate=0.2, buffer_size = 10000)
model = sb3.DQN.load(save_path, env)
try:
    print("Start training")
    model.learn(10000)
        # model = PPO('MlpPolicy', env, verbose=1, tensorboard_log = log_path)
        # model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log = log_path)
        # model = QRDQN('MultiInputPolicy', env, verbose=1, tensorboard_log = log_path)
    model.save(save_path)
        # evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print("Training completed")
except Exception as e:
    # print exception
    print(e)
    model.save(backup_path)
    print("Error in training, model saved")