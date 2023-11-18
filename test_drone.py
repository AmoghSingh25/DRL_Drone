import numpy as np
import os
import airsim
# from drone_env import AirSimDroneEnv
# import stable_baselines3 as sb3
# from stable_baselines3 import PPO, A2C, DQN
# # from sb3_contrib import QRDQN
# # from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.evaluation import evaluate_policy


# backup_path = os.path.join('Training', 'Backup', 'DQN_10000_4_Camera_Compress_stepLen_0_2_LR_0_1_MaxStep_20_Vel_add')
# save_path = os.path.join('Training', 'Backup', 'QRDQN_10000_4_Camera_Compress_stepLen_0_2_LR_0_1_MaxStep_20_Vel_add')
# save_path = os.path.join('Training', 'SavedModels', 'DQN_10000_4_Camera_Compress_stepLen_0_2_LR_0_1_MaxStep_20_Vel_add_SDE')
# log_path = os.path.join('Training', 'logs', 'DQN')


# env = AirSimDroneEnv(max_steps=20, clock_speed=20)
# env = Monitor(env, log_path)
# env.reset()
# check_env(env, warn=True)

drone = airsim.MultirotorClient()
drone.confirmConnection()
drone.enableApiControl(True)
drone.armDisarm(True)
drone.takeoffAsync().join()

# drone.moveToPositionAsync(13,1,-1, 5).join()

drone.moveByVelocityAsync(1,0,0, 5).join()
# drone.moveByMotorPWMsAsync(1,1,1,1,5).join()
# print("pos", drone.getMultirotorState().kinematics_estimated.position)

drone.reset()
# model = DQN.load(save_path, env, verbose=1)
# evaluate_policy(model, env, n_eval_episodes=10, render=False)