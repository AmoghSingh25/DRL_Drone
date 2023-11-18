import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
import gym
from gym import spaces, Env


class AirSimDroneEnv2(Env):
    def __init__(self, step_length=0.2, image_shape=(144,256,1), max_steps = 5):
        self.max_step = max_steps

        self.drone = airsim.MultirotorClient()
        self.step_length = step_length
        self.image_shape = image_shape
        self.observation_space = spaces.Dict({'left_image': spaces.Box(low=0, high=255, shape=image_shape, dtype=np.float32),
        'right_image': spaces.Box(low=0, high=255, shape=image_shape, dtype=np.float32),
        'front_image': spaces.Box(low=0, high=255, shape=image_shape, dtype=np.float32),
        })

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
            'step': 0
        }

        self.action_space = spaces.Discrete(6)
        self.count_step = 0

        self.image_request = [
            airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPerspective, True),
        airsim.ImageRequest("front_left_custom", airsim.ImageType.DepthPerspective, True ),
        airsim.ImageRequest("front_right_custom", airsim.ImageType.DepthPerspective, True ),
        ]
        

        self.start_state = self.drone.getMultirotorState()
        self.start_pos = self.start_state.kinematics_estimated.position
        self._setup_flight()



    def __del__(self):
        self.drone.reset()

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        # Set home position and velocity
        x = self.start_pos.x_val
        y = self.start_pos.y_val
        z = self.start_pos.z_val
        self.drone.moveToPositionAsync(x, y, z, 5).join()
        # self.drone.takeoffAsync(timeout_sec= 2).join()
        self.drone.moveByVelocityAsync(0, 0, 0, 1).join()

    def transform_obs(self, responses):
        img= np.array(responses.image_data_float)
        img_rgb = img.reshape(responses.height, responses.width, 1)
        return img_rgb

    def _get_obs(self):
        responses = self.drone.simGetImages(self.image_request)
        resp = {}
        for i in range(len(responses)):
            responses[i] = self.transform_obs(responses[i])
        resp['front_image'] = responses[0]
        resp['left_image'] = responses[1]
        resp['right_image'] = responses[2]
        self.drone_state = self.drone.getMultirotorState()

        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        self.state['step'] = self.count_step

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision
        return resp

    def _do_action(self, action):
        quad_offset = self.interpret_action(action)
        quad_vel = self.drone.getMultirotorState().kinematics_estimated.linear_velocity
        self.drone.moveByVelocityAsync(
            quad_offset[0] + quad_vel.x_val,
            quad_offset[1]+ quad_vel.y_val,
            quad_offset[2]+ quad_vel.z_val,
            2,
        ).join()

    def _compute_reward(self):
        thresh_dist = 7
        beta = 1
        done = False
        z = -10

        start_pos = self.state["prev_position"]

        x_val, y_val = start_pos.x_val, start_pos.y_val

        dist_covered = np.linalg.norm(
            np.array([x_val, y_val]) - np.array([self.state["position"].x_val, self.state["position"].y_val])
        )
        disp = np.linalg.norm(np.array([self.state["position"].x_val, self.state["position"].y_val]) - np.array([self.start_pos.x_val, self.start_pos.y_val])) - np.linalg.norm(np.array([self.state["prev_position"].x_val, self.state["prev_position"].y_val]) - np.array([self.start_pos.x_val, self.start_pos.y_val])) 

        reward = 0

        self.count_step += 1
        collision = self.drone.simGetCollisionInfo().has_collided
        if collision:
            reward = -1000
            done = True
        else:
            reward = 10 * dist_covered + 30 * disp
        
        if(self.count_step == self.max_step):
            done = True

        print("Reward = ",reward)
        return reward, done


    def step(self, action):
        self._do_action(action)
        # self.drone.moveByVelocityAsync(0, 0,0, 1)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        self.count_step = 0
        obs = self._get_obs()
        return obs

    def interpret_action(self, action):
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (0, self.step_length, 0)
        elif action == 2:
            quad_offset = (0, 0, self.step_length)
        elif action == 3:
            quad_offset = (-self.step_length, 0, 0)
        elif action == 4:
            quad_offset = (0, -self.step_length, 0)
        elif action == 5:
            quad_offset = (0, 0, -self.step_length)

        return quad_offset

    def close(self):
        self.reset()
        
