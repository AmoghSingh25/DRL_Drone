import setup_path
import airsim
import numpy as np
import math
import time
from argparse import ArgumentParser
import gym
from gym import spaces, Env


class AirSimDroneEnv(Env):
    def __init__(self, step_length=0.2, image_shape=(144,256,1),n=3, max_steps = 20, clock_speed = 1):
        self.max_step = max_steps
        self.n = n
        self.drone = airsim.MultirotorClient()
        self.step_length = step_length
        self.goal_states = [[8.7,2.6],[10,-2.2],[13,1],[13,-12]]
        self.level_no = 0
        self.observation_space = spaces.Box(low=0, high = 255, shape=(2,*image_shape), dtype=np.float32)

            # spaces.Box(low=0, high = 255,shape=image_shape, dtype=np.float32),
            # spaces.Box(low=0, high = 255,shape=image_shape, dtype=np.float32),

        # self.observation_space = spaces.Dict({'back_image': spaces.Box(low=0, high=255, shape=image_shape, dtype=np.float32),
        # 'front_image': spaces.Box(low=0, high=255, shape=image_shape, dtype=np.float32),
        # 'velocity': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),
        # 'position': spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        # 'goal': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
        # })
        self.clock_speed = clock_speed
        self.prev_action = -1
        self.prev_reward = -1000

        # 'front_image': spaces.Box(low=0, high=255, shape=self.reduced_shape, dtype=np.float32),
        # 'back_image': spaces.Box(low=0, high=255, shape=self.reduced_shape, dtype=np.float32),
        # })
        # self.velocity = np.array([0, 0, 0], dtype=np.float32)

        self.state = {
            "position": np.zeros(3),
            "collision": False,
            "prev_position": np.zeros(3),
            'step': 0,
            'prev_action':-1,
            'best_action':-1,
            'prev_reward':-1
        }

        self.action_space = spaces.Discrete(7)
        # self.action_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.count_step = 0

        self.image_request = [
            airsim.ImageRequest("back_center_custom", airsim.ImageType.DepthPerspective,  pixels_as_float = True, compress = False),
            airsim.ImageRequest("front_center_custom", airsim.ImageType.DepthPerspective, pixels_as_float = True, compress = False),
        ]
        

        self.start_state = self.drone.getMultirotorState()
        self.start_pos = self.start_state.kinematics_estimated.position
        self._setup_flight()



    def __del__(self):
        self.drone.reset()

    def compress_image(self, image):
        x_step  =image.shape[0]//self.n
        y_step = image.shape[1]//self.n
        new_image = np.zeros((9,1))
        ct = 0
        for i in range(self.n):
            x_min = i*x_step
            x_max = (i+1)*x_step
            for j in range(self.n):
                y_min = j*y_step
                y_max = (j+1)*y_step
                curr_val = image[x_min][y_min]
                for k in range(x_min,x_max):
                    curr_val = min(curr_val, min(image[k]))
                new_image[ct] = curr_val
                ct+= 1
        new_image = new_image.reshape(9,)
        return new_image

    def _setup_flight(self):
        self.drone.reset()
        self.drone.enableApiControl(True)
        self.drone.armDisarm(True)

        x = self.start_pos.x_val
        y = self.start_pos.y_val
        z = self.start_pos.z_val
        self.drone.takeoffAsync(1)
        # self.drone.moveToPositionAsync(x, y, z-2, 1).join()
        # self.drone.moveByVelocityAsync(0, 0, 0, 1).join()

    def transform_obs(self, responses):
        img= np.array(responses.image_data_float)
        img_rgb = img.reshape(responses.height, responses.width, 1)
        # img_rgb = self.compress_image(img_rgb)
        img_rgb = np.array(img_rgb)
        img_rgb = np.array(img_rgb).astype(np.float32)
        return img_rgb

    def _debug_pos(self):
        print(self.state)

    def _get_obs(self):
        responses = self.drone.simGetImages(self.image_request)
        resp = []
        # resp = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width, 1)
        for i in range(len(responses)):
            responses[i] = self.transform_obs(responses[i])
            resp.append(responses[i])
            # responses[i] = self.compress_image(responses[i])
        # resp['front_image'] = responses[0]
        # resp['left_image'] = responses[1]
        # resp['right_image'] = responses[2]
        # resp['back_image'] = responses[1]
        # resp['velocity'] = np.array([self.drone.getMultirotorState().kinematics_estimated.linear_velocity.x_val, self.drone.getMultirotorState().kinematics_estimated.linear_velocity.y_val, self.drone.getMultirotorState().kinematics_estimated.linear_velocity.z_val])
        # resp['position'] = np.array([self.drone.getMultirotorState().kinematics_estimated.position.x_val, self.drone.getMultirotorState().kinematics_estimated.position.y_val, self.drone.getMultirotorState().kinematics_estimated.position.z_val])
        # resp['goal'] = np.array(self.goal_states[self.level_no])
        # resp['space_above'] = np.array([self.drone.getBarometerData().altitude])
        # print("Distance above = ", resp["space_above"])
        # image = self.transform_obs(responses)

        self.drone_state = self.drone.getMultirotorState()
        self.state["prev_position"] = self.state["position"]
        self.state["position"] = self.drone_state.kinematics_estimated.position
        self.state["velocity"] = self.drone_state.kinematics_estimated.linear_velocity
        self.state['step'] = self.count_step

        collision = self.drone.simGetCollisionInfo().has_collided
        self.state["collision"] = collision
        # if collision:
        #     resp["collision"] = 1
        # else:
        #     resp["collision"] = 0
        # self.observation_space['front_image'] = responses[0]
        # self.observation_space['back_image'] = responses[1]
        # temp = np.zeros(22)
        # pos = [self.state["position"].x_val, self.state["position"].y_val]
        # temp[:2]= pos[:2]
        # temp[2:4] = self.goal_states[0][:2]
        # temp[4:13] = responses[0]
        # temp[13:22] = responses[1]
        # temp = np.array(temp).astype(np.float32)
        return np.array(resp)

    def _do_action(self, action):

        # self.drone.moveByMotorPWMsAsync(
        #     float(action[0]),
        #     float(action[1]),
        #     float(action[2]),
        #     float(action[3]),
        #     self.step_length
        # ).join()


        # quad_offset = (0,0,0)
        # quad_offset = action
        # self.drone.moveByVelocityAsync(
        #     float(quad_offset[0]),
        #     float(quad_offset[1]),
        #     0,
        #     1
        # ).join()
        # time.sleep(1/self.clock_speed)

        if(action == 6):
            quad_offset = (0,0,0)
        else:
            quad_offset = self.interpret_action(action)
        self.drone.moveByVelocityAsync(
            float(quad_offset[0]),
            float(quad_offset[1]),
            float(quad_offset[2]),
            1/self.clock_speed
        ).join()
        self.drone.moveByVelocityAsync(0,0,0,1/self.clock_speed).join()
        
        
        

    def _compute_reward(self):
        beta = 1
        done = False
        z = -10
        start_pos = self.state["prev_position"]

        x_val, y_val = start_pos.x_val, start_pos.y_val
        
        dist_covered = np.linalg.norm(
            np.array([x_val, y_val]) - np.array([self.state["position"].x_val, self.state["position"].y_val])
        )

        dist_to_goal = np.linalg.norm(
            np.array([self.state["position"].x_val, self.state["position"].y_val]) - np.array([self.goal_states[self.level_no][0],self.goal_states[self.level_no][1]])
        )

        x_dist, y_dist = self.state["position"].x_val - self.goal_states[self.level_no][0],self.state["position"].y_val - self.goal_states[self.level_no][1]

        if(abs(x_dist) > abs(y_dist)):
            if(x_dist > 0):
                self.state['best_action'] = 1
            else:
                self.state['best_action'] = 0
        else:
            if(y_dist > 0):
                self.state['best_action'] = 2
            else:
                self.state['best_action'] = 3

        disp = np.linalg.norm(np.array([self.state["position"].x_val, self.state["position"].y_val]) - np.array([self.goal_states[self.level_no][0],self.goal_states[self.level_no][1]]))
        disp2 = np.linalg.norm(np.array([self.state["prev_position"].x_val, self.state["prev_position"].y_val]) - np.array([self.goal_states[self.level_no][0],self.goal_states[self.level_no][1]])) 
        
        reward = 0

        self.count_step += 1
        if self.state["collision"]:
            reward = -1000
            done = True
        elif dist_to_goal < 2.5:
            reward = 100 + (100 * self.level_no)
            self.level_no += 1
            print("Reached goal ", self.level_no)
            if(self.level_no == 4):
                done = True
        elif disp2 > disp:
            reward = 10
        else: 
            reward = (-1 * (dist_to_goal)**2 + 10*(self.level_no)) / 10
        
        
        if(self.count_step == self.max_step):
            done = True

        print("Reward = ",reward)
        return reward, done


    def step(self, action):
        self._do_action(action)
        obs = self._get_obs()
        reward, done = self._compute_reward()
        if(self.prev_action == -1):
            self.prev_action = action
            self.prev_reward = reward
        else:
            self.state['prev_action'] = self.prev_action
            self.state['prev_reward'] = self.prev_reward
            self.prev_action = action
            self.prev_reward = reward
        return obs, reward, done, self.state

    def reset(self):
        self._setup_flight()
        self.level_no = 0
        self.count_step = 0
        obs = self._get_obs()
        return obs, self.state

    def interpret_action(self, action):
        quad_offset = (0,0,0)
        if action == 0:
            quad_offset = (self.step_length, 0, 0)
        elif action == 1:
            quad_offset = (-self.step_length,0, 0)
        elif action == 2:
            quad_offset = (0, -self.step_length, 0)
        elif action==3:
            quad_offset = (0, self.step_length, 0)
        elif action==4:
            quad_offset = (0, 0, self.step_length)
        elif action==5:
            quad_offset = (0, 0, -self.step_length)

        return quad_offset

    def close(self):
        self.reset()
        
