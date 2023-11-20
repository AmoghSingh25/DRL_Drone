import numpy as np
import random
from collections import deque
import tensorflow as tf
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers





class DQNAgent:
    def __init__(self, state_shape, num_actions,replay_memory, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, load_model=True):
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.gamma = gamma  ## Weights to update Q values in Bellman Equation
        self.memory = replay_memory
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = 32
        self.front_model = self._create_dqn_model()
        self.back_model = self._create_dqn_model()
        self.q_model = self._create_q_model()
        if(load_model):
            self.__load_model()
        # self.target_model = _create_dqn_model(state_shape, num_actions)
        # self.target_model.set_weights(self.model.get_weights())
        # self.replay_buffer = ReplayBuffer(capacity=10000)
        os.system('cls')


    def __load_model(self):
        self.front_model = tf.keras.models.load_model('saved_model\\DQN_front_cam.keras')
        self.back_model = tf.keras.models.load_model('saved_model\\DQN_back_cam.keras')
        self.q_model = tf.keras.models.load_model('saved_model\\DQN_q.keras')


    def _create_q_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=(2,7)),
            layers.Dense(8, activation='softmax'),
            layers.Dense(16),
            layers.Dense(34, activation='softmax'),
            layers.Dense(64),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Flatten(),
            layers.Reshape((-1, 256)),
            layers.LSTM(256),
            layers.Dense(self.num_actions, activation='linear')
        ])
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.2,decay_steps=10000,decay_rate=0.45)
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', run_eagerly=True)
        return model

    def _create_dqn_model(self):
        model = tf.keras.Sequential([
            layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape[1:]),
            layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Flatten(),
            layers.Reshape((-1, 256)),
            layers.LSTM(256),
            layers.Dense(self.num_actions-1, activation='linear')
        ])
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.2,decay_steps=10000,decay_rate=0.45)
        optimizer = Adam(learning_rate=lr_schedule)
        model.compile(optimizer=optimizer, loss='mse', run_eagerly=True)
        return model

    def select_action(self, state1, state2, info):
    
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        front_q_values = self.front_model.predict(state1, verbose='0')[0]
        back_q_values = self.back_model.predict(state2, verbose='0')[0]
        q_values = []
        q_values.extend(front_q_values)
        q_values.extend(back_q_values)
        q_values.extend([info['prev_action'], info['prev_reward']])
        q_values = np.array(q_values)
        q_values = q_values.reshape((1,2,7))
        combined_q_values = self.q_model.predict(q_values, verbose='0')[0]
        ## Front Q Values - Actions - {0,2,3,4,5,6}
        ## Back Q Values - Actions -  {1,2,3,4,5,6}
        # q_values[0] = front_q_values[0]
        # q_values[1] = back_q_values[0]
        # q_values[2:] = front_q_values[1:]
        # q_values[2:] += back_q_values[1:]
        # q_values[2:] = q_values[2:] / 5
        # q_values = np.append(q_values, [])
        return np.argmax(combined_q_values)
        # except:
        #     return -1

    def train(self,states, actions, rewards, next_states,info, dones, batch_size=32):
        # if len(smelf.replay_buffer.buffer) < batch_size:
        #     return
        # states, actions, rewards, next_states, dones
        # states = np.concatenate(states)
        # next_states = np.concatenate(next_states)
        # targets = self.model.predict(states)
        # print(states.shape)
        back_targets = self.back_model.predict(np.array([states[0]]), verbose='0')
        front_targets = self.front_model.predict(np.array([states[1]]) , verbose='0')
        front_target = rewards
        back_target = rewards
        q_model_input = np.array([back_targets[0], front_targets[0]])
        q_model_input = np.append(q_model_input, [info['prev_action'], info['prev_reward']])
        q_model_input = np.reshape(q_model_input, (2,7))
        target_q_model = np.zeros(self.num_actions)
        target_q_model[info['best_action']] = 1
        self.q_model.fit(np.array([q_model_input]), np.array([target_q_model]), epochs=1, verbose='0')
        if not dones:
            back_target += self.gamma * np.amax(back_targets[0])
            front_target += self.gamma * np.amax(front_targets[0])
        back_targets = back_targets
        front_targets = front_targets
        if(actions != 0):
            back_targets[0][actions-1] = back_target
        if(actions != 1):
            if(actions == 0):
                front_targets[0][0] = front_target
            else:
                front_targets[0][actions-1] = front_target
        
        self.back_model.fit(np.array([states[0]]), back_targets, epochs=1, verbose='0')
        self.front_model.fit(np.array([states[1]]), front_targets, epochs=1, verbose='0')
        
        # targets[0][actions] = target

        # self.model.fit(states, targets, epochs=1, verbose='0')
        
        if(not dones):
           return
        ## Offline training
        chance = random.random()
        if(chance > 0.98):
            batch_size = min(1000, len(self.memory.positive_reward_memory))
        elif(chance > 0.9):
            batch_size = 128
        elif(chance > 0.8):
            batch_size = 64
        minibatch = self.memory.get_memory(batch_size)
        print("Training on memory, Size=",len(minibatch), end="   ---   ")
        back_states = np.array([x['state'][0] for x in minibatch])
        front_states = np.array([x['state'][1] for x in minibatch])
        back_targets = self.back_model.predict(back_states, verbose='0')
        front_targets = self.front_model.predict(front_states, verbose='0')
        q_model_input = np.zeros((len(minibatch),2,7))
        q_model_output = np.zeros((len(minibatch), self.num_actions))
        for i in range(len(minibatch)):
            # state = minibatch[i]['state']
            action = minibatch[i]['action']
            reward = minibatch[i]['reward']
            # next_state = minibatch[i]['next_state']
            info = minibatch[i]['info']
            done = minibatch[i]['done']
            back_target = reward
            front_target = reward
            q_values = np.array(front_targets[i])
            q_values = np.append(q_values,back_targets[i])
            q_values = np.append(q_values, [info['prev_action'], info['prev_reward']])
            # q_values = np.array(q_values)
            q_values = q_values.reshape((1,2,7))
            combined_q_values = self.q_model.predict(q_values, verbose='0')[0]
            q_model_input[i] = combined_q_values
            q_model_output[i][info['best_action']] = 1
            if not done:
                back_target += self.gamma * np.amax(back_targets[i])
                front_target += self.gamma * np.amax(front_targets[i])
            if(action != 0):
                back_targets[i][action-1] = back_target
            if(action != 1):
                if(actions == 0):
                    front_targets[i][0] = front_target
                else:
                    front_targets[i][action-1] = front_target
        self.back_model.fit(back_states, back_targets, epochs=1, verbose='0')
        self.front_model.fit(front_states, front_targets, epochs=1, verbose='0')
        self.q_model.fit(q_model_input, q_model_output, epochs=1, verbose='0')
        print("Done", end="\n")
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # def update_target_model(self):
    #     self.target_model.set_weights(self.model.get_weights())

    def save_model(self):
        self.back_model.save('saved_model\\DQN_back_cam.keras')
        self.front_model.save('saved_model\\DQN_front_cam.keras')
        self.q_model.save('saved_model\\DQN_q.keras')