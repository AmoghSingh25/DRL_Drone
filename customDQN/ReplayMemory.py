import numpy as np
import random
import os
import pickle

class ReplayMemory:
    def __init__(self, state_shape, length=1000, load_memory=True):
        self.state_shape = state_shape
        self.length = length
        self.positive_reward_memory = np.array([])
        self.negative_reward_memory = np.array([])
        if(load_memory):
            self.__load_memory()
        
        '''{
            "state": np.zeros(*self.state_shape),
            "action": 0,
            "reward": 0,
            "next_state": np.zeros(*self.state_shape),
            "done": False,
        }
        '''
        os.system('cls')

    def __load_memory(self):
        with open('memory\\positive_memory.pkl', 'rb') as file: 
            self.positive_reward_memory = pickle.load(file)
        
        with open('memory\\negative_memory.pkl', 'rb') as file:
            self.negative_reward_memory = pickle.load(file)
        

    def store_memory(self, mem_content):
        if(mem_content['reward'] > 0):
            if(self.positive_reward_memory.shape[0] > self.length):
                self.positive_reward_memory = np.roll(self.positive_reward_memory, 1, axis=0)
                self.positive_reward_memory[0] = mem_content
            else:
                self.positive_reward_memory = np.append(self.positive_reward_memory, [mem_content], axis=0)
        else:
            if(self.negative_reward_memory.shape[0] > self.length):
                self.negative_reward_memory = np.roll(self.negative_reward_memory, 1, axis=0)
                self.negative_reward_memory[0] = mem_content
            else:
                self.negative_reward_memory = np.append(self.negative_reward_memory, [mem_content], axis=0)
    
    def get_memory(self, num_memory):
        pos_available_memory = min(num_memory,self.positive_reward_memory.shape[0])
        neg_available_memory = min(num_memory,self.negative_reward_memory.shape[0])
        positive_len =  random.randint(0,pos_available_memory)
        rem = num_memory - positive_len
        negative_len = min(rem, neg_available_memory)
        positive_mem_ret = np.random.choice(self.positive_reward_memory, positive_len)
        negative_mem_ret = np.random.choice(self.negative_reward_memory, negative_len)
        ret = np.concatenate([positive_mem_ret, negative_mem_ret])
        return ret


    def save_file(self):
        with open('memory\\positive_memory.pkl', 'wb+') as file: 
            pickle.dump(self.positive_reward_memory, file) 
        with open('memory\\negative_memory.pkl', 'wb+') as file: 
            pickle.dump(self.negative_reward_memory, file) 