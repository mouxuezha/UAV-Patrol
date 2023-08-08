from collections import deque
import os
import random
import pickle
import numpy as np

class ReplayBuffer(object):
    def __init__(self, memory_size: int,**kargs) -> None:
        self.memory_size = memory_size
        self.buffer_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)
        if 'state_dim' in kargs:
            self.state_dim = kargs['state_dim']
        else:
            self.state_dim = 7 
        if 'action_dim' in kargs:
            self.action_dim = kargs['action_dim']
        else:
            self.action_dim = 2 
        self.num_experiences = 0 

    # def add(self, experience) -> None:
    #     self.buffer.append(experience)
    def add(self, state, action, reward, new_state, done):
        if len(state.shape) == 1:
            state = state
        elif len(state.shape) == 2:
            state = state.reshape(state.shape[0],)
        experience = (state, action, reward, new_state, done)
        # self.jilu_buffer(experience)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        changdu = len(self.buffer)
        return changdu
 
    def get_batch(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]     

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences
    
    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0    
    def save_buffer(self,location):
        # pickle.dump(self.buffer,open('C:/Users/y/Desktop'),'w')
        pickle.dump(self.buffer,open(location,'wb'))

    def load_buffer(self,location):
        # self.num_experiences = 0
        self.buffer = deque()

        try:
            self.buffer = pickle.load(open(location,'rb'))
        except:
            print('MXairfoil: no prepared buffer')
        self.num_experiences = len(self.buffer)        