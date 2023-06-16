# this is actor network using paddle and dynamic graph.
import gym
import paddle
import paddle.nn as nn
from itertools import count
from paddle.distribution import Normal
import numpy as np
from collections import deque
import random
import paddle.nn.functional as F

class ActorNetwork(nn.Layer):
    def __init__(self,**kargs):
        if 'LAYER_SIZE' in kargs:
            self.LAYER_SIZE = kargs['LAYER_SIZE']
        else:
            self.LAYER_SIZE = [500,500,500] 
        if 'LEARNING_RATE' in kargs:
            self.LEARNING_RATE = kargs['LEARNING_RATE']
        else:
            self.LEARNING_RATE = 1e-5
        if 'TAU' in kargs:
            self.TAU = kargs['TAU']
        else: 
            self.TAU = 0.005
        if 'L2' in kargs:
            self.L2 = kargs['L2']
        else:
            self.L2 = 0.01        
        self.state_dim = state_dim
        self.action_dim = action_dim 
        super(ActorNetwork, self).__init__()
        self.layers_linear = [] 
        print('CriticNetwork under construction')

    def create_network(self,state_dim,action_dim,LAYER_SIZE,is_train=True):
        N_layers = len(LAYER_SIZE)
        real_size = np.append(state_dim,LAYER_SIZE)
        real_size = np.append(real_size,np.array([1]))

        for i in range(N_layers):
            # get the omegas 
            if i == 0:
                # before add the action. 
                yiceng = nn.Linear(real_size[i]+action_dim, real_size[i+1])
                # Wi = self.variable([real_size[i],real_size[i+1]],real_size[i])
            else: 
                # which means the lay next to the action. w2 and w3
                # Wi = self.variable([real_size[i],real_size[i+1]],real_size[i]+action_dim)
                yiceng = nn.Linear(real_size[i], real_size[i+1])
            self.layers_linear.append(yiceng)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.noisy = Normal(0, 0.2)
        self.is_train = is_train  

    def forward(self,x):
        for i in range(len(self.layers_linear)):
            if i == len(self.layers_linear)-1:
                x = self.tanh(self.layers_linear[i](x))
            else:
                x = self.relu(self.layers_linear[i](x))
            
        return x 
    
    def select_action(self, epsilon, state):
        state = paddle.to_tensor(state,dtype="float32").unsqueeze(0) # 这里肯定得后面再来重写的。要把卷积整进去。
        with paddle.no_grad():
            action = self.forward(state).squeeze() + self.is_train * epsilon * self.noisy.sample([1]).squeeze(0)
        return 2 * paddle.clip(action, -1, 1).numpy() 
    
    