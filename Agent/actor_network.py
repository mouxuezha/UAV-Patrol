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
    def __init__(self,state_dim=[4,101], action_dim =1 , **kargs):
        if 'LAYER_SIZE' in kargs:
            self.LAYER_SIZE = kargs['LAYER_SIZE']
        else:
            self.LAYER_SIZE = [500,500,500] 
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
    
    def save_network(self,**kargs):
        if 'location' in kargs:
            location = kargs['location'] + r'\saved_actor_networks'
        else:
            location = self.location        
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/model_save_load_cn.html#moxingbaocunyujiazai
        canshu_name = location + r'\linear_net.pdparams'
        adam_name = location + r'\adam.pdopt'
        checkpoint_name = location + r'\final_checkpoint.pkl'
        paddle.save(self.state_dict(), canshu_name)

    
    def load_network(self,**kargs):
        if 'location' in kargs:
            location = kargs['location'] + r'\saved_actor_networks'
        else:
            location = self.location
        canshu_name = location + r'\linear_net.pdparams'
        adam_name = location + r'\adam.pdopt'
        checkpoint_name = location + r'\final_checkpoint.pkl'            
        layer_state_dict = paddle.load(canshu_name)
        final_checkpoint_dict = paddle.load(checkpoint_name)

        self.set_state_dict(layer_state_dict)

        print("Loaded Final Checkpoint. Epoch : {}, Loss : {}".format(final_checkpoint_dict["epoch"], final_checkpoint_dict["loss"].numpy()))
