# this is critic network using paddle and dynamic graph.

import gym
import paddle
import paddle.nn as nn
from itertools import count
from paddle.distribution import Normal
import numpy as np
from collections import deque
import random
import paddle.nn.functional as F

class CriticNetwork(nn.Layer):
    def __init__(self, state_dim=[2,2,1],action_dim=1,env_switch=0,evaluate_array_dim=(101,101),**kargs):
        # state = {"location": location,
        #          "direction": direction,
        #          "omega": omega,
        #          "evaluate_array": evaluate_array}        
        if 'LAYER_SIZE' in kargs:
            self.LAYER_SIZE = kargs['LAYER_SIZE']
        else:
            self.LAYER_SIZE = [500,500,500]         
        super(CriticNetwork, self).__init__()
        self.env_switch = env_switch
        if self.env_switch == 0 :
            self.state_dim = state_dim
            self.action_dim = action_dim
        else:
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.location_dim = state_dim[0] 
            self.direction_dim = state_dim[1]  
            self.omega_dim = state_dim[2] 
            self.evaluate_array_dim = evaluate_array_dim
        self.set_location()
        self.layers_linear = []
        print('CriticNetwork under construction')
        self.create_q_network(self.state_dim,self.action_dim,self.LAYER_SIZE)
    
    def set_location(self,location=r'E:\EnglishMulu\agents'):
        self.location = location + r'\saved_critic_networks'

    def create_q_network(self,state_dim,action_dim,LAYER_SIZE):
        N_layers = len(LAYER_SIZE)
        real_size = np.append(state_dim,LAYER_SIZE)
        real_size = np.append(real_size,np.array([1]))

        for i in range(len(real_size)-1):
            # get the omegas 
            if i == 0:
                # before add the action. 
                yiceng = nn.Linear(real_size[i]+action_dim, real_size[i+1])
                # yiceng = nn.Linear(real_size[i], real_size[i+1])
                # Wi = self.variable([real_size[i],real_size[i+1]],real_size[i])
            else: 
                # which means the lay next to the action. w2 and w3
                # Wi = self.variable([real_size[i],real_size[i+1]],real_size[i]+action_dim)
                yiceng = nn.Linear(real_size[i], real_size[i+1])
            self.layers_linear.append(yiceng)        
        self.relu = nn.ReLU()      

    def parameters(self):
        # 为了图自动化，我这个里面各层网格写成了list，导致它这个原版用不了了。
        """

        Returns a list of all Parameters from current layer and its sub-layers.

        Returns:
            list of Tensor, a list of Parameters.

        Examples:
            .. code-block:: python

                import paddle

                linear = paddle.nn.Linear(1,1)
                print(linear.parameters())  # print linear_0.w_0 and linear_0.b_0

        """     
        
        paramter_list = [] 
        for i in range(len(self.layers_linear)):
        #    paramter_list.append(self.layers_linear[i].parameters()) 
           paramter_list_single_layer = self.layers_linear[i].parameters()
           for j in range(len(paramter_list_single_layer)):
               paramter_list.append(paramter_list_single_layer[j])
        return paramter_list


    def forward(self, x, a):
        for i in range(len(self.layers_linear)):
            if i == 0 :
                x = paddle.concat((x, a), axis=2)
            x = self.relu(self.layers_linear[i](x))
            # if i == 0 :
            #    x = paddle.concat((x, a), axis=1)
        return x
    
    def save_network(self,adam,checkpoint,**kargs):
        if 'location' in kargs:
            location = kargs['location'] + r'\saved_critic_networks'
        else:
            location = self.location        
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/model_save_load_cn.html#moxingbaocunyujiazai
        canshu_name = location + r'\linear_net.pdparams'
        adam_name = location + r'\adam.pdopt'
        checkpoint_name = location + r'\final_checkpoint.pkl'

        # 保存Layer参数
        # paddle.save(layer.state_dict(), canshu_name)
        # for i in range(len(self.layers_linear)):
        #     canshu_name = location + r'\linear_net'+str(i) + r'.pdparams'
        #     paddle.save(self.layers_linear[i].state_dict(), canshu_name)

        paddle.save(self.state_dict(), canshu_name)
        paddle.save(adam.state_dict(), adam_name)
        paddle.save(checkpoint, checkpoint_name)
    
    def load_network(self,**kargs):
        if 'location' in kargs:
            location = kargs['location'] + r'\saved_critic_networks'
        else:
            location = self.location
        canshu_name = location + r'\linear_net.pdparams'
        adam_name = location + r'\adam.pdopt'
        checkpoint_name = location + r'\final_checkpoint.pkl'            

        try:           
            # 载入模型参数、优化器参数和最后一个epoch保存的检查点
            layer_state_dict = paddle.load(canshu_name)
            # final_checkpoint_dict = paddle.load(checkpoint_name)

            # 将load后的参数与模型关联起来
            self.set_state_dict(layer_state_dict)
            # adam.set_state_dict(opt_state_dict)
            adam_state_dict = paddle.load(adam_name)
            checkpoint_dict = paddle.load(checkpoint_name)            
            return adam_state_dict , checkpoint_dict
            # 打印出来之前保存的 checkpoint 信息
            # print("Loaded Final Checkpoint. Epoch : {}, Loss : {}".format(final_checkpoint_dict["epoch"], final_checkpoint_dict["loss"].numpy()))
        except:
            print('critic: attension, something wrong when loading')