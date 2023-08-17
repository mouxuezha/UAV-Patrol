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
from paddle.nn import Conv2D, MaxPool2D

class CriticNetwork(nn.Layer):
    def __init__(self, state_dim=[2,2,1,101],action_dim=1,env_switch=0,**kargs):
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
        elif self.env_switch == 1:
            self.state_dim = state_dim
            self.action_dim = action_dim
            self.location_dim = state_dim[0] 
            self.direction_dim = state_dim[1]  
            self.omega_dim = state_dim[2] 
            self.CNN_out_dim = 5
            self.evaluate_array_dim = (state_dim[-1],state_dim[-1])
        self.set_location()
        self.layers_linear = []
        self.layers_CNN = [] 
        
        print('CriticNetwork under construction')
        self.create_q_network(self.state_dim,self.action_dim,self.LAYER_SIZE)
    
    def set_location(self,location=r'E:\EnglishMulu\agents'):
        self.location = location + r'\saved_critic_networks'

    def create_q_network(self,state_dim,action_dim,LAYER_SIZE):
        N_layers = len(LAYER_SIZE)
        if len(self.state_dim)>0:  # 由于state是dic所以这里需要处理一下
            state_dim_liner = state_dim[0] + state_dim[1] + state_dim[2] + self.CNN_out_dim
        else:
            state_dim_liner = state_dim
        real_size = np.append(state_dim_liner,LAYER_SIZE)
        real_size = np.append(real_size,np.array([action_dim]))


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

        self.creat_CNN(asdasdsadas)  

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
        
        # 这个是加了CNN部分之后才有的。
        for i in range(len(self.layers_CNN)):
            paramter_list_single_layer = self.layers_CNN[i].parameters()
            for j in range(len(paramter_list_single_layer)):
               paramter_list.append(paramter_list_single_layer[j])            
        return paramter_list

    def creat_CNN(self, H=100, W=100):

        in_channels = 1
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        conv1 = Conv2D(in_channels=in_channels, out_channels=20, kernel_size=5, stride=1, padding=2)
        H,W = self.Conv2D_parameter_change(H,W, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        H,W = self.Conv2D_parameter_change(H, W, padding=0, kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        H,W = self.Conv2D_parameter_change(H, W, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        H,W = self.Conv2D_parameter_change(H, W, padding=0, kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是1【我这个不能是1,但是5也是随手给的】
        fc = nn.Linear(in_features=H*W, out_features=self.CNN_out_dim)
        
        self.layers_CNN.append(conv1)
        self.layers_CNN.append(max_pool1)
        self.layers_CNN.append(conv2)
        self.layers_CNN.append(max_pool2)
        self.layers_CNN.append(fc)

    
    def Conv2D_parameter_change(self,H_in,W_in, paddings=2,dilations=1,kernel_size=5,strides=1):
        # CNN和max_pool都是这个公式，不重新写了。
        H_out = (H_in+2*paddings-(dilations*(kernel_size-1)+1))/strides+1
        W_out = (W_in+2*paddings-(dilations*(kernel_size-1)+1))/strides+1
        return H_out, W_out   

    def forward(self, x, a):
        x_CNN = x["CNN"]   #2023-8-10 22:51:06写到这里，还没有最终定下来这个怎么说。
        
        for i in range(len(self.layers_CNN)):
            x_CNN = self.relu(self.layers_linear[i](x_CNN))
        
        x_state = x["state"]
        for i in range(len(self.layers_linear)):
            if i == 0 :
                x = paddle.concat((x_state, a, x_CNN), axis=2)
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