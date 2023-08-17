# this is actor network using paddle and dynamic graph. and CNN
import paddle
import paddle.nn as nn
from paddle.distribution import Normal
import numpy as np
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D

class ActorNetwork(nn.Layer):
    def __init__(self,state_dim=[2,2,1,101],action_dim=1 , env_switch =0,**kargs):
        if 'LAYER_SIZE' in kargs:
            self.LAYER_SIZE = kargs['LAYER_SIZE']
        else:
            self.LAYER_SIZE = [500,500,500] 
        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.env_switch = env_switch
        super(ActorNetwork, self).__init__()
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
        self.layers_linear = []
        self.layers_CNN = [] 
        print('ActorNetwork under construction')
        self.create_network(self.state_dim,self.action_dim,self.LAYER_SIZE,is_train=True)
        self.creat_CNN(H=self.evaluate_array_dim[0],W=self.evaluate_array_dim[1])  
        

    def create_network(self,state_dim,action_dim,LAYER_SIZE,is_train=True):
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
                # yiceng = nn.Linear(real_size[i]+action_dim, real_size[i+1])
                yiceng = nn.Linear(real_size[i], real_size[i+1])
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

    def forward(self,x):
        for i in range(len(self.layers_linear)):
            if i == len(self.layers_linear)-1:
                x = self.tanh(self.layers_linear[i](x))
            else:
                x = self.relu(self.layers_linear[i](x))
            
        return x 
    
    def select_action(self, epsilon, state):
        if type(state)==paddle.Tensor:
            state = state
        elif type(state)==np.ndarray:
            if len(state.shape) == 1:
                state = state
            elif len(state.shape) == 2:
                state = state.reshape(state.shape[0],)
            state = paddle.to_tensor(state,dtype="float32").unsqueeze(0) # 这里肯定得后面再来重写的。要把卷积整进去。
        with paddle.no_grad():
            # print(self.noisy.sample([1]).shape)
            # action = self.forward(state).squeeze(0) + self.is_train * epsilon * self.noisy.sample([1]).squeeze(0) # 这个是示例代码，但是应该不对
            action = self.forward(state).squeeze(0) + self.is_train * epsilon * self.noisy.sample([1])
        return 2 * paddle.clip(action, -1, 1).numpy() 
    
    def save_network(self,adam,checkpoint,**kargs):
        if 'location' in kargs:
            location = kargs['location'] + r'\saved_actor_networks'
        else:
            location = self.location        
        # https://www.paddlepaddle.org.cn/documentation/docs/zh/guides/beginner/model_save_load_cn.html
        # #moxingbaocunyujiazai
        canshu_name = location + r'\linear_net.pdparams'
        adam_name = location + r'\adam.pdopt'
        checkpoint_name = location + r'\final_checkpoint.pkl'
        paddle.save(self.state_dict(), canshu_name)
        paddle.save(adam.state_dict(), adam_name)
        paddle.save(checkpoint, checkpoint_name)

    
    def load_network(self,**kargs):
        if 'location' in kargs:
            location = kargs['location'] + r'\saved_actor_networks'
        else:
            location = self.location
        canshu_name = location + r'\linear_net.pdparams'
        adam_name = location + r'\adam.pdopt'
        checkpoint_name = location + r'\final_checkpoint.pkl' 
        try:           
            layer_state_dict = paddle.load(canshu_name)
            adam_state_dict = paddle.load(adam_name)
            checkpoint_dict = paddle.load(checkpoint_name)

            self.set_state_dict(layer_state_dict)
            return adam_state_dict , checkpoint_dict

            # print("Loaded Final Checkpoint. Epoch : {}, Loss : {}".format(final_checkpoint_dict["epoch"], final_checkpoint_dict["loss"].numpy()))
        except:
            print('actor: attension, something wrong when loading')
