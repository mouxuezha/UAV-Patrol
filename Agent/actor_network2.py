# this is actor network using paddle and dynamic graph. and CNN
import paddle
import paddle.nn as nn
from paddle.distribution import Normal
import numpy as np
import paddle.nn.functional as F
from paddle.nn import Conv2D, MaxPool2D

class ActorNetwork(nn.Layer):
    def __init__(self,state_dim=[2,2,1,101],action_dim=1 , env_switch =0,state_name=[],**kargs):
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
        self.state_name = state_name
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

    def creat_CNN(self, H=101, W=101):

        in_channels = 1
        out_channels = 5
        # 定义卷积层，
        conv1 = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        H,W = self.Conv2D_parameter_change(H,W, paddings=2,dilations=1,kernel_size=5,strides=1)
        # 定义池化层，
        max_pool1 = MaxPool2D(kernel_size=4, stride=4)
        H,W = self.MaxPool2D_parameter_change(H, W, paddings=0, kernel_size=4, strides=4)
        # 定义卷积层，
        conv2 = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=5, stride=1, padding=2)
        H,W = self.Conv2D_parameter_change(H, W, paddings=2,dilations=1,kernel_size=5,strides=1)
        # 定义池化层，
        max_pool2 = MaxPool2D(kernel_size=4, stride=4)
        H,W = self.MaxPool2D_parameter_change(H, W, paddings=0, kernel_size=4, strides=4)
        # 定义一层全连接层，输出维度是1【我这个不能是1,但是5也是随手给的】
        fc = nn.Linear(in_features=H*W*out_channels, out_features=self.CNN_out_dim)
        
        self.layers_CNN.append(conv1)
        self.layers_CNN.append(max_pool1)
        self.layers_CNN.append(conv2)
        self.layers_CNN.append(max_pool2)
        self.layers_CNN.append(fc)
    
    def Conv2D_parameter_change(self,H_in,W_in, paddings=2,dilations=1,kernel_size=5,strides=1):
        # CNN和max_pool都是这个公式，不重新写了。
        H_out = (H_in+2*paddings-(dilations*(kernel_size-1)+1))/strides+1
        W_out = (W_in+2*paddings-(dilations*(kernel_size-1)+1))/strides+1
        H_out = int(H_out)
        W_out = int(W_out)
        return H_out, W_out
    def MaxPool2D_parameter_change(self,H_in,W_in, paddings=2,dilations=1,kernel_size=5,strides=1):
        # 不行还是得重新写
        m = 0 # max(kernel_size-1,0)
        n = 0 # max(kernel_size-1,0)
        H_out = (H_in - m)/strides
        W_out = (W_in - n)/strides
        H_out = int(H_out)
        W_out = int(W_out)
        return H_out, W_out
    
    def forward(self,x):
        x_CNN = x["CNN"]   
        
        for i in range(len(self.layers_CNN)):
            
            if i == len(self.layers_CNN)-1:
                x_CNN = paddle.reshape(x_CNN, [x_CNN.shape[0], -1])
                x_CNN = self.layers_CNN[i](x_CNN)
            else:
                x_CNN = self.relu(self.layers_CNN[i](x_CNN))
        
        # x_CNN = x_CNN.unsqueeze(0)

        x_other = x["other"]
        x_other = x_other.squeeze(1)
        for i in range(len(self.layers_linear)):
            if i == 0 :
                x = paddle.concat((x_other, x_CNN), axis=1)
            x = self.relu(self.layers_linear[i](x))
            # if i == 0 :
            #    x = paddle.concat((x, a), axis=1)
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
        elif type(state) == dict:
            state = self.state_to_tensor(state) # this is for CNN

        with paddle.no_grad():
            # print(self.noisy.sample([1]).shape)
            # action = self.forward(state).squeeze(0) + self.is_train * epsilon * self.noisy.sample([1]).squeeze(0) # 这个是示例代码，但是应该不对
            action = self.forward(state).squeeze(0) + self.is_train * epsilon * self.noisy.sample([1])
        return 2 * paddle.clip(action, -1, 1).numpy() 
    
    def state_to_tensor(self,state):
        # 把state转化成能够输入到神经网络里面去的形式。
        state_tensor = {} 
        x = {} 
        # x_CNN = np.append(state['location'],state['direction'])
        # X_CNN = np.append(x_CNN,state['omega'])
        # state_tensor['location']= paddle.to_tensor(state['location'],dtype="float32").unsqueeze(0)
        # state_tensor['direction'] = paddle.to_tensor(state['direction'],dtype="float32").unsqueeze(0)
        # state_tensor['omega'] = paddle.to_tensor(state['omega'],dtype="float32").unsqueeze(0)
        # state_tensor['evaluate_array'] = paddle.to_tensor(state['evaluate_array'],dtype="float32").unsqueeze(0)
        for name in self.state_name:
            if type(state[name])==np.ndarray:
                if len(state[name].shape) == 2 and name!='evaluate_array':
                    state[name] = state[name].reshape(state[name].shape[1],)
                
            state_tensor[name] = paddle.to_tensor(state[name],dtype="float32").unsqueeze(0).unsqueeze(0)
        # x_CNN = state['evaluate_array']

        x_CNN = state_tensor['evaluate_array']
        x_other = paddle.concat((state_tensor['location'],state_tensor['direction'],state_tensor['omega']),axis=2)
        x['CNN'] = x_CNN 
        x['other'] = x_other
        return x

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
