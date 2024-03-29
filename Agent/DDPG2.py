import gym
import numpy as np


import os
import pickle
import sys
WEIZHI =r'E:\EnglishMulu\UAV-Patrol' 
sys.path.append(WEIZHI+r'/Support')
from critic_network2 import CriticNetwork
from actor_network2 import ActorNetwork
from replay_buffer import ReplayBuffer
import paddle
from visualdl import LogWriter
import paddle.nn.functional as F

class DDPG:
    """docstring for DDPG, En Taro XXH"""
    def __init__(self,env,**kargs) -> None:
        if 'REPLAY_START_SIZE' in kargs:
            self.REPLAY_START_SIZE = kargs['REPLAY_START_SIZE']
        else:
            self.REPLAY_START_SIZE = 50000

        if 'actor_chicun' in kargs:
            self.actor_chicun = kargs['actor_chicun']
        else:
            self.actor_chicun = [200,300,200]
        if 'critic_chicun' in kargs:
            self.actor_chicun = kargs['actor_chicun']
        else:
            self.actor_chicun = [200,300,200]            
        if 'LEARNING_RATE_a' in kargs:
            self.LEARNING_RATE_a = kargs['LEARNING_RATE_a']
        else:
            self.LEARNING_RATE_a = 1e-5 
        if 'LEARNING_RATE_c' in kargs:
            self.LEARNING_RATE_c = kargs['LEARNING_RATE_c']
        else:
            self.LEARNING_RATE_c = 3e-5
        if 'TAU' in kargs:
            self.TAU = kargs['TAU']
        else:
            self.TAU = 0.001
        if 'L2' in kargs:
            self.L2 = kargs['L2']
        else:
            self.L2 = 0.01 
        
        # self.BATCH_SIZE = 32 # 64 
        self.GAMMA = 0.99
        self.env = env
        # self.state_name = self.env.state_name
        self.state_name = ["location", "direction", "omega","evaluate_array"]
        self.step_max = 200 
        self.epsilon = 1 # this is to control the noise.
        self.explore = 50000

        self.final_checkpoint = dict()
        
        if self.__env_switch()==0:
            self.__init_Pendulum()
        elif self.__env_switch() ==1:
            self.__init_UAV()
        

        self.actor_network = ActorNetwork(self.state_dim,self.action_dim,self.__env_switch(),state_name=self.state_name)
        self.critic_network = CriticNetwork(self.state_dim,self.action_dim,self.__env_switch(),state_name=self.state_name)

        self.target_actor_network = ActorNetwork(self.state_dim,self.action_dim,self.__env_switch(),state_name=self.state_name)
        self.target_critic_network = CriticNetwork(self.state_dim,self.action_dim,self.__env_switch(),state_name=self.state_name)

        # 定义优化器
        self.critic_optim = paddle.optimizer.Adam(parameters=self.critic_network.parameters(), learning_rate=self.LEARNING_RATE_a)
        self.actor_optim = paddle.optimizer.Adam(parameters=self.actor_network.parameters(), learning_rate=self.LEARNING_RATE_c)

        self.memory_replay = ReplayBuffer(50000)
        self.begin_train = False
        self.batch_size = 32
        self.learn_steps = 0
        self.epochs = 2500

        self.writer = LogWriter('logs')

    def __env_switch(self):
        # 'Pendulum-v1' is case 0, UAV is case 1
        try:
            name = self.env.env.env.env.spec.id
        except:
            name = self.env.env.env.spec.id
        if name == 'Pendulum-v1':
            return 0 
            self.__init_Pendulum()
        elif name == 'UAV_Patrol_env-v0':
            return 1 
        else:
            raise Exception('DDPG: Invalid env case, G!')     

    def __init_Pendulum(self):
        self.state_dim = 3 
        self.action_dim = 1 
    
    def __init_UAV(self):
        self.state_dim = [2,2,1,101] # 101不好卷积貌似
        # self.state_dim = [2,2,1,100]
        self.action_dim = 1

    def set_location(self,location=r'E:\EnglishMulu\agents',index=0):
        while os.path.exists(location + r'\agent'+str(index)):
            index = index + 1 
        self.location = location + r'\agent'+str(index)
        return self.location

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.set_value( target_param * (1.0 - tau) + param * tau) 

    def unpackage(self,batch):
        batch_state = [] 
        batch_next_state= [] 
        batch_action=[] 
        batch_reward=[]
        for i in range(self.batch_size):
            batch_state.append(batch[i][0])
            batch_action.append(batch[i][1])
            batch_reward.append(batch[i][2])
            if type(batch[i][3]) == dict:
                batch_next_state.append(batch[i][3])
            else:
                batch_next_state.append(batch[i][3].reshape((3,)))
        
        return batch_state, batch_next_state, batch_action, batch_reward

    def state_to_tensor_batch(self,states):
        # 把state转化成能够输入到神经网络里面去的形式。
        state_tensor = {} 
        x_batch = {} 
        for i in range(len(states)):
            state = states[i]
            for name in self.state_name:
                if type(state[name])==np.ndarray:
                    if len(state[name].shape) == 2 and name!='evaluate_array':
                        state[name] = state[name].reshape(state[name].shape[1],)
                    
                state_tensor[name] = paddle.to_tensor(state[name],dtype="float32").unsqueeze(0).unsqueeze(0)

            x_CNN_single = state_tensor['evaluate_array']
            x_other_single = paddle.concat((state_tensor['location'],state_tensor['direction'],state_tensor['omega']),axis=2)

            if i == 0 :
                x_CNN = x_CNN_single
                x_other = x_other_single
            else:
                x_CNN = paddle.concat((x_CNN,x_CNN_single),axis=0)
                x_other = paddle.concat((x_other,x_other_single),axis=0)
        x_batch['CNN'] = x_CNN 
        x_batch['other'] = x_other
        return x_batch
    
    def trainning(self):
        for epoch in range(0, self.epochs):
            state = self.env.reset()
            # state = state[0] # Pendulum-v1直接给出的话后面会有一个奇怪的{}
            episode_reward = 0
            for time_step in range(self.step_max):
                action = self.actor_network.select_action(self.epsilon, state)
                # next_state, reward, done, _ ,_= self.env.step([action])
                next_state, reward, done, _= self.env.step(action)
                episode_reward += reward
                # self.memory_replay.add((state, next_state, action, reward)) # 这个又是，给的代码调不通。那只能自己想着点 改改了。而且顺序也不对
                self.memory_replay.add(state, action, reward,next_state, done) 

                if self.memory_replay.size() > 1280: #1280
                    self.learn_steps += 1
                    if not self.begin_train:
                        print('train begin!')
                        self.begin_train = True
                    experiences = self.memory_replay.sample(self.batch_size, False)
                    # experiences = self.memory_replay.get_batch(self.batch_size, False) 
                    batch_state, batch_next_state, batch_action, batch_reward = self.unpackage(experiences)
                    # 原版写了个zip(*experience)在这里，运行下来是不对的。手动改了个unpackge

                    # 在的state是个抽象数据类型了，得重新整整。
                    batch_action = paddle.to_tensor(batch_action,dtype="float32")
                    batch_reward = paddle.to_tensor(batch_reward,dtype="float32").unsqueeze(1)
                    x_batch_state = self.state_to_tensor_batch(batch_state)
                    x_batch_next_state = self.state_to_tensor_batch(batch_next_state)
                  
                    # 均方误差 y - Q(s, a) ， y是目标网络所看到的预期收益， 而 Q(s, a)是Critic网络预测的操作值。
                    # y是一个移动的目标，评论者模型试图实现的目标；这个目标通过缓慢的更新目标模型来保持稳定。 
                    with paddle.no_grad():
                        Q_next = self.target_critic_network(x_batch_next_state, self.target_actor_network(x_batch_next_state))
                        Q_target = batch_reward + self.GAMMA * Q_next

                    critic_loss = F.mse_loss(self.critic_network(x_batch_state, batch_action), Q_target)

                    self.critic_optim.clear_grad() # 这一段和静态图里面的说法有点不一样了
                    critic_loss.backward()
                    self.critic_optim.step()

                    self.writer.add_scalar('critic loss', critic_loss.numpy(), self.learn_steps)
                    # 使用Critic网络给定值的平均值来评价Actor网络采取的行动。 我们力求使这一数值最大化。 
                    # 因此，我们更新了Actor网络，对于一个给定状态，它产生的动作尽量让Critic网络给出高的评分。 
                    self.critic_network.eval()
                    actor_loss = - self.critic_network(x_batch_state, self.actor_network(x_batch_state))
                    # print(actor_loss.shape)
                    actor_loss = actor_loss.mean()
                    self.actor_optim.clear_grad()
                    actor_loss.backward()
                    self.actor_optim.step()
                    self.critic_network.train()
                    self.writer.add_scalar('actor loss', actor_loss.numpy(), self.learn_steps)

                    self.soft_update(self.target_actor_network, self.actor_network, self.TAU)  
                    self.soft_update(self.target_critic_network, self.critic_network, self.TAU)                  
                if self.epsilon > 0:
                    self.epsilon -= 1 / self.explore
                state = next_state
            self.writer.add_scalar('episode reward', episode_reward, epoch)
            if epoch % 50 == 0:
                print('Epoch:{}, episode reward is {}'.format(epoch, episode_reward))
                if epoch>=100:
                    self.final_checkpoint["epoch"] = epoch
                    self.final_checkpoint["actor_loss"] = actor_loss
                    self.final_checkpoint["critic_loss"] = critic_loss        
                
            if epoch % 200 == 0:
                # paddle.save(self.actor_network.state_dict(), 'model/ddpg-actor' + str(epoch) + '.para')
                # paddle.save(self.critic_network.state_dict(), 'model/ddpg-critic' + str(epoch) + '.para')
                # self.actor_network.save_network()
                # self.critic_network.save_network()
                # print('model saved!') 
                self.save_agent(index= epoch)
        
        print('XXH: train finished!')
        return 0   

    def save_agent(self,index=0):
        self.location = self.set_location(index=index)
        try:
            os.makedir(self.location)
        except: 
            print('DDPG: agent file exists')
        self.actor_network.save_network(self.actor_optim,self.final_checkpoint,location = self.location)
        self.critic_network.save_network(self.critic_optim,self.final_checkpoint,location = self.location)

    def load_agent(self,**kargs):
        if 'location' in kargs :
            location = kargs['location']
        else:
            location = self.location
        actor_adam_state_dict , checkpoint_dict = self.actor_network.load_network(location = location)
        critic_adam_state_dict , checkpoint_dict = self.critic_network.load_network(location = location)

        self.actor_optim.set_state_dict(actor_adam_state_dict)
        self.critic_optim.set_state_dict(critic_adam_state_dict)
        self.final_checkpoint = checkpoint_dict

        print('DDPG: agent loaded successfully')