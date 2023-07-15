import gym
import numpy as np


import os
import pickle
import sys
WEIZHI =r'E:/EnglishMulu/UAV-Patrol' 
sys.path.append(WEIZHI+r'/Support')
from critic_network import CriticNetwork
from actor_network import ActorNetwork
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
        
        self.BATCH_SIZE = 32 # 64 
        self.GAMMA = 0.99
        self.environment = env
        self.step_max = 200 
        self.epsilon = 1 # this is to control the noise.

        self.final_checkpoint = dict()
        
        if self.__env_switch()==0:
            self.__init_Pendulum()
        

        self.actor_network = ActorNetwork(self.state_dim,self.action_dim,self.__env_switch())
        self.critic_network = CriticNetwork(self.state_dim,self.action_dim,self.__env_switch())

        self.target_actor_network = ActorNetwork(self.state_dim,self.action_dim,self.__env_switch())
        self.target_critic_network = CriticNetwork(self.state_dim,self.action_dim,self.__env_switch())

        # 定义优化器
        self.critic_optim = paddle.optimizer.Adam(parameters=self.critic_network.parameters(), learning_rate=self.LEARNING_RATE_a)
        self.actor_optim = paddle.optimizer.Adam(parameters=self.actor_network.parameters(), learning_rate=self.LEARNING_RATE_c)

        self.memory_replay = ReplayBuffer(50000)
        begin_train = False
        batch_size = 32
        self.learn_steps = 0
        self.epochs = 250

        self.writer = LogWriter('logs')

    def __env_switch(self):
        # 'Pendulum-v1' is case 0 
        if self.environment.env.env.env.spec.id == 'Pendulum-v1':
            return 0 
            self.__init_Pendulum()   
        else:
            raise Exception('DDPG: Invalid env case, G!')     

    def __init_Pendulum(self):
        self.state_dim = 3 
        self.action_dim = 1 

    def set_location(self,location=r'E:\EnglishMulu\agents',index=0):
        while os.path.exists(location + r'\agent'+str(index)):
            index = index + 1 
        self.location = location + r'\agent'+str(index)
        return self.location

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.set_value( target_param * (1.0 - tau) + param * tau) 

    def trainning(self):
        for epoch in range(0, self.epochs):
            state = self.env.reset()
            episode_reward = 0
            for time_step in range(self.step_max):
                action = self.actor_network.select_action(self.epsilon, state)
                next_state, reward, done, _ = self.env.step([action])
                episode_reward += reward
                reward = (reward + 8.1) / 8.1 #这尼玛？这么暴力的操作，就直接写在代码里
                self.memory_replay.add((state, next_state, action, reward))

                if self.memory_replay.size() > 1280:
                    learn_steps += 1
                    if not begin_train:
                        print('train begin!')
                        begin_train = True
                    experiences = self.memory_replay.sample(self.batch_size, False)
                    batch_state, batch_next_state, batch_action, batch_reward = zip(*experiences)

                    batch_state = paddle.to_tensor(batch_state,dtype="float32")
                    batch_next_state = paddle.to_tensor(batch_next_state,dtype="float32")
                    batch_action = paddle.to_tensor(batch_action,dtype="float32").unsqueeze(1)
                    batch_reward = paddle.to_tensor(batch_reward,dtype="float32").unsqueeze(1)
                    # 均方误差 y - Q(s, a) ， y是目标网络所看到的预期收益， 而 Q(s, a)是Critic网络预测的操作值。
                    # y是一个移动的目标，评论者模型试图实现的目标；这个目标通过缓慢的更新目标模型来保持稳定。 
                    with paddle.no_grad():
                        Q_next = self.target_critic_network(batch_next_state, self.target_actor_network(batch_next_state))
                        Q_target = batch_reward + self.GAMMA * Q_next

                    critic_loss = F.mse_loss(self.critic_network(batch_state, batch_action), Q_target)

                    self.critic_optim.clear_grad() # 这一段和静态图里面的说法有点不一样了
                    critic_loss.backward()
                    self.critic_optim.step()

                    self.writer.add_scalar('critic loss', critic_loss.numpy(), learn_steps)
                    # 使用Critic网络给定值的平均值来评价Actor网络采取的行动。 我们力求使这一数值最大化。 
                    # 因此，我们更新了Actor网络，对于一个给定状态，它产生的动作尽量让Critic网络给出高的评分。 
                    self.critic_network.eval()
                    actor_loss = - self.critic_network(batch_state, self.actor_network(batch_state))
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