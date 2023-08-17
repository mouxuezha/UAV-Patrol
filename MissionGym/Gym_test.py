import gym
import numpy as np
import sys
import os

WEIZHI =r'E:\EnglishMulu\UAV-Patrol' 
sys.path.append(WEIZHI+r'/Support')
sys.path.append(WEIZHI+r'/UAV')
sys.path.append(WEIZHI+r'/BattleField')
sys.path.append(WEIZHI+r'/MissionGym/UAV_Patrol/envs')

from huatu_support import huatu_support
from zuobiaoxi import zuobiaoxi
from zuobiaoxi import zuobiao
from UAV import UAV
from BattleField import BattleField

import UAV_Patrol_gym

import time 
import pickle

class debug_env(object):
    def __init__(self) :
        env_name = 'UAV_Patrol_env-v0'
        self.env = gym.make(env_name)
        self.work_location = self.env.render_location
        pass

    def standardization_performance(self,N_points = 114514):
        # calculate several steps to get perfromance_w and performance_b 
        performance_all = np.zeros((N_points,self.env.performance_dim))
        weizhi = self.work_location + '/performance_all.pkl'
        if os.path.exists(weizhi):
            return
        total_time_start = time.time()
        for i in range(N_points):
            # caonima
            performance_all[i] = self.get_random_performance(self.env)
            print('MXairfoil: ' + str(i) + ' points done.')
        
        total_time_end = time.time()
        total_time_cost = total_time_end - total_time_start
        print('MXairfoil: total time cost ='+str(total_time_cost) + ', for ' + str(N_points) + ' steps')     
        
        # self.X = pickle.load(open(location_X,'rb'))
        pickle.dump(performance_all,open(weizhi,'wb'))

    def get_random_performance(self,env):
        env.reset()
        env._has_reseted = True
        performance = env.get_performance()
        return performance

    def get_random_reward(self,env):
        env.reset()
        env._has_reseted = True
        state, reward, done, _ = env.step(np.zeros(env.real_dim))
        return reward 

    def calculate_weight_performance(self):
        # map the performance into [0,1]
        weizhi = self.work_location + '/performance_all.pkl'
        self.performance_all = pickle.load(open(weizhi,'rb'))

        # if self.performance_all:
        #     self.performance_all = pickle.load(open(weizhi,'rb'))
        performance_min = np.min(self.performance_all,axis=0)
        performance_max = np.max(self.performance_all,axis=0)
        performance_cha = performance_max - performance_min
        
        w = 1.0 / performance_cha
        b = 0 - w*performance_min
        performance_all_normal = w*self.performance_all + b
        check_1 = np.max(np.max(performance_all_normal))
        check_0 = np.min(np.min(performance_all_normal))
        if abs(check_1-1) + abs(check_0-0) < 0.00000001:
            print('MXairfoil: successfully get weights. \n w='+ str(w) +'\n b=' + str(b))

    def standardization_reward(self,N_points=114514,reuse = True):
        # calculate several steps to get parameters for r.
        # calculate several steps to get perfromance_w and performance_b 
        performance_all = np.zeros((N_points,1))
        weizhi = self.work_location + '/reward_all.pkl'
        if os.path.exists(weizhi) and reuse:
            return
        total_time_start = time.time()
        for i in range(N_points):
            # caonima
            performance_all[i] = self.get_random_reward(self.env)
            print('MXairfoil: ' + str(i) + ' points done.')
        
        total_time_end = time.time()
        total_time_cost = total_time_end - total_time_start
        print('MXairfoil: total time cost ='+str(total_time_cost) + ', for ' + str(N_points) + ' steps')     
        
        # self.X = pickle.load(open(location_X,'rb'))
        pickle.dump(performance_all,open(weizhi,'wb'))        

    def calculate_weight_reward(self):
        # map the performance into [0,1]
        weizhi = self.work_location + '/reward_all.pkl'
        self.reward_all = pickle.load(open(weizhi,'rb'))

        # performance_min = np.min(self.reward_all,axis=0)
        # performance_max = np.max(self.reward_all,axis=0)
        pingjun = np.mean(self.reward_all)
        performance_min = 0
        performance_max = 0 
        N_ave = 3
        for i in range(N_ave):
            # more point, to avoid pinnacle 
            index_min  = np.argmin(self.reward_all,axis=0)
            index_max  = np.argmax(self.reward_all,axis=0)
            performance_min = performance_min + self.reward_all[index_min]
            performance_max = performance_max + self.reward_all[index_max]
            self.reward_all[index_min] = pingjun
            self.reward_all[index_max] = pingjun 

        performance_min = performance_min / N_ave
        performance_max = performance_max / N_ave
        performance_cha = performance_max - performance_min
        
        w = 1.0 / performance_cha
        b = 0 - w*performance_min
        reward_all_normal = w*self.reward_all + b
        check_1 = np.max(np.max(reward_all_normal))
        check_0 = np.min(np.min(reward_all_normal))
        if abs(check_1-1) + abs(check_0-0) < 0.2:
            print('MXairfoil: successfully get weights. \n w='+ str(w) +'\n b=' + str(b))

    def test_time_comsumption(self,N=114514):
        # 
        state0 = self.env.reset()
        self.env._has_reseted = True
        total_time_start = time.time()
        for steps in range(N):
            if (steps % 100 == 9):
                print('MXairfoil: episode = '+str(steps))
            # state, reward, done, _ = self.env.step(np.zeros(self.env.real_dim))
            state, reward, done, _ = self.env.step(np.zeros(1))
        total_time_end = time.time()
        total_time_cost = total_time_end - total_time_start
        print('MXairfoil: total time cost in test_time_comsumption ='+str(total_time_cost)) 

if __name__ == '__main__':
    # env_name = 'UAV_Patrol_env-v0'
    # shishi_env = gym.make(env_name)
    # state = shishi_env.reset() 
    # action = 0 
    # next_state,reward,done,_ = shishi_env.step(action)
    # shishi_env.render()

    shishi = debug_env()
    # # shishi.standardization_performance(N_points=1000) 
    # # # shishi.get_random_performance()
    # # shishi.calculate_weight_performance()
    # shishi.standardization_reward(N_points=300,reuse=False) 
    # shishi.calculate_weight_reward()
    for i in range(10):
        shishi.test_time_comsumption(N=1000)

