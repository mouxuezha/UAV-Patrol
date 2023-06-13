import gym
import numpy as np
from gym import error, spaces
from gym.spaces import Dict, Box, Discrete
from transfer import transfer
import sys
import os

WEIZHI =r'E:/EnglishMulu/UAV-Patrol'
sys.path.append(WEIZHI+r'/Support')
sys.path.append(WEIZHI+r'/UAV')
sys.path.append(WEIZHI+r'/BattleField')
from huatu_support import huatu_support
from zuobiaoxi import zuobiaoxi
from zuobiaoxi import zuobiao
from UAV import UAV
from BattleField import BattleField

class UAVPatrolEnv(gym.Env):
    metadata = {'render.modes': ['human']}  
    # it seems only a simple static member of Env class
    # https://gymnasium.farama.org/

    def __init__(self):
        print('MXairfoil: start UAVPatrolEnv initialization. En Taro XXH!')
        self.viewer=None
        self.server_process=None
        self.server_port = None
        # self.BattleField_init(x_0=x_0,y_0=y_0,sudu_0 = sudu_0,a_0=a_0,omega_0=omega_0,dt = dt,omega_max = omega_max,sudu_max=sudu_max,r_observation=r_observation,L_x=L_x,L_y=L_y,dL=dL,S_target=S_target)
        self.BattleField_init(x_0=0,y_0=50*10**3,sudu_0 = 40,a_0=0,omega_0=0,dt = 60,omega_max = 0.2,sudu_max=50,r_observation=3000,L_x=100*10**3,L_y=100*10**3,dL=1*10**3,S_target=4*10**8)
        

        self.hangshu =  int(self.BattleField.L_x/self.BattleField.dL) + 1
        self.observation_space = Dict({"location": Box(0 , self.BattleField.L_x, shape=(2,), dtype=np.float64),"direction": Box(0 , 1, shape=(2,), dtype=np.float64), "evaluate_array": Box(0 , self.BattleField.L_x, shape=(self.hangshu,self.hangshu), dtype=int) })

        self.action_space = spaces.Box(low=-1.0*self.BattleField.UAV_feiji.omega_v_max, high= self.BattleField.UAV_feiji.omega_v_max, shape=(1, ), dtype=np.float64)
        # observation_space.sample()
        self.other_init() 

    def BattleField_init(self,x_0=0,y_0=50*10**3,sudu_0 = 40,a_0=0,omega_0=0,dt = 60,omega_max = 0.2,sudu_max=50,r_observation=3000,L_x=100*10**3,L_y=100*10**3,dL=1*10**3,S_target=4*10**8,**kargs):
        # IS. m, s, rad
        self.feiji = UAV(location=np.array([x_0,y_0]), sudu_max=sudu_max, omega_max = omega_max,r=r_observation)
        self.feiji.set_chuzhi(sudu_0=np.array([sudu_0,0]),a_0=a_0,omega_0=omega_0,dt = dt)
        self.BattleField = BattleField(L_x=L_x,L_y=L_y)
        self.BattleField.UAV_online(UAV_feiji = self.feiji)
        self.BattleField.generate_patrol_area2(S_target=S_target,L_fanwei_min = 0.1*L_x,L_fanwei_max = 0.9*L_x)
        print('UAVPatrolEnv: BattleField initialized. En Taro XXH!')
        return 

    def other_init(self):
        # locations and others.
        self.render_location = r"E:\EnglishMulu\UAV-Patrol\MissionGym" 
    
    def state_init(self):
        # init the state.
        location = np.zeros(2)
        direction = np.zeros(2)
        evaluate_array = np.zeros((self.hangshu,self.hangshu), dtype=int)
        state = {"location": location,
                 "direction": direction,
                 "evaluate_array": evaluate_array}
        raise Exception('UAVPatrolEnv.render waiting construction')


    def step(self,x):
        self.N_step=self.N_step+1
        x = np.array(x)
        x = x.reshape(self.real_dim,)
        raise Exception('UAVPatrolEnv.render waiting construction')

        notdone = np.isfinite(self.state).all() & notdone

        done = not notdone

        return self.state, self.reward, done, {}        
    
    def reset(self):
        self.reward = 0.0
        self.state = 1145141919810
        return self.state
    
    def render(self):
        # implement some huatu here.
        raise Exception('UAVPatrolEnv.render waiting construction')

    def pre_step(self,flag_array):
        # this is to process the arrays about the situational awareness.
        # the paddle is must here.
        raise Exception('UAVPatrolEnv.pre_step waiting construction')