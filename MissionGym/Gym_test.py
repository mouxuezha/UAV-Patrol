import gym
import numpy as np
import sys
import os

WEIZHI =r'E:/EnglishMulu/UAV-Patrol'
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

env_name = 'UAV_Patrol_env-v0'
shishi_env = gym.make(env_name)
state = shishi_env.reset() 
action = 0 
next_state,reward,done,_ = shishi_env.step(action)
shishi_env.render()

