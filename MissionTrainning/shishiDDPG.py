# 不要飘，还是先把之前那个兼容了，不要搞大跃进空中楼阁
import gym
import sys
WEIZHI =r'E:/EnglishMulu/UAV-Patrol' 
sys.path.append(WEIZHI+r'/Agent')
sys.path.append(WEIZHI+r'/Support')
sys.path.append(WEIZHI+r'/UAV')
sys.path.append(WEIZHI+r'/BattleField')
sys.path.append(WEIZHI+r'/MissionGym/UAV_Patrol/envs')

# from DDPG1 import DDPG # 这个是原版的
from DDPG2 import DDPG # 这个是加了CNN的
from UAV import UAV
from BattleField import BattleField
if __name__ == "__main__":
    # 定义环境
    # env = gym.make('Pendulum-v1')
    env = gym.make('UAV_Patrol_env-v0')
    print(env.env.env.env.spec.id)

    # 定义智能体了。
    agent0 = DDPG(env)

    # 测试一下读写
    # agent0.save_agent()

    # agent0.load_agent()


    # 然后走一波训练，这个就对应原来的训练循环。
    agent0.trainning()

