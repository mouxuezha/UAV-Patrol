# 不要飘，还是先把之前那个兼容了，不要搞大跃进空中楼阁
import gym
import sys
WEIZHI =r'E:/EnglishMulu/UAV-Patrol' 
sys.path.append(WEIZHI+r'/Agent')
from DDPG1 import DDPG 

if __name__ == "__main__":
    # 定义环境
    env = gym.make('Pendulum-v1')
    print(env.env.env.env.spec.id)

    # 定义智能体了。
    agent0 = DDPG(env)

    # 测试一下读写
    # agent0.save_agent()

    # agent0.load_agent()


    # 然后走一波训练，这个就对应原来的训练循环。
    agent0.trainning()

