import gym

class ShishiEnvExtend(gym.Env):
    def __init__(self):
        print('ShishiEnvExtend Environment initialized')
    def step(self):
        print('ShishiEnvExtend Step successful!')
    def reset(self):
        print('ShishiEnvExtend Environment reset')