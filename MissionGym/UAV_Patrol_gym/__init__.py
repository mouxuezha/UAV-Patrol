from gym.envs.registration import register

register(
    id='UAV_Patrol_env-v0',                                   # Format should be xxx-v0, xxx-v1....
    entry_point='UAV_Patrol_gym.envs:UAVPatrolEnv',              # Expalined in envs/__init__.py
)
register(
    id='UAV_Patrol_env_extend-v0',
    entry_point='UAV_Patrol_gym.envs:ShishiEnvExtend',
)