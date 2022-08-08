from gym.envs.registration import register 

register(id='basic-v0',entry_point='gym_basic.envs:BasicEnv',) 
# register(id='basic-v2',entry_point='gym_basic.envs:BasicEnv2',)
register(id='contGrid-v0',entry_point='gym_basic.envs:ContGridWorld',) 
register(id='contGrid-v2',entry_point='gym_basic.envs:ContGridWorld_v2',)
register(id='contGrid-v3',entry_point='gym_basic.envs:ContGridWorld_v3',)
register(id='contGrid-v4',entry_point='gym_basic.envs:ContGridWorld_v4',)
register(id='contGrid-v5',entry_point='gym_basic.envs:ContGridWorld_v5',)
