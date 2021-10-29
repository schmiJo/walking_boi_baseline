from gym.envs.registration import register
register(
    id='WalkingBoi-v0', 
    entry_point='walking_boi.envs:WalkingBoiEnv'
)