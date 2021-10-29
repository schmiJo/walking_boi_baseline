import gym 
import walking_boi 
env = gym.make("WalkingBoi-v0") 

observation = env.reset()
for _ in range(100000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action) 
  if done:
    observation = env.reset()
env.close()