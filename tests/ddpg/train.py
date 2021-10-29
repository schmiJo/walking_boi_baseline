import gym 
import walking_boi 
from collections import deque
from ddpg_agent import Agent


import numpy as np

import torch 
import torch.nn.functional as F
import torch.optim as optim


MAX_EPISODE_AMOUNT = 1000000
MAX_STEP_AMOUNT = 100000


def main():
    
    env = gym.make("MountainCarContinuous-v0") 
    

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    

    state = env.reset()
    reward_window = deque(maxlen=100)
    
    agent = Agent(action_size=env.action_space.shape[0], state_size=env.observation_space.shape[0], random_seed=0)
    
    #agent.restore_weights()
    
    for i_episode in range(100000):
        #env.render()
        
        cumulative_reward = 0
  
        for i_action in range(MAX_STEP_AMOUNT):
            
             
            action = agent.act(state=state, add_noise= i_episode < 200) # your agent here (this takes random actions)
            
            next_state, reward, done, info = env.step(action[0]) 
             
             
            
             
            cumulative_reward += reward
            
            #Take a step and learn from the s,a,r,s,d pair
            agent.step(state, action, reward, next_state, done)
            
            state = next_state
            
            if done:
                state = env.reset()
                break
         
            
        reward_window.append(cumulative_reward)
            
        print('\rEpisode {}\tavg Score: {:.2f}'.format(i_episode, np.mean(reward_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tavg Score: {:.2f}'.format(i_episode, np.mean(reward_window)))
            agent.save_weights()
      
      
    env.close()
    
    
if __name__ == "__main__":
    main()