import math
import random

from torch._C import device
from torch.serialization import save
import walking_boi 
from collections import deque

import time


import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from ppo_agent import PPO


HORIZON_T = 128
MINIBATCH_SIZE = 32
GAMMA = 0.99
GAE_PARAM = 0.95
CLIP_PARAM = 0.1 # Also called epsilon in the paper (or eps_clip in the following)
UPDATE_AFTER_N_STEPS = 1000

K_EPOCHS = 5

ACTION_STD_INIT = 0.4
ACTION_STD_DECAY = 0.05
ACTION_STD_MIN = 0.05
ACTION_STD_DECAY_FREQ = 3e5 # Every n steps decrement the standard deviation of the action by the decay rate (until min is reached)

LR_ACTOR =  0.0002
LR_CRITIC = 0.001 



def test():
    
    print('====================================================')
    
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
    print('Device is Cuda' if torch.cuda.is_available() else 'Device is CPU')
     

    env_name = "Humanoid-v2"
 
    env = gym.make(env_name)

    print('Observation Space: ', env.observation_space)
    print('Action Space:', env.action_space)
    
    print(f'Training Env: {env_name}')
    
    print('====================================================')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    
    
    # initialize the PPO Agent 
    ppo_agent = PPO(state_size, action_size, LR_ACTOR, LR_CRITIC, K_EPOCHS , GAMMA, CLIP_PARAM, ACTION_STD_INIT)
    
    episodic_reward = 0;
    reward_window = deque(maxlen=1000)
    i_episode = 0
    i_step_total = 0
    
    ppo_agent.restore_weights()
    
    while i_episode < 5:
        i_episode += 1
        state = env.reset()
        episodic_reward = 0
        
        for i_step_episode in range(100000):
            # select an action using the policy
            #env.render()
            action, _ = ppo_agent.select_action(state)
            i_step_total += 1
            
            time.sleep(0.05)
            
            state, reward, done, info = env.step(action.cpu().numpy().flatten())
            
            
            episodic_reward += reward
            
                
            if done:
                break;
        
        reward_window.append(episodic_reward)
        print('\rEpisode {} Step {} \tavg Score: {:.2f}'.format(i_episode,  i_step_total,np.mean(reward_window)), end="")
      
        
    env.close()



if __name__ == '__main__':
    test()