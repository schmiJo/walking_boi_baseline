
import math
import random
import walking_boi 
  

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def main(num_episodes = 100):
    
    device = torch.device( 'cuda' if torch.cuda.is_available else 'cpu')
    

    env_name = "MountainCarContinuous-v0"
 
    env = gym.make(env_name)

    print('Observation Space: ', env.observation_space)
    print('action Space:', env.action_space)
    
    policy = Policy(state_size=5, action_size=5)

    state = env.reset()
    
    for i_episode in range(num_episodes):
        
        for i_action in range(10000):
            env.render()
        
            action = env.action_space.sample()
        
            next_state, reward, done, info = env.step(action)
        
            state = next_state
        
            if done: 
                state = env.reset()
                break

    

if __name__ == '__main__':
    main()