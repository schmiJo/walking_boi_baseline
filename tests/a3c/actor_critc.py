import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_size , action_size, fc1_size = 128, fc2_size = 128, gru_size = 128):
        super(ActorCritic, self).__init__()
        
        
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        # A GRU is used to retain temporal information
        self.gru = nn.GRUCell(fc2_size, gru_size)
        #Pi is the action network of the agent (compare to paper a3c)
        self.pi = nn.Linear(gru_size, action_size)
        # v is the value function of the agent (compare to paper a3c)
        self.v = nn.Linear(gru_size, 1)
        
        
    def forward(self, state, hx):
        """
        hx: hidden state of the GRU cell
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        hx = self.gru(x, hx)
        
        action = self.pi(hx)
        state_value = self.v(hx)
        
        return action, state_value, hx
    
    
    def calc_R(self, done, rewards, values):
        values = T.cat(values.squeeze())
        
        # If the episode terminates make the value zero
        if len(values.size()) == 1:  # batch of states
            R = values[-1]*(1-int(done))
        elif len(values.size()) == 0:  # single state
            R = values*(1-int(done))

        batch_return = []
        
        # Go trough the rewards in reversed order and add the reward with discounted return
        for reward in rewards[::-1]:
            R = reward * self.gamma * R
            batch_return.append(R)
        
        batch_return.reverse()
        batch_return = T.tensor(batch_return,
                                dtype=T.float).reshape(values.size())
        return batch_return
        
        
        