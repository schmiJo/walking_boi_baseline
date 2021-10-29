from typing import List
from collections import deque



class EspisodicMemory:
    def __init__(self):
        """
        everything recorded during one episode
        """
        self.rewards = []
        self.states = []
        self.new_states = []
        self.actions = []

    def append(self, state, action, new_state, reward):
        self.rewards.append(reward)
        self.states.append(state)
        self.new_states.append(new_state)
        self.actions.append(action)

    def get_episode(self):
        return self.states, self.actions, self.new_states, self.rewards
    
    
class ReplayBuffer:
    
    def __init__(self, BUFFER_SIZE = 1000) -> None:
        self.episodes = deque(maxlen= BUFFER_SIZE)
        self.BUFFER_SIZE = BUFFER_SIZE
        
        
    def append_episode(self, episode: EspisodicMemory):
        self.episodes
        
    
    def sample_episode():
        
        pass
        