class Buffer():
    
    def __init__(self) -> None:
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = [] 
        self.dones = [] # Determines whether this was the terminal action within the episode
        
        
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        
    def add(self, action, state, logprob, reward, done):
        self.actions.append(action)
        self.states.append(state)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.dones.append(done)
        
        
    