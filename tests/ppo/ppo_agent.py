import os 

import torch
import torch.nn as nn
import torch.nn.functional as F
from buffer import Buffer
from torch.distributions import MultivariateNormal


device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
    
class ActorCritic(nn.Module):
    
    def __init__(self, state_size, action_size, action_std_init , hidden_sizes = [50, 50]):
        super().__init__()
        self.action_size = action_size
        
        
        # The Variance of the action used in the multidim gaussian
        self.action_var = torch.full((action_size,), action_std_init * action_std_init).to(device)
        
        # Array of fully connected layers
        
        self.layers_actor = []
        self.layers_critic = []
        
        
        if len(hidden_sizes) == 0: 
            self.layers_actor.append(nn.Linear(state_size, action_size))
            self.layers_critic.append(nn.Linear(state_size, 1))
        else:
            self.layers_actor.append(nn.Linear(state_size, hidden_sizes[0]))
            self.layers_critic.append(nn.Linear(state_size, hidden_sizes[0]))
            
            for i , size in enumerate(hidden_sizes):
                if i == 0:
                    continue;
                # Add a relu actication function too both actor and criritc
                
                self.layers_actor.append(nn.ReLU())
                self.layers_critic.append(nn.ReLU())
                
                # Add the linear layers for each layer specified
                self.layers_actor.append(nn.Linear(hidden_sizes[i-1], size))
                self.layers_critic.append(nn.Linear(hidden_sizes[i-1], size))
                
            
            self.layers_actor.append(nn.ReLU())
            self.layers_critic.append(nn.ReLU())
            
            # The last layer that connects the previous layer with the action size
            self.layers_actor.append(nn.Linear(hidden_sizes[-1], action_size))
            self.layers_critic.append(nn.Linear(hidden_sizes[-1], 1))
        
        
        # Add a last tanh function to map to output space between -1 and 1    
        self.layers_actor.append(nn.Tanh())
        
        self.actor = nn.Sequential(*self.layers_actor)
        self.critic = nn.Sequential(*self.layers_critic)
        
            
        
        
        
    def set_action_std(self,action_std_new):
        self.action_var = torch.full((self.action_size,), action_std_new * action_std_new).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
         
        action_mean = self.actor(state)
        
        # Covarient Matrix for the multivarieate Normal distribution
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        
        # Multivariate gaussian matrix using the 
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.detach(), log_prob.detach()
    

    def eval(self, state, action):
        
      
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # For single Action Environments; 
        if self.action_size == 1:
            action = action.reshape(-1, self.action_size)
            
        log_probs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return log_probs, state_values, dist_entropy
    
class PPO:
    
    def __init__(self, state_size, action_size, lr_actor, lr_critic, K_epochs , gamma, eps_clip, action_std_init=0.5) -> None:
        
        self.action_std = action_std_init
        
        self.gamma = gamma 
        self.buffer = Buffer()
        self.eps_clip = eps_clip
        self.k_epochs = K_epochs
        
        
        self.policy = ActorCritic(state_size, action_size, action_std_init).to(device)
        
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])
        
        self.policy_old = ActorCritic(state_size, action_size, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)
        
    
    def select_action(self, state):
        
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
            
        return action.detach(), action_logprob.detach()
    
    def update(self):
        
        
        # Caclulate a Monte carlo estimate for the returns
        rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(self.buffer.rewards), reversed(self.buffer.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward )
            
            rewards.insert(0, discounted_reward)
            
        # Normalize the rewards
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # convert list to tensor
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
         
        
        for _ in range(self.k_epochs):
            
            # Eval old actions and values
            logprobs, state_values, dist_entropy = self.policy.eval(old_states, old_actions)
            
            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        
        self.buffer.clear()
        
    
    def save_weights(self, path='./weights/') -> None:
        """Saves the weights of the local networks
        (both the agent and the critic)"""
        torch.save(self.policy_old.state_dict(), path+ 'model.pth') 

    def restore_weights(self, path='./weights/') -> None:
        """Restore the saved local network weights to both the target and the local network"""
        self.policy_old.load_state_dict(torch.load(path + 'model.pth')) 
        self.policy.load_state_dict(torch.load(path + 'model.pth')) 
 
        print('weights restored')
        
    def add_to_buffer(self, action, state, logprob, reward, done):
        self.buffer.add(action, state, logprob, reward, done )

        
        
    
        
        
        
        
        
        
        
        
        
        

