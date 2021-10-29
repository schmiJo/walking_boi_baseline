#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random
import copy
from collections import namedtuple, deque
from networks import ActorNet, CriticNet
import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 100  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
LEARNING_RATE_ACTOR = 1.5e-4  # learning rate of the actor
LEARNING_RATE_CRITIC = 2.1e-4  # learning rate of the critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object. Who interacts with and learns from the environment

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        
        
        print('Device: '+ str(device))


        # actor initialization
        self.actor_local = ActorNet(state_size, action_size, random_seed).to(device)
        self.actor_target = ActorNet(state_size, action_size, random_seed).to(device)
        self.actor_optim = optim.Adam(self.actor_local.parameters(), lr=LEARNING_RATE_ACTOR)

        #  critic initialization
        self.critic_local = CriticNet(state_size, action_size, random_seed).to(device)
        self.critic_target = CriticNet(state_size, action_size, random_seed).to(device)
        self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=LEARNING_RATE_CRITIC, weight_decay=0)

        # The weights of the target networks are updated with the weights of the local networks to start with the same shared state
        self.hard_update(self.actor_local, self.actor_target)
        self.hard_update(self.critic_local, self.critic_target)

        # Initialize the ReplayBuffe
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def step(self, state: np.ndarray, action: np.ndarray, reward: np.ndarray, next_state: np.ndarray,
             done: np.ndarray) -> None:
        """The step taken in the environment is recorded and the learning process started"""
        self.memory.add(state, action, reward, next_state, done)

        # If the amount of stored experiences exeeds the batch size -> learn from the replay buffer
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state: np.ndarray, add_noise: bool) -> np.ndarray:
        """The actor returns actions for given state as per current policy.
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state: torch.Tensor = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            # Aquire an action by passing the current state to the local actor network
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # Add noise to the action
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma) -> None:
        """Learn from the experiences"""

        states, actions, rewards, next_states, dones = experiences

        # --- train the critic ---
        next_actions = self.actor_target(next_states)
        # Get expected Q values from local critic by passing in both the states and the actions
        Q_expected = self.critic_local.forward(states, actions)
        # Get next expected Q values from local critic by passing in both the next_states and the next_actions
        next_Q_targets = self.critic_target(next_states, next_actions)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * next_Q_targets * (1 - dones))
        # Caclulate the loss function using the expected return and the target
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optim.step()

        # --- train the actor ---
        # create the action predictions by passing the states to the local network
        actions_prediction = self.actor_local.forward(states)
        # calculate the loss function of the actor
        actor_loss = -self.critic_local(states, actions_prediction).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # soft update the target networks
        self.soft_updatee(self.critic_local, self.critic_target, TAU)
        self.soft_updatee(self.actor_local, self.actor_target, TAU)

    def soft_updatee(self, local_model, target_model, tau):
        """Soft update the target network from the weights and biases of the local network with the factor tau"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, local_model, target_model):
        """Copy the weights and biases from the local to the target network"""
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(param.data)

    def save_weights(self, path='./weights/') -> None:
        """Saves the weights of the local networks
        (both the agent and the critic)"""
        torch.save(self.critic_local.state_dict(), path + 'critic')
        torch.save(self.actor_local.state_dict(), path + 'actor')

    def restore_weights(self, path='./weights/') -> None:
        """Restore the saved local network weights to both the target and the local network"""

        self.critic_local.load_state_dict(torch.load(path + 'critic'))
        self.critic_local.eval()

        self.critic_target.load_state_dict(torch.load(path + 'critic'))
        self.critic_target.eval()

        self.actor_local.load_state_dict(torch.load(path + 'actor'))
        self.actor_local.eval()

        self.actor_target.load_state_dict(torch.load(path + 'actor'))
        self.actor_target.eval()
        
        print('weights restored')

    def reset_noise(self):
        self.noise.reset()


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a experience to the memory deque."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class OUNoise:
    """The Ornstein-Uhlenbeck process used to generate random noise.
    """

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters"""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state
