import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class CriticNet(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_size=128, fc2_size=256, fc3_size=128):
        """Initialize the Neural Q network:
        This critic needs to map state action pairs to Q values
            Params:
                state_size -> size of the input layer
                action_size -> size of the output layer
                fc1 -> size of the first fully connected hidden layer
                fc2 -> size of the second fully connected hidden layer
            """
        super(CriticNet, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        self.seed = torch.manual_seed(seed)

        # The first layer only takes the states as input
        self.fc1 = nn.Linear(state_size, fc1_size)
        # The second layer takes the output of the first layer as well as the action as input
        self.fc2 = nn.Linear(fc1_size + action_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, 1)

    def forward(self, state, action):
        # add an extra dim for the batch normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x1 = F.relu(self.fc1(state))
        x = torch.cat((x1, action), dim=1)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class ActorNet(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_size=128, fc2_size=256, fc3_size=128):
        """Initialize the Actor ‚network:
            Params:
                state_size -> size of the input layer
                action_size -> size of the output layer
                fc1 -> size of the first fully connected hidden layer
                fc2 -> size of the second fully connected hidden layer
            """
        super(ActorNet, self).__init__()

        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, action_size)

        self.tanh = nn.Tanh()

    def forward(self, state):
        # add an extra dim for the batch normalization
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)

        x = F.relu(self.fc1(state))
        #x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # The tanh activation function restricts the output state between -1 and 1
        return self.tanh(self.fc4(x))
