import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Source:
https://arxiv.org/pdf/1511.05952.pdf
"""
class QNet(nn.Module):
    def __init__(self, state_size, action_size, seed) -> None:
        super(QNet, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
