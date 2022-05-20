"""
Source: https://arxiv.org/pdf/1511.05952.pdf
"""
import numpy as np
import random

from QNet import QNet
from ReplayBuffer import ReplayBuffer

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import operator
from Config import Config
from enums import Action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, config: Config, action_size, seed, compute_weights = False) -> None:
        pass

    def step(self, state, action, reward, next_state, done):
        pass

    def act(self, state, eps=0.):
        pass

    def learn(self, sample, gamma):
        pass

    def update(self, local_model, target_model, tau):
        pass



