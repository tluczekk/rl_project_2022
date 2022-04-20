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

BUFFER_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 1e-3
LR = 5e-4
UPDATE_NN_EVERY = 1

UPDATE_MEM_EVERY = 20
UPDATE_MEM_PAR_EVERY = 3000
EXPERIENCES_PER_SAMPLING = math.ceil(BATCH_SIZE * UPDATE_MEM_EVERY / UPDATE_NN_EVERY)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent_DQN():

    def __init__(self, config: Config):
        # Hyperparameters
        self.config = config
        



    def update(self):
        """

        """

class Agent:
    def __init__(self, state_size, action_size, seed, compute_weights = False) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.compute_weights = compute_weights

        self.qnet_local = QNet(state_size, action_size, seed).to(device)
        self.qnet_target = QNet(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

        self.memory = ReplayBuffer(
            action_size, BUFFER_SIZE, BATCH_SIZE, EXPERIENCES_PER_SAMPLING, seed, compute_weights
        )
        self.time_step_nn = 0
        self.time_step_mem_par = 0
        self.time_step_mem = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        self.time_step_nn = (self.time_step_nn + 1) % UPDATE_NN_EVERY
        self.time_step_mem = (self.time_step_mem + 1) % UPDATE_MEM_EVERY
        self.time_step_mem_par = (self.time_step_mem_par + 1) % UPDATE_MEM_PAR_EVERY

        if self.time_step_mem_par == 0:
            self.memory.update_parameters()
        if self.time_step_nn == 0:
            if self.memory.experience_count > EXPERIENCES_PER_SAMPLING:
                sample = self.memory.sample()
                self.learn(sample, GAMMA)
        if self.time_step_mem == 0:
            self.memory.update_memory_sampling()

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnet_local.eval()
        with torch.no_grad():
            action_values = self.qnet_local(state)
        self.qnet_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arrange(self.action_size))

    def learn(self, sample, gamma):
        states, actions, rewards, next_states, dones, weights, indices = sample
        q_target = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        expected_values = rewards + gamma * q_target*(1 - dones)
        output = self.qnet_local(states).gather(1, actions)
        loss = F.mse_loss(output, expected_values)
        if self.compute_weights:
            with torch.no_grad():
                weight = sum(np.multiply(weights, loss.data.cpu().numpy()))
            loss *= weight
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnet_local, self.qnet_target, TAU)

        delta = abs(expected_values - output.detach()).numpy()
        self.memory.update_priorities(delta, indices)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)

