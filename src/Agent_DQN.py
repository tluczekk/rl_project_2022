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

class Agent_DQN(Agent):
    def __init__(self, config: Config, action_size, seed, compute_weights = False) -> None:
        # Hyperparameters
        self.config = config
        # Getting parameters of environment
        # Each field can be either merchant, enemy or empty, except field of agent
        super().__init__(config, len(Action), 123)

        self.state_size = config.state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.compute_weights = compute_weights
        self.config = config

        # Two Q-networks to avoid overestimation bias
        self.qnet_local = QNet(self.state_size, action_size, seed).to(device)
        self.qnet_target = QNet(self.state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnet_local.parameters(), lr=self.config.agent_learning_rate)
        self.criterion = nn.MSELoss()

        # Setting up replay buffer
        self.memory = ReplayBuffer(
            action_size, config.agent_buffer_size, config.agent_batch_size, config.agent_experiences_per_sampling, seed, compute_weights, self.config
        )
        # Counters for updating neural network, parameters and memory sampling
        self.time_step_nn = 0
        self.time_step_mem_par = 0
        self.time_step_mem = 0

    def step(self, state, action, reward, next_state, done):
        # adding current tupple to buffer
        self.memory.add(state, action, reward, next_state, done)

        # Updating counters
        self.time_step_nn = (self.time_step_nn + 1) % self.config.agent_update_nn_every
        self.time_step_mem = (self.time_step_mem + 1) % self.config.agent_update_mem_every
        self.time_step_mem_par = (self.time_step_mem_par + 1) % self.config.agent_update_mem_par_every

        if self.time_step_mem_par == 0:
            self.memory.update_parameters()
        if self.time_step_nn == 0:
            if self.memory.experience_count > self.config.agent_experiences_per_sampling:
                sample = self.memory.sample()
                self.learn(sample, self.config.agent_gamma)
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
            return random.choice(np.arange(self.action_size))

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

        self.update(self.qnet_local, self.qnet_target, self.config.agent_tau)

        delta = abs(expected_values - output.detach()).numpy()
        self.memory.update_priorities(delta, indices)

    def update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
