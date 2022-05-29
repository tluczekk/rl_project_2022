"""
Source: https://arxiv.org/pdf/1511.05952.pdf
"""

import random
from collections import namedtuple, deque
import numpy as np
from numpy.random import choice
import torch
import operator

from Config import Config

# Speeding up PyTorch calculations if Nvidia's CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, experiences_per_sampling, seed, compute_weights, config: Config) -> None:
        # Parameters passed by agent
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experiences_per_sampling = experiences_per_sampling

        # Parameters specific to Replay Buffer
        self.alpha = config.buf_alpha
        self.alpha_decay_rate = config.buf_alpha_decay
        self.beta = config.buf_beta
        self.beta_growth_rate = config.buf_beta_growth
        self.seed = random.seed(seed)
        self.compute_weights = compute_weights
        self.experience_count = 0

        # Dictionaries storing vital information about experience buffer
        #
        # Dicitionary storing details about each experience
        self.experience = namedtuple("Experience",
            field_names=["state", "action", "reward", "next_state", "done"])
        # Dictionary storing data associated with experience, that is its priority, probability of being sampled and weight
        self.data = namedtuple("Data",
            field_names=["priority", "probability", "weight", "index"])
        
        # Initializing buffer
        indices = []
        index_data = []
        for i in range(buffer_size):
            indices.append(i)
            d = self.data(0, 0, 0, i)
            index_data.append(d)

        # Initializing utility variables 
        self.memory = {key: self.experience for key in indices}
        self.memory_data = {key: data for key, data in zip(indices, index_data)}
        self.sampled_batches = []
        self.current_batch = 0
        self.priorities_sum_alpha = 0
        self.priorities_max = 1
        self.weights_max = 1

    def update_priorities(self, tds, indices):
        for td, index in zip(tds, indices):
            N = min(self.experience_count, self.buffer_size)

            updated_priority = td[0]

            if updated_priority > self.priorities_max:
                self.priorities_max = updated_priority
            
            if self.compute_weights:
                if updated_priority == 0:
                    updated_weight = 0
                else: 
                    updated_weight = ((N * updated_priority)**(-self.beta))/self.weights_max
                if updated_weight > self.weights_max:
                    self.weights_max = updated_weight
            else:
                updated_weight = 1

            old_priority = self.memory_data[index].priority
            self.priorities_sum_alpha += updated_priority**self.alpha - old_priority**self.alpha
            updated_probability = td[0]**self.alpha / self.priorities_sum_alpha
            data = self.data(updated_priority, updated_probability, updated_weight, index)
            self.memory_data[index] = data

    def update_memory_sampling(self):
        self.current_batch = 0
        values = list(self.memory_data.values())
        random_values = random.choices(self.memory_data,
                                        [data.probability for data in values],
                                        k = self.experiences_per_sampling)
        self.sampled_batches = [random_values[i: i + self.batch_size] for i in range(0,len(random_values), self.batch_size)]

    def update_parameters(self):
        self.alpha *= self.alpha_decay_rate
        self.beta *= self.beta_growth_rate
        if self.beta > 1:
            self.beta = 1
        N = min(self.experience_count, self.buffer_size)
        self.priorities_sum_alpha = 0
        sum_prob_before = 0
        for element in self.memory_data.values():
            sum_prob_before += element.probability
            self.priorities_sum_alpha += element.priority**self.alpha
        sum_prob_after = 0
        for element in self.memory_data.values():
            probability = element.priority**self.alpha / self.priorities_sum_alpha
            sum_prob_after += probability
            weight = 1
            if self.compute_weights:
                if element.probability == 0:
                    weight = 0
                else: 
                    weight = ((N * element.probability)**(-self.beta))/self.weights_max
            d = self.data(element.priority, probability, weight, element.index)
            self.memory_data[element.index] = d
        print("Sum of probabilities before:\t ", sum_prob_before)
        print("Sum of probabilities after:\t", sum_prob_after)

    def add(self, state, action, reward, next_state, done):
        """
        Function adding new experience to memory
        
        """
        self.experience_count += 1
        index = self.experience_count % self.buffer_size

        if self.experience_count > self.buffer_size:
            temp = self.memory_data[index]
            self.priorities_sum_alpha -= temp.priority ** self.alpha
            if temp.priority == self.priorities_max:
                self.memory_data[index].priority = 0
                self.priorities_max = max(self.memory_data.items(), key=operator.itemgetter(1)).priority        
            if self.compute_weights:
                if temp.weight == self.weights_max:
                    self.memory_data[index].weight = 0
                    self.weights_max = max(self.memory_data.index(), key=operator.itemgetter(2)).weight
        
        priority = self.priorities_max
        weight = self.weights_max
        self.priorities_sum_alpha += priority ** self.alpha
        probability = priority ** self.alpha / self.priorities_sum_alpha
        e = self.experience(state, action, reward, next_state, done)
        self.memory[index] = e
        d = self.data(priority, probability, weight, index)
        self.memory_data[index] = d

    def sample(self):
        sampled_batch = self.sampled_batches[self.current_batch]
        self.current_batch += 1
        experiences = []
        weights = []
        indices = []

        for data in sampled_batch:
            experiences.append(self.memory.get(data.index))
            weights.append(data.weight)
            indices.append(data.index)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).long().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return (states, actions, rewards, next_states, dones, weights, indices)

    def __len__(self):
        return len(self.memory)