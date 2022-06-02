import logging
from collections import deque

import numpy as np
from Config import Config
from agent import Agent_DQN
from environment import Environment
import torch 

class Experiment:

    def __init__(self, config: Config, compute_weights = False, env = None):
        """

        """
        self.config = config
        self.agent = Agent_DQN(config, compute_weights)
        if env is None:
            self.environment = Environment(config)
        else:
            self.environment = env


    def runExperiment(self) -> None:
        """
        This function runs the main algorithm
        """
        scores = []
        scores_window = deque(maxlen=100)
        scores_avgs = []
        eps = self.config.env_epsilon_start

        for episode in range(1, self.config.exp_nbr_episodes+1):
            logging.info(f'########### Episode {episode} ###########\n')
            state = self.environment.reset()
            score = 0
            for t in range(self.config.exp_max_nbr_of_steps):
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.environment.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores.append(score)
            scores_window.append(score)
            eps = max(self.config.env_epsilon_end, self.config.env_epsilon_decay*eps)
            if episode % 10 == 0:
                scores_avgs.append(np.mean(scores_window))
        
        self.saveResults('model.pth')
        return scores, scores_avgs

    def saveResults(self, path):
        """

        """
        # Saving the model for later evaluation
        torch.save(self.agent.qnet_local.state_dict(), path)

    
