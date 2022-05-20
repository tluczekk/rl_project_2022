import logging
from collections import deque
import Agent
import numpy as np
from Config import Config
from environment import Environment
import torch


class Experiment:

    def __init__(self, config: Config):
        """

        """
        self.config = config
        self.agent = Agent.agentSwitcher(config)
        self.environment = Environment(config)


    def runExperiment(self, config) -> None:
        """
        This function runs the main algorithm
        """

        scores = []
        scores_window = deque(maxlen=100)
        scores_avgs = []

        for episode in range(1, config.exp_episodes + 1):
            logging.info(f'########### Episode {episode} ###########\n')
            state = self.environment.reset()
            score = 0
            for t in range(config.exp_max_nbr_of_steps):
                action = self.agent.generateAction(currentState=state)
                next_state, reward, done, _ = self.environment.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores.append(score)
            scores_window.append(score)
            eps = max(config.env_eps_end, config.env_eps_decay * eps)
            if episode % 50 == 0:
                scores_avgs.append(np.mean(scores_window))
        
        self.saveResults('model.pth')
        return scores, scores_avgs

    def saveResults(self, path):
        """

        """
        # Saving the model for later evaluation
        torch.save(self.agent.qnet_local.state_dict(), path)

    
