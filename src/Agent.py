from abc import ABC, abstractmethod
import logging

import tensorflow.keras as keras


def agentSwitcher(config):
    """
    Switch statement to execute the correct agent creation function according to the agent name from config.
    :param agent_name_from_config: The agent name from config
    :param config: Config File of this Environment.
    :return: the function for model creation which corresponds with the config model name
    """
    switcher = {
        "Agent_QLearning": "get_agent_qlearning(config)",
        "Agent_DQN": "get_agent_dqn(config)"
    }
    return eval(switcher.get(config.agent_name))


def get_agent_qlearning(config):
    Agent_QLearning
    logging.info("Building Agent_QLearning")
    return Agent_QLearning(config)


def get_agent_dqn(config):
    import Agent_DQN
    logging.info("Building Agent_DQN")
    return Agent_DQN(config)

class Agent(ABC):
    """
    This abstract class is used to define the API for agents.
    """

    @abstractmethod
    def generateAction(self, currentState: State):
        pass

    @abstractmethod
    def update(self, reward: float, lastState: State, newState: State):
        pass

    @abstractmethod
    def getTargets(self):
        pass
