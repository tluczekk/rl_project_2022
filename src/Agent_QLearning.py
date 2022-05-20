from src.model.logic.config import config
from src.model.logic.agent.Agent import Agent
from src.model.logic.State import State
import tensorflow.keras as keras
import numpy as np
import logging
import enums

class Agent_QLearning(Agent):

    def __init__(self, config: config):
        self.config = config
        # Hyperparameters
        self.alpha = self.config.agent_stepsize_alpha
        self.gamma = self.config.agent_discount_factor_gamma
        self.epsilon = 0.0
        # Switch back to a small number if Exploration is still needed.
        # self.epsilon = 0.01  # Exploration factor
        self.epsilon_decay = 0.999

        self.previous_action = None
        self.action_space = enums.Action
        self.state_space = [-2, -1, 0, 1, 2]

        # Initialize the q-Table (self.q)
        self.state_size = (len(self.state_space) ** config.state_size-1) * 2
        self.q = np.zeros((self.state_size, len(self.action_space)))
        self.rand_generator = np.random.RandomState(seed=42)
        np.set_printoptions(precision=3)
        self.step = 0

    def generateAction(self, currentState: State) -> int:
        """
        This function evaluates all possible actions and chooses (with epsilon-greedy function)
        the best action for any state.
        :currentState: The State for Action evaluation.
        :return: integer of Action.
        """
        data_list = np.asarray(currentState.data_list).astype('float32')

        state = self.translate_state(data_list)
        self.epsilon *= self.epsilon_decay
        # Choose action using epsilon greedy.
        if self.rand_generator.rand() < self.epsilon:
            action = np.random.choice(self.action_space)
            logging.info(f'Action: {action} with value "exploration" (current epsilon is: '
                         f'{self.epsilon}')
        else:
            action, top = self.argmax(self.q[state, :])
            action = self.action_space[action]
            logging.info(f'Action: {action} with value {top}')
        self.previous_action = action

        return action

    def update(self, reward_list: list, lastState: State, newState: State):
        """
        The following code follows the one-step Actor-Critic pseudocode from Sutton, Barto, p. 332
        """

        data_list_old = np.asarray(lastState.data_list).astype('float32')
        data_list_new = np.asarray(newState.data_list).astype('float32')

        for action in self.action_space:
            # Prepare next state:
            if action == -1:
                # All Stocks gone
                data_list_new[-1] = 0
            elif action == 0:
                # Nothing done
                data_list_new[-1] = data_list_old[-1]
            else:
                # All money in stocks
                data_list_new[-1] = 1

            state_old = self.translate_state(data_list_old)
            state_new = self.translate_state(data_list_new)

            target_action = self.action_space.index(action)
            reward = reward_list[target_action]
            # Perform update
            logging.debug(f'updating action {action} with value {reward}')
            logging.debug(f'before: {self.q[state_old]}')
            self.q[state_old, target_action] += self.alpha * (reward + self.gamma *
                                                              np.max(self.q[state_new, :])
                                                              - self.q[state_old, target_action])
            logging.debug(f'after: {self.q[state_old]}')

        # Print Q-Table after every 100th step, if needed.
        if self.step % 100 == 0:
            logging.debug(f'Weights after {self.step} steps:\n{self.q}')
        self.step += 1



    def getTargets(self):
        """
        No targets for actor and critic
        """
        return 0, 0

    def argmax(self, q_values):
        """
        argmax function with random tie-breaking
        :param q_values: the Numpy array of action-values
        :return: action (int): an action with the highest value
        """
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = []

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties), top

    def translate_state(self, data_list):
        """
        Translates the state to a one dimensional number
        """
        state_number = 0
        base = 0
        for element in data_list[:-1]:
            state_number += len(self.state_space) ** base * self.state_space.index(element)
            base += 1
        if data_list[-1] == 1:
            state_number += int(self.state_size/2)
        logging.debug(f'Calculated State number for {data_list}: {state_number}')
        return state_number
