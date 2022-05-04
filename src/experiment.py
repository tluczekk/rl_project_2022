from Config import Config
from agent import Agent_DQN
from environment import Environment

class Experiment:

    def __init__(self, config: Config):
        """

        """
        self.config = config
        self.agent = Agent_DQN(config)
        self.environment = Environment(config)


    def runExperiment(self, n_episodes=200, max_t=500, eps_start=1.0, eps_end=0.001, eps_decay=0.995) -> None:
        """
        This function runs the main algorithm
        """
        scores = []
        eps = eps_start

        for episode in range(1, n_episodes+1):
            state = self.environment.reset()
            #env = Environment(self.config)
            #state = self.environment.get_state()
            score = 0
            for t in range(max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done, _ = self.environment.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores.append(score)
            eps = max(eps_end, eps_decay*eps)

        return scores

    def saveResults(self):
        """

        """
        pass

    
