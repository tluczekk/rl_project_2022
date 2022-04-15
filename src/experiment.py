from Config import Config
from environment import Environment


class Experiment:

    def __init__(self, config: Config):
        """

        """
        self.config = config
        # self.agent = Agent(params_from_config)
        self.environment = Environment(config)


    def runExperiment(self) -> None:
        """
        This function runs the main algorithm
        """
        pass


    def saveResults(self):
        """

        """
        pass

    
