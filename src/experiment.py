
class Experiment:

    def __init__(self, config: Config):
        """

        """
        if experimentDbId is None:
            self.config = config
            self.agent = Agent(params_from_config)
            self.environment = Environment(params_from_config)


    def runExperiment(self) -> None:
        """
        This function runs the main algorithm
        """
        pass


    def saveResults(self):
        """

        """
        pass

    
