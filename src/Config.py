import datetime
import os
import configparser
import logging
import random


class Config:
    """
    The Config class imports a configuration from file and prepares python classes which will be used at various points.
    """

    # Constructor for new model
    def __init__(self, configFileName: str):
        self.configuration = configparser.ConfigParser()
        # Get absolute path to config file -> they need to be in /src/config_files/
        pathToConfigFile = os.path.join(os.path.dirname(__file__), 'config_files/', configFileName + '.ini')
        try:
            # Load configuration from file
            self.configuration.read(pathToConfigFile)
            # generate config objects from file
            self.__extractConfigFromDict()

        except FileNotFoundError:
            print("Error in RunConfig.init(): File with path ", pathToConfigFile, " not found.")

        ### Initialize default params ###

        ## environment
        if self.env_random_map:
            self.__set_random_map()
        else:
            self.env_size = 10

        if self.env_random_enemies:
            self.__set_random_enemies()
        else:
            self.env_nbr_enemies = 5

        if self.env_random_merchants:
            self.__set_random_merchants()
        else:
            self.env_nbr_merchants = 5

        self.env_action_success_prop = 1
        self.env_random_map = False
        self.env_random_enemies = False
        self.env_random_merchants = False

        # agent
        self.agent_discount_factor_gamma = 0.9
        self.agent_stepsize_alpha = 0.1

        # experiment
        self.exp_nbr_game_per_exp = 100
        self.exp_max_nbr_of_steps = 100

        ### set random params if random = True from config
        if self.env_random_merchants:
            self.__set_random_merchants()
        else:
            self.env_nbr_merchants = 5




    def __extractConfigFromDict(self):
        """
        Store infos from config file to variables in Config object.
        :return: Nothing
        """
        # Environment section
        environment_section = self.configuration['environment']
        self.env_size = environment_section.get('env_size')                                 # side length of square map
        self.env_nbr_enemies = environment_section.get('env_nbr_enemies')
        self.env_nbr_merchants = environment_section.get('env_nbr_merchants')
        self.env_action_success_prop = environment_section.get('env_action_success_prop')    # probability that action will succeed
        self.env_random_map = environment_section.get('env_random_map')                      # size, nbr of enemies and merchants is random
        self.env_random_enemies = environment_section.get('env_random_enemies')              # only nbr of enemies is random
        self.env_random_merchants = environment_section.get('env_random_merchants')          # only nbr of merchants is random

        # Agent section
        agent_section = self.configuration['agent']
        self.agent_discount_factor_gamma = agent_section.get('agent_discount_factor_gamma')
        self.agent_stepsize_alpha = agent_section.get('agent_stepsize_alpha')

        # Experiment section
        experiment_section = self.configuration['experiment']
        self.exp_max_nbr_of_steps = experiment_section.get('exp_max_nbr_of_steps')
        self.exp_nbr_game_per_exp = experiment_section.get('exp_nbr_game_per_exp')

        # Persistence options
        # self.load_model_name = self.configuratio['load_model_name']   # enter name if you want to load an existing model

    def __set_random_merchants(self):
        """
        :return: random number between 1 and the side length of the map
        """
        self.env_nbr_merchants = random.randint(1,self.env_size)

    def __set_random_enemies(self):
        """
        :return: random number between 1 and the side length of the map
        """
        self.env_nbr_merchants = random.randint(1,self.env_size)

    def __set_random_map(self):
        self.env_size = random.randint(5, 100)
        self.__set_random_merchants()
        self.__set_random_enemies()


