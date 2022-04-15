from Config import Config
from environment import Environment
import logging

logging.basicConfig(level=logging.INFO)


config = Config('config_1')
env = Environment(config)
env.step(1)


