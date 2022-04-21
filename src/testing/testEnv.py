from Config import Config
from environment import Environment
import logging
from enums import Action

logging.basicConfig(level=logging.INFO)

### check if map creation makes sense
config = Config('config_1')
env = Environment(config)

### check if players move correctly
for i in range(100):
    new_state, reward, done, info =  env.step(Action.DOWN.value)
    if done:
        print(reward)
        env.print_map()
        break


