from Config import Config
import sys
from experiment import Experiment

"""
main runs the experiments
"""

def main(config_file_name):
    config = Config(config_file_name)
    # Experiment
    experiment = Experiment(config)
    experiment.run_experiment()


if __name__ == '__main__':
    if len(sys.argv) > 2:
        raise IOError
    if len(sys.argv) == 1:

        # list different configs
        main('config_1')
        # main('config_1')
        # main('config_1')


    else:
        config = sys.argv[1]
        main(config)
