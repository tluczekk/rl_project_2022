from Config import Config
import sys
from experiment import Experiment
import logging
from matplotlib import pyplot as plt
# set logging
logging.basicConfig(level=logging.INFO)

def main(config_file_name):
    config = Config(config_file_name)
    # Experiment
    experiment = Experiment(config)
    scores = experiment.runExperiment()
    print(scores)
    plt.plot(scores)
    plt.savefig("experiment_results.png")


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
