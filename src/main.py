from environment import Environment
from Config import Config
import sys
from experiment import Experiment
import logging
from matplotlib import pyplot as plt
import time
from SAC import sac
from scipy.interpolate import splrep, splev
# set logging
logging.basicConfig(level=logging.INFO)

def main(config_file_name):
    config = Config(config_file_name)
    # Experiment
    experiment = Experiment(config, compute_weights=False)
    experiment_w = Experiment(config, compute_weights=True)
    start = time.time()

    scores, scores_avgs = experiment.runExperiment()
    scores_w, scores_avgs_w = experiment_w.runExperiment()
    #scores, scores_avgs = sac(Environment(config))

    end = time.time()
    #print(scores)
    print(f"Elapsed time:\t{end - start}")
    #plt.plot(scores)
    plt.plot(scores_avgs, 'r')
    plt.plot(scores_avgs_w, 'b')
    plt.savefig("experiment_weights.png")


if __name__ == '__main__':
    if len(sys.argv) > 2:
        raise IOError
    if len(sys.argv) == 1:

        # list different configs
        main('config_1')
        #main('config_2_frozen_lake')
        # main('config_1')


    else:
        config = sys.argv[1]
        main(config)
