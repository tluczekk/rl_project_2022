from environment import Environment
from Config import Config
import sys
from experiment import Experiment
import logging
from matplotlib import pyplot as plt
import time
from SAC import sac
from scipy.interpolate import splrep, splev
import numpy as np
from agent import Agent_DQN
import torch
# set logging
logging.basicConfig(level=logging.INFO)

def main(config_file_name):
    config = Config(config_file_name)
    # Experiment
    experiment = Experiment(config, compute_weights=False)
    #experiment_w = Experiment(config, compute_weights=True)
    start = time.time()

    avgs = []
    for i in range(5):
        scores, scores_avgs = experiment.runExperiment()
        avgs.append(scores_avgs)
        
        
    # Evaluation on 100 cases
    agent = Agent_DQN(config, compute_weights=False)
    agent.qnet_local.load_state_dict(torch.load('model.pth'))
    env = Environment(config)
    eval_rewards = []
    for j in range(100):
        state = env.reset()
        episode_rewards = 0
        for k in range(config.exp_max_nbr_of_steps * 5):
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            episode_rewards += reward
            if done:
                break
        eval_rewards.append(episode_rewards)

    #scores_w, scores_avgs_w = experiment_w.runExperiment()
    #scores, scores_avgs = sac(Environment(config))

    end = time.time()
    
    means = np.mean(avgs, axis=0)
    stds = np.std(avgs, axis=0)

    with open('uniform_means.txt', 'w') as f:
        for mean in means:
            f.write(str(mean) + '\n')

    with open('uniform_stds.txt', 'w') as g:
        for std in stds:
            g.write(str(std) + '\n') 

    with open('uniform_eval.txt', 'w') as h:
        for ev in eval_rewards:
            h.write(str(ev) + '\n')

    #print(scores)
    print(f"Elapsed time:\t{end - start}")
    #plt.plot(scores)
    plt.plot(means, 'b')
    #plt.plot(scores_avgs_w, 'b')
    plt.fill_between(range(len(means)), (means-stds), (means+stds), color='b', alpha=.1)
    plt.savefig("experiment_uniform.png")


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
