import torch
from Config import Config
from agent import Agent_DQN
from experiment import Experiment
from environment import Environment

config = Config('config_1')
experiment = Experiment(config, compute_weights=False)
agent = Agent_DQN(config, compute_weights=False)
agent.qnet_local.load_state_dict(torch.load('uniform.pth'))
env = Environment(config)
eval_rewards = []
for j in range(10000):
    state = env.reset()
    episode_rewards = 0
    for k in range(config.exp_max_nbr_of_steps * 5):
        action = agent.act(state)
        state, reward, done, _ = env.step(action)
        episode_rewards += reward
        if done:
            break
    eval_rewards.append(episode_rewards)

with open('uniform_eval.txt', 'w') as h:
        for ev in eval_rewards:
            h.write(str(ev) + '\n')