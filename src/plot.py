from cProfile import label
from matplotlib import pyplot as plt
import numpy as np
#import torch

base_mean = []
base_sd = []
with open('base_means.txt', 'r') as bm:
    for line in bm:
        base_mean.append(float(line))

with open('base_stds.txt', 'r') as bs:
    for line in bs:
        base_sd.append(float(line))

weight_mean = []
weight_sd = []
with open('weight_means.txt', 'r') as wm:
    for line in wm:
        weight_mean.append(float(line))

with open('weight_stds.txt', 'r') as ws:
    for line in ws:
        weight_sd.append(float(line))

uni_mean = []
uni_sd = []
with open('uniform_means.txt', 'r') as um:
    for line in um:
        uni_mean.append(float(line))

with open('uniform_stds.txt', 'r') as us:
    for line in us:
        uni_sd.append(float(line))

base_mean = np.array(base_mean)
base_sd = np.array(base_sd)
weight_mean = np.array(weight_mean)
weight_sd = np.array(weight_sd)
uni_mean = np.array(uni_mean)
uni_sd = np.array(uni_sd)

plt.plot(base_mean, 'b', label = "No weights")
plt.fill_between(range(len(base_mean)), (base_mean-base_sd), (base_mean+base_sd), color='b', alpha=.1)
plt.plot(weight_mean, 'r', label = "With weights")
plt.fill_between(range(len(weight_mean)), (weight_mean-weight_sd), (weight_mean+weight_sd), color='r', alpha=.1)
plt.plot(uni_mean, 'g', label = "Uniform sampling")
plt.fill_between(range(len(uni_mean)), (uni_mean-uni_sd), (uni_mean+uni_sd), color='g', alpha=.1)
plt.legend(loc="lower right")
plt.xlabel("Epoch x10")
plt.ylabel("Average return for 100 epochs")
plt.title("Ablation experiments")
plt.savefig("experiment.png")
