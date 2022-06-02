from matplotlib import pyplot as plt
import numpy as np
#import torch

base_mean = []
with open('No_weights.txt', 'r') as bm:
    for line in bm:
        base_mean.append(float(line))


weight_mean = []
with open('Weights.txt', 'r') as wm:
    for line in wm:
        weight_mean.append(float(line))

uni_mean = []
with open('Uniform.txt', 'r') as um:
    for line in um:
        uni_mean.append(float(line))

base_mean = np.array(base_mean)
weight_mean = np.array(weight_mean)
uni_mean = np.array(uni_mean)

plt.plot(base_mean, 'b', label = "No weights")
plt.plot(weight_mean, 'r', label = "With weights")
plt.plot(uni_mean, 'g', label = "Uniform sampling")
plt.legend(loc="lower right")
plt.xlabel("Epoch x10")
plt.ylabel("Average return for 100 epochs")
plt.title("Lunar lander experiments")
plt.savefig("experiment_ll.png")
