import numpy as np
from scipy import stats

base_eval = []
weight_eval = []
uni_eval = []

with open('base_eval.txt', 'r') as be:
    for line in be:
        base_eval.append(int(line))

with open('weight_eval.txt', 'r') as we:
    for line in we:
        weight_eval.append(int(line))

with open('uniform_eval.txt', 'r') as ue:
    for line in ue:
        uni_eval.append(int(line))

print("=====================")
print(f"Average value for base:\t{np.mean(base_eval)}")
print(f"Standard deviation for base:\t{np.std(base_eval)}")
print(f"Average value for weight:\t{np.mean(weight_eval)}")
print(f"Standard deviation for weight:\t{np.std(weight_eval)}")
print(f"Average value for uni:\t{np.mean(uni_eval)}")
print(f"Standard deviation for uni:\t{np.std(uni_eval)}")
print("==============================")
print("t-tests")
bwt, bwp = stats.ttest_rel(base_eval, weight_eval)
print(f"Base & Weight - t: {bwt} p: {bwp}")
wut, wup = stats.ttest_rel(weight_eval, uni_eval)
print(f"Weight & Uni - t: {wut} p: {wup}")
ubt, ubp = stats.ttest_rel(uni_eval, base_eval)
print(f"Uni & Base - t: {ubt} p: {ubp}")