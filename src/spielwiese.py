import numpy as np
import logging
from enums import Action

m = np.zeros((5,5))

m[2:5, 2:5] = 1

r = np.where(m>0)
r = np.asarray(r)
print()

print(r)
print(m)

print(r.shape[1])

for i in range(r.shape[1]):
    tup = r[0,i], r[1,i]
    print(tup)



print(a)



