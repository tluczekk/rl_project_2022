import numpy as np
import logging
from enums import Action

a = np.zeros(25)+1

a[a>0] = 0

print(a)



