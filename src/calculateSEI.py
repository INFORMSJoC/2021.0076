# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import simps
from numpy import trapz


m = np.loadtxt("LR_sub_acc.txt").T

# Compute the area using the composite trapezoidal rule.
SEIs = trapz(m, dx=0.1)/2
print("SEIs =", SEIs)

