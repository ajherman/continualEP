import numpy as np

val = 0.83
error 0.0
n_steps=10000
omega=10
for t in range(n_steps):
    spike = np.floor(omega*(val+error))/omega
    error += val-spike
