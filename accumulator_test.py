import numpy as np

val = 0.2
error=0.0
n_steps=10000
omega=16
spikes=[]
errors=[]
for omega in [14,15,16,17,18]:
    for t in range(n_steps):
        spike = np.floor(omega*(val+error))/omega
        error += val-spike
        spikes.append(spike)
        errors.append(error)
    # print(spikes)
    print(omega,np.std(errors))
