import numpy as np
import pickle as pkl

def f(omega,epoch):
    path = 'compare_stdp2_omega_'+str(omega)+'/data_'+str(epoch)+'.pkl'
    with open(path,'rb') as f:
        x=pkl.load(f)
    return x

x=f(1024,10)
y=x['phase1']['s']
print(len(y))
