import numpy as np
import pickle as pkl
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='graphing')

parser.add_argument(
    '--directory',
    type=str,
    default='compare_stdp2_omega_16',
    help='choose file')

parser.add_argument(
    '--node1',
    nargs='+',
    default=[1,0],
    help='node 1')

parser.add_argument(
    '--node2',
    nargs='+',
    default=[0,0],
    help='node 2')

args = parser.parse_args()


# def f(omega,epoch):
#     path = 'compare_stdp2_omega_'+str(omega)+'/data_'+str(epoch)+'.pkl'
#     with open(path,'rb') as f:
#         x=pkl.load(f)
#     return x

def f(dir,epoch):
    # path = 'compare_stdp2_omega_'+str(omega)+'/data_'+str(epoch)+'.pkl'
    path = dir+'/data_'+str(epoch)+'.pkl'
    with open(path,'rb') as f:
        x=pkl.load(f)
    return x

def g(x,phase,layer,batch,node):
    y=x['phase1']['s']
    w=x['phase1']['spike']
#layer=1
#batch=0
#node=73
    vals = np.array([z[layer][batch,node] for z in y])
    # spikes = np.array([z[layer][batch,node] for z in w])
    spikes = vals

    if phase == 2:
        pass
    return vals,spikes

# node1 = (1,7)
# node2 = (0,7)
node1 = tuple(args.node1)
node2 = tuple(args.node2)

data_dict=f(args.directory,0)

def getInfo(data_dict,node1,node2):
    layer1,pos1 = node1
    layer2,pos2 = node2
    w_idx = 2*layer2
    if layer1<layer2:
        w_idx-=1
    phase2_dict = data_dict['phase2']
    s,spike,w = phase2_dict['s'],phase2_dict['spike'],phase2_dict['w']
    s1_arr = np.array([z[layer1][0,pos1] for z in s])
    s2_arr = np.array([z[layer2][0,pos2] for z in s])
    spike1_arr = np.array([z[layer1][0,pos1] for z in spike])
    spike2_arr = np.array([z[layer2][0,pos2] for z in spike])
    w_arr = np.array([z[w_idx][pos2,pos1] for z in w])
    #print(np.shape(w[0][w_idx]))
    return s1_arr,s2_arr,spike1_arr,spike2_arr,w_arr

def plot(data_dict,node1,node2,fname):
    s1_arr,s2_arr,spike1_arr,spike2_arr,w_arr=getInfo(data_dict,node1,node2)
    fig, ax = plt.subplots(figsize=(40,40))
    ax.plot(spike1_arr)
    ax.plot(spike2_arr)
    ax.plot(w_arr)
    fig.savefig(fname)

max_num_plots=10
k = 0
for pos1 in range(200):
    for pos2 in range(10):
        node1=(1,pos1)
        node2=(0,pos2)
        if k<max_num_plots:
            s1_arr,s2_arr,spike1_arr,spike2_arr,w_arr = getInfo(data_dict,node1,node2)
            print(np.shape(w_arr))
            assert(0)
            if np.std(w_arr)>3e-5:
                plot(data_dict,node1,node2,'weight_updates_'+str(pos1)+'_'+str(pos2)+'.png')
                k+=1


#print(s1_arr)
#print(s2_arr)
#print(spike1_arr)
#print(spike2_arr)
#print(w_arr)

#vals,spikes=g(x,1,1,0,73)
#print(vals)
#print(spikes)
#print(vals-spikes)
#print(np.shape(y[0][1]))
