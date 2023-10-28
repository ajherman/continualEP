import numpy as np
from matplotlib import pyplot as plt
import csv
import argparse
import pickle
from itertools import product
import json
import os

"""
'Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid',
'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale',
'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette',
'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper',
'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white',
'seaborn-whitegrid', 'tableau-colorblind10'
"""

# plt.style.use('ggplot')
colormap = plt.cm.nipy_spectral
parser = argparse.ArgumentParser(description='graphing')

parser.add_argument(
    '--directory',
    type=str,
    default='.',
    help='choose file')

parser.add_argument(
    '--plot-type',
    default=None,
    help='what to plot')

args = parser.parse_args()

param_list = ['N1','N2','dt','size_tab','lr_tab','use_bias','no_reset','no_rhop','plain_data','update_rule','trace_decay','spiking','step','no_clamp','beta']


def getFiles(root_dir):
    file_dict = {}
    for relative_dir in os.listdir(root_dir):
        dir = os.path.join(root_dir,relative_dir)
        if os.path.isdir(dir):
            param_path = os.path.join(dir,"params.txt")
            if os.path.isfile(param_path):
                with open(param_path,'rb') as f:
                    param_dict = json.load(f)
                results_file = os.path.join(dir,"results.csv")
                if os.path.isfile(results_file):
                    with open(results_file,'r',newline='') as csv_file:
                        csv_reader = csv.reader(csv_file)
                        train_error,test_error = np.array(list(csv_reader)).astype('float').T
                    file_dict[dir] = {'param_dict':param_dict,'train_error':train_error,'test_error':test_error}
    return file_dict

pairs = getFiles(args.directory)
print(pairs)


# fig, ax = plt.subplots(figsize=(40,40))
# n_steps,n_layers = np.shape(deltas)
# color = iter(colormap(np.linspace(0,1,12)))
# layers=[i for i in range(n_layers)]
# for layer in layers:
#     ax.plot(deltas[:,layer],c=next(color),linewidth=1)
#     ax.set_xlabel('Step')
#     ax.set_ylabel('Test error rate (%)')
#     # ax.set_xlim([0,120])
#     # ax.set_ylim([0,1])
#     ax.grid(axis='y')
#     ax.set_title('L2 norm of layer diffs',fontsize=40)
# fig.suptitle(args.directory,fontsize=50)
# fig.legend(layers, loc='lower right', ncol=len(layers), bbox_transform=fig.transFigure,fontsize=30)
# fig.savefig(args.directory+"/deltas.png",bbox_inches="tight")


fig, ax = plt.subplots(2,2,figsize=(40,40))
for idx,update_rule in enumerate(['cep','skewsym','stdp','nonspikingstdp']):
    ax[idx//2,idx%2].grid(axis='y')
    ax[idx//2,idx%2].set_ylim([0,20])
    ax[idx//2,idx%2].set_xlabel('Epoch',fontsize=50)
    ax[idx//2,idx%2].set_ylabel('Test error rate (%)',fontsize=50)
    ax[idx//2,idx%2].set_title('Update rule: '+update_rule,fontsize=40)
    color = iter(colormap(np.linspace(0,1,12)))

    for key in pairs.keys():
        file_dict = pairs[key]
        params = file_dict['param_dict']
        test_error = file_dict['test_error']
        if params['update_rule'] == update_rule:
            if 'spike_method' in params.keys():
                if params['spike_method'] == 'poisson':
                    ax[idx//2,idx%2].plot(test_error,label="poisson",linewidth=5,c='blue')
                elif params['spike_method'] == 'accumulator':
                    ax[idx//2,idx%2].plot(test_error,label='accumulator: omega='+str(params['omega']),linewidth=5)
    leg = plt.legend(loc='upper center',fontsize=50)
    title = ""
    # title += "\nUpdate rule: "+update_rule
    title += "\nArchitecture: "+str(params['size_tab'])
    # print(key)
    # title += "\nSpiking: "+update_rule
    fig.suptitle(title,fontsize=80)
    fig.savefig(args.directory+"/test.png",bbox_inches="tight")
