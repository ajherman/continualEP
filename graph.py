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
'''
fixed: N1,N2,step,archi
variable:
same graph:
'''

def getFiles(root_dir):
    file_dict = {}
    for x in os.walk(root_dir):
        dir = x[0]
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

update_rule='cep'
fig, ax = plt.subplots(figsize=(40,40))
for key in pairs.keys():
    file_dict = pairs[key]
    params = file_dict['param_dict']
    if params['update_rule'] == update_rule:
        if params['spike_method'] == 'poisson':
            ax.plot(params['test_error'])
            ax.set_title('poisson')
        elif params['spike_method'] == 'accumulator':
            ax.set_title('accumulator: omega='+str(omega))
fig.suptitle(update_rule)
fig.savefig(args.directory+"/test.png",bbox_inches="tight")

# for key in pairs.keys():
#     print("Key: ",key,"\n",pairs[key],"\n")
#
# def findFixedParams(file_dict):
#     fixed = {}
#     for param ['N1','N2','size_tab','update_rule','spiking']:
#         param_list = []
#         for file in file_dict:
#             val = file_dict[file]['param_dict'][param]
#             param_list.append(val)
#         if all(x==param_list[0] for x in param_list):
#             fixed[param] = val
#     return fixed
#
# print(findFixedParams(pairs))
#
# for dir in file_dict.keys():
#     data_dict = file_dict[dir]
#     param_dict,train_error,test_error = data_dict['param_dict'],data_dict['train_error'],data_dict['test_error']
