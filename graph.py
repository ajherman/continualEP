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
# print(pairs)



'''
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
'''
'''
fig, ax = plt.subplots(2,2,figsize=(40,40))
rules = ['cep','skewsym','stdp','nonspikingstdp']
spike_methods = ['none','poisson','accumulator_1','accumulator_2','accumulator_4','accumulator_8','accumulator_16','accumulator_32']
# spike_methods = ['none','poisson','accumulator_1','accumulator_4','accumulator_16']

for idx,update_rule in enumerate(rules):
    ax[idx//2,idx%2].grid(axis='y')
    ax[idx//2,idx%2].set_ylim([0,100])
    ax[idx//2,idx%2].set_xlabel('Epoch',fontsize=40)
    ax[idx//2,idx%2].set_ylabel('Test error rate (%)',fontsize=40)
    ax[idx//2,idx%2].set_title('Update rule: '+update_rule,fontsize=50)
    colors = iter(colormap(np.linspace(0,1,len(spike_methods))))
    for spike_method in spike_methods:
        train_error,test_error=[0],[0]
        results_file = args.directory+"/compare_spike_methods_"+update_rule+"_"+spike_method+"/results.csv"
        train_error,test_error=[],[] # So it will increment the color even if it can't find the file
        with open(results_file,'r',newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            train_error,test_error = np.array(list(csv_reader)).astype('float').T
        ax[idx//2,idx%2].plot(test_error,linewidth=3,color=next(colors))
title = "Error over time"
fig.suptitle(title,fontsize=80)
fig.legend(spike_methods, loc='lower center', ncol=len(spike_methods)//2, bbox_transform=fig.transFigure,fontsize=40)
fig.savefig(args.directory+"/test2.png",bbox_inches="tight")
'''


rules=['skewsym','slow_stdp','fast_stdp','slug_stdp','glacial_stdp']
# rules=['cep','skewsym','stdp']
# Ms=[1,4,7,15]
Ms=[1,4,8,16,32]
omegas=['1','4096']
freqs=[8,16]
# for rule in rules:
fig, ax = plt.subplots(5,2,figsize=(60,100))
for idx1,rule in enumerate(rules):
    for idx2,omega in enumerate(omegas):
        ax[idx1,idx2].grid(axis='y')
        ax[idx1,idx2].set_xlim([0,30])
        ax[idx1,idx2].set_ylim([0,16])
        ax[idx1,idx2].set_xlabel('Epoch',fontsize=40)
        ax[idx1,idx2].set_ylabel('Test error rate (%)',fontsize=40)
        ax[idx1,idx2].set_title("Rule: "+rule+r", $\omega=$"+str(omega),fontsize=50)
        colors = iter(colormap(np.linspace(0,1,len(Ms))))
        for M in Ms:
            subdir=rule+"_M_"+str(M)+"_omega_"+str(omega)
            train_error,test_error=[0],[0]

            results_file = args.directory+"/"+subdir+"/results.csv"
            train_error,test_error=[],[] # So it will increment the color even if it can't find the file
            if os.path.isfile(results_file):
                with open(results_file,'r',newline='') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    try:
                        train_error,test_error = np.array(list(csv_reader)).astype('float').T
                    except:
                        print(results_file)
            c=next(colors)
            ax[idx1,idx2].plot(test_error,linewidth=2,color=c)
            # ax[idx1,idx2].plot(train_error,linewidth=2,color=c,linestyle='dashed')

title = "Error over time: "+rule
fig.suptitle(title,fontsize=80)
# fig.legend([1,1,4,4,8,8,16,16,32,32], loc='lower center', ncol=len(Ms), bbox_transform=fig.transFigure,fontsize=40)
fig.legend(Ms, loc='lower center', ncol=len(Ms), bbox_transform=fig.transFigure,fontsize=40)
fig.savefig(args.directory+"/pop_avg.png",bbox_inches="tight")
fig.clf()


fig, ax = plt.subplots(4,3,figsize=(60,60))
spike_method = 'binom'
rules=['skewsym','stdp1','stdp2','stdp3']
# rules=['cep','skewsym','stdp']
# Ms=[1,4,7,15]
Ms=[1,8,16,40]
omegas=['1','7','1e15']
for idx1,rule in enumerate(rules):
    for idx2,omega in enumerate(omegas):
        ax[idx1,idx2].grid(axis='y')
        ax[idx1,idx2].set_xlim([0,30])
        ax[idx1,idx2].set_ylim([0,20])
        ax[idx1,idx2].set_xlabel('Epoch',fontsize=40)
        ax[idx1,idx2].set_ylabel('Test error rate (%)',fontsize=40)
        ax[idx1,idx2].set_title('Rule = '+str(rule)+r", $\omega=$"+str(omega),fontsize=50)
        colors = iter(colormap(np.linspace(0,1,len(Ms))))
        for M in Ms:
            # subdir="poisson_"+rule+"_M_"+str(M)
            subdir="fast_"+spike_method+"_"+rule+"_M_"+str(M)+"_omega_"+str(omega)
            train_error,test_error=[0],[0]

            results_file = args.directory+"/"+subdir+"/results.csv"
            train_error,test_error=[],[] # So it will increment the color even if it can't find the file
            if os.path.isfile(results_file):
                with open(results_file,'r',newline='') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    try:
                        train_error,test_error = np.array(list(csv_reader)).astype('float').T
                    except:
                        print(results_file)
            c=next(colors)
            ax[idx1,idx2].plot(test_error,linewidth=2,color=c)
            # ax[idx1,idx2].plot(train_error,linewidth=2,color=c,linestyle='dashed')

title = "Error over time"
fig.suptitle(title,fontsize=80)
# fig.legend([1,1,4,4,7,7,], loc='lower center', ncol=len(Ms), bbox_transform=fig.transFigure,fontsize=40)
fig.legend([1,4,7,15], loc='lower center', ncol=len(Ms), bbox_transform=fig.transFigure,fontsize=40)
fig.savefig(args.directory+"/"+spike_method+"_fast_blowups.png",bbox_inches="tight")





fig, ax = plt.subplots(2,4,figsize=(60,60))
# omegas=[1,4,16,64,256,1024]
# omegas=[0.8,1,2,3,15,63,255,1023]
# omegas=[14,15,16,17,18,63,64,65]
# omegas=[1,4]
# taus=[0.025,0.05,0.1,0.2]
rules=['cep','skewsym','stdp0','stdp1','stdp2','stdp3','stdp4','stdp5']
# rules=['cep','skewsym','stdp']
# Ms=[1,2,4,7,15]
omegas=[1,2,4,8,16,1024]
for idx,rule in enumerate(rules):
    ax[idx%2,idx//2].grid(axis='y')
    ax[idx%2,idx//2].set_xlim([0,30])
    ax[idx%2,idx//2].set_ylim([0,20])
    ax[idx%2,idx//2].set_xlabel('Epoch',fontsize=40)
    ax[idx%2,idx//2].set_ylabel('Test error rate (%)',fontsize=40)
    ax[idx%2,idx//2].set_title('Rule = '+str(rule),fontsize=50)
    colors = iter(colormap(np.linspace(0,1,len(omegas))))
    # for M in Ms:
    for omega in omegas:
        # subdir="poisson_"+rule+"_M_"+str(M)
        subdir="compare_"+rule+"_omega_"+str(omega)+"_tau_0.2"
        train_error,test_error=[0],[0]

        results_file = args.directory+"/"+subdir+"/results.csv"
        train_error,test_error=[],[] # So it will increment the color even if it can't find the file
        if os.path.isfile(results_file):
            with open(results_file,'r',newline='') as csv_file:
                csv_reader = csv.reader(csv_file)
                train_error,test_error = np.array(list(csv_reader)).astype('float').T
        ax[idx%2,idx//2].plot(test_error,linewidth=2,color=next(colors))

        # params_file = args.directory+"/"+subdir+"/params.txt"
        # if os.path.isfile(params_file):
        #     with open(params_file,'rb') as f:
        #         param_dict = json.load(f)
        #         tau_dynamic = -param_dict['step']/np.log(1-param_dict['dt'])
        #         tail_str = 'tau'+str(int(tau_dynamic*10))+'_step'+str(int(param_dict['step']*100))
title = "Error over time"
fig.suptitle(title,fontsize=80)
fig.legend(omegas, loc='lower center', ncol=len(rules)//2, bbox_transform=fig.transFigure,fontsize=40)
# fig.savefig(args.directory+"/accumulator_"+tail_str+".png",bbox_inches="tight")
fig.savefig(args.directory+"/compare_omega.png",bbox_inches="tight")




fig, ax = plt.subplots(8,2,figsize=(60,60))
omegas=[1,4,16,64,256,1024]
# omegas=[0.8,1,2,3,15,63,255,1023]
# omegas=[14,15,16,17,18,63,64,65]
# omegas=[1,4]
taus=[None]
rules=['cep','skewsym','stdp0','stdp1','stdp2','stdp3','stdp4','stdp5']
for idx1,omega in enumerate(omegas):
    for idx2,tau in enumerate(taus):
        ax[idx1,idx2].grid(axis='y')
        ax[idx1,idx2].set_xlim([0,12])
        ax[idx1,idx2].set_ylim([0,20])
        ax[idx1,idx2].set_xlabel('Epoch',fontsize=40)
        ax[idx1,idx2].set_ylabel('Test error rate (%)',fontsize=40)
        ax[idx1,idx2].set_title(r'Accumulator $\omega$='+str(omega)+r', $\tau=$'+str(tau),fontsize=50)
        colors = iter(colormap(np.linspace(0,1,len(rules))))
        for rule in rules:
            subdir="compare_spike_methods_"+rule+"_accumulator_"+str(omega) #+"_tau_"+str(tau)
            train_error,test_error=[0],[0]

            results_file = args.directory+"/"+subdir+"/results.csv"
            train_error,test_error=[],[] # So it will increment the color even if it can't find the file
            if os.path.isfile(results_file):
                with open(results_file,'r',newline='') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    train_error,test_error = np.array(list(csv_reader)).astype('float').T
            ax[idx1,idx2].plot(test_error,linewidth=2,color=next(colors))

            params_file = args.directory+"/"+subdir+"/params.txt"
            if os.path.isfile(params_file):
                with open(params_file,'rb') as f:
                    param_dict = json.load(f)
                    tau_dynamic = -param_dict['step']/np.log(1-param_dict['dt'])
                    tail_str = 'tau'+str(int(tau_dynamic*10))+'_step'+str(int(param_dict['step']*100))
title = "Error over time"
fig.suptitle(title,fontsize=80)
fig.legend(rules, loc='lower center', ncol=len(rules)//2, bbox_transform=fig.transFigure,fontsize=40)
# fig.savefig(args.directory+"/accumulator_"+tail_str+".png",bbox_inches="tight")
fig.savefig(args.directory+"/accumulator.png",bbox_inches="tight")




#
# fig, ax = plt.subplots(5,2,figsize=(40,100))
# # omegas=[1,4,16,64,256,1024]
# # omegas=[0.8,1,2,3,15,63,255,1023]
# # omegas=[14,15,16,17,18,63,64,65]
# omegas=[1,2,3,4,15,16,63,64,1023,1024]
# rules=['cep','skewsym','stdp0','stdp1','stdp2','stdp3','stdp4','stdp5']
# for idx,omega in enumerate(omegas):
#     ax[idx//2,idx%2].grid(axis='y')
#     ax[idx//2,idx%2].set_xlim([0,12])
#     ax[idx//2,idx%2].set_ylim([0,20])
#     ax[idx//2,idx%2].set_xlabel('Epoch',fontsize=40)
#     ax[idx//2,idx%2].set_ylabel('Test error rate (%)',fontsize=40)
#     ax[idx//2,idx%2].set_title('Accumulator omega: '+str(omega),fontsize=50)
#     colors = iter(colormap(np.linspace(0,1,len(rules))))
#     for rule in rules:
#         train_error,test_error=[0],[0]
#
#         results_file = args.directory+"/compare_"+rule+"_omega_"+str(omega)+"/results.csv"
#         train_error,test_error=[],[] # So it will increment the color even if it can't find the file
#         with open(results_file,'r',newline='') as csv_file:
#             csv_reader = csv.reader(csv_file)
#             train_error,test_error = np.array(list(csv_reader)).astype('float').T
#         ax[idx//2,idx%2].plot(test_error,linewidth=1,color=next(colors))
#
#         params_file = args.directory+"/compare_"+rule+"_omega_"+str(omega)+"/params.txt"
#         if os.path.isfile(params_file):
#             with open(params_file,'rb') as f:
#                 param_dict = json.load(f)
#         tau_dynamic = -param_dict['step']/np.log(1-param_dict['dt'])
#         tail_str = 'tau'+str(int(tau_dynamic*10))+'_step'+str(int(param_dict['step']*100))
# title = "Error over time"
# fig.suptitle(title,fontsize=80)
# fig.legend(rules, loc='lower center', ncol=len(rules)//2, bbox_transform=fig.transFigure,fontsize=40)
# # fig.savefig(args.directory+"/accumulator_"+tail_str+".png",bbox_inches="tight")
# fig.savefig(args.directory+"/accumulator.png",bbox_inches="tight")
