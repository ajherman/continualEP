import numpy as np
from matplotlib import pyplot as plt
import csv
import argparse
from itertools import product

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
parser = argparse.ArgumentParser(description='Graphing Utils')

parser.add_argument(
    '--directory',
    type=str,
    default=None,
    help='choose file')

parser.add_argument(
    '--plot-type',
    default=None,
    help='what to plot')

args = parser.parse_args()

def csv2array(directory,skiplines=0):
    with open(directory+'/results.csv','r',newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        error = np.array(list(csv_reader)[skiplines:]).astype('float')
        # for row in csv_reader:
        #     print(', '.join(row))
        # print(list(csv_reader))
        # print(error)
        # print(directory)
        try:
            train_error,test_error = error[:,0],error[:,1]
        except:
            return [],[]
    return train_error, test_error

activity_type_li = ['spiking','nonspiking']
learning_rule_li = ['cepalt','skewsym','stdp_rock','stdp_slug','stdp_slow','stdp_med']
max_fr_li = [3]
N2_li = [15,30,60]
n_dynamic_li = [1,2,4,8]
beta_li = [0.4,0.9,1.5]
batch_size_li = [25,50,200]

def dir_name(activity_type,learning_rule,max_fr,N2,tau_dynamic,beta,batch_size):
    name = activity_type+"_"+learning_rule+"_maxfr="+str(max_fr)+"_N2="+str(N2)+"_tau="+str(tau_dynamic)+"_beta="+str(int(10*beta))+"_batch="+str(batch_size)
    return name

def dir_name2(learning_rule,N1,N2,n_dynamic,beta,batch_size):
    name = learning_rule+"_N1="+str(N1)+"_N2="+str(N2)+"_dyn="+str(n_dynamic)+"_beta="+str(int(10*beta))+"_batch="+str(batch_size)
    return name

# n_col,n_row=3,4
# # batch_size=200
# N1=100
# # N2=30
# batch_size=200
# for beta in [0.4,0.9,1.5,2,4]:
#     fig, ax = plt.subplots(n_row,n_col,figsize=(20,40))
#
#     labels=[]
#     for idx,(n_dynamic,N2) in enumerate(product(n_dynamic_li,N2_li)):
#         color = iter(colormap(np.linspace(0,1,12)))
#         for learning_rule in learning_rule_li:
#             if idx==0:
#                 labels.append("spiking"+learning_rule)
#             try:
#                 directory_name = dir_name2(learning_rule,N1,N2,n_dynamic,beta,batch_size)
#                 error = csv2array(directory_name,skiplines=2)
#                 # if len(error[1])<90:
#                 #     print("N2",N2)
#                 #     print("lr",learning_rule)
#                 #     print("dyn",n_dynamic)
#                 #     print("")
#                 ax[idx//n_col,idx%n_col].plot(error[1],c=next(color))
#             except:
#                 ax[idx//n_col,idx%n_col].plot([0],c=next(color))
#                 print(directory_name)
#         ax[idx//n_col,idx%n_col].set_xlabel('Epoch')
#         ax[idx//n_col,idx%n_col].set_ylabel('Test error rate (%)')
#         ax[idx//n_col,idx%n_col].set_xlim([0,200])
#         ax[idx//n_col,idx%n_col].set_ylim([0,20])
#         ax[idx//n_col,idx%n_col].grid(axis='y')
#         ax[idx//n_col,idx%n_col].set_title('n dyn = '+str(n_dynamic)+', N2 = '+str(N2))
#     fig.suptitle('Test Error (%)\n'+r'$N_1=$'+str(N1)+r', $N_2=$1'+str(N2)+'\n'+r'$\beta=$'+str(beta))
#     fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
#     fig.savefig('beta_'+str(int(10*beta))+'.png',bbox_inches="tight")
#
#
#
# ###################################################################################################
# # learning_rule_li = ['cepalt','skewsym','stdp_slow','stdp_med']
# #
# # n_col,n_row=3,4
# # # batch_size=200
# # N1=100
# # # N2=30
# # batch_size=200
# # for beta in [1.5]:
# #     fig, ax = plt.subplots(n_row,n_col,figsize=(20,40))
# #
# #     labels=[]
# #     for idx,(n_dynamic,N2) in enumerate(product(n_dynamic_li,N2_li)):
# #         color = iter(colormap(np.linspace(0,1,12)))
# #         for learning_rule in learning_rule_li:
# #             if idx==0:
# #                 labels.append("spiking"+learning_rule)
# #             try:
# #                 directory_name = dir_name2(learning_rule,N1,N2,n_dynamic,beta,batch_size)
# #                 error = csv2array(directory_name,skiplines=2)
# #                 # if len(error[1])<90:
# #                 #     print("N2",N2)
# #                 #     print("lr",learning_rule)
# #                 #     print("dyn",n_dynamic)
# #                 #     print("")
# #                 ax[idx//n_col,idx%n_col].plot(error[1],c=next(color))
# #             except:
# #                 ax[idx//n_col,idx%n_col].plot([0],c=next(color))
# #                 print(directory_name)
# #         ax[idx//n_col,idx%n_col].set_xlabel('Epoch')
# #         ax[idx//n_col,idx%n_col].set_ylabel('Test error rate (%)')
# #         ax[idx//n_col,idx%n_col].set_xlim([0,200])
# #         ax[idx//n_col,idx%n_col].set_ylim([0,20])
# #         ax[idx//n_col,idx%n_col].grid(axis='y')
# #         ax[idx//n_col,idx%n_col].set_title('n dyn = '+str(n_dynamic)+', N2 = '+str(N2))
# #     fig.suptitle('Test Error (%)\n'+r'$N_1=$'+str(N1)+r', $N_2=$1'+str(N2)+'\n'+r'$\beta=$'+str(beta))
# #     fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# #     fig.savefig('beta_'+str(int(10*beta))+'.png',bbox_inches="tight")
#
#
# #########################################################################################
#
# ###################################################################################################
# learning_rule_li = ['cepalt','skewsym','stdp_rock','stdp_slug','stdp_slow','stdp_med','stdp_fast']
#
# n_col,n_row=3,8
# # batch_size=200
# N1=100
# # N2=30
# # n_dynamic=16
# batch_size=200
# # for beta in [1.5,0.9,0.4]:
# for n_dynamic in [1,2,4,8,16]:
#     fig, ax = plt.subplots(n_row,n_col,figsize=(40,80))
#
#     labels=[]
#     for idx,(beta,N2) in enumerate(product([0.4,0.9,1.5,2,4,8,12,16],[15,30,60])):
#         color = iter(colormap(np.linspace(0,1,12)))
#         for learning_rule in learning_rule_li:
#             if idx==0:
#                 labels.append(learning_rule)
#             try:
#                 directory_name = dir_name2(learning_rule,N1,N2,n_dynamic,beta,batch_size)
#                 error = csv2array(directory_name,skiplines=2)
#                 # if len(error[1])<90:
#                 #     print("N2",N2)
#                 #     print("lr",learning_rule)
#                 #     print("dyn",n_dynamic)
#                 #     print("")
#                 ax[idx//n_col,idx%n_col].plot(error[1],c=next(color))
#             except:
#                 ax[idx//n_col,idx%n_col].plot([0],c=next(color))
#                 print(directory_name)
#         ax[idx//n_col,idx%n_col].set_xlabel('Epoch')
#         ax[idx//n_col,idx%n_col].set_ylabel('Test error rate (%)')
#         ax[idx//n_col,idx%n_col].set_xlim([0,200])
#         ax[idx//n_col,idx%n_col].set_ylim([0,20])
#         ax[idx//n_col,idx%n_col].grid(axis='y')
#         ax[idx//n_col,idx%n_col].set_title(r'$\beta = $'+str(beta)+', N2 = '+str(N2))
#     fig.suptitle('Test Error (%)\n'+r'$N_1=$'+str(N1)+r', $N_2=$1'+str(N2)+'\n'+r'$\beta=$'+str(beta))
#     fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
#     fig.savefig('beta_compare_'+str(n_dynamic)+'.png',bbox_inches="tight")
#
#
# #MAIN
# ###################################################################################################
# learning_rule_li = ['cep','cepalt','skew','skewsym','stdp_0','stdp_1','stdp_2','stdp_3']
#
# n_col,n_row=2,4
# N1=64
# N2=64
# batch_size=200
# n_dynamic=8
# fig, ax = plt.subplots(n_row,n_col,figsize=(40,40))
#
# labels=[]
# for idx,(beta,batch_size) in enumerate(product([1,2,4,8],[20,200])):
#     color = iter(colormap(np.linspace(0,1,12)))
#     for learning_rule in learning_rule_li:
#         if idx==0:
#             labels.append(learning_rule)
#         try:
#             directory_name = dir_name2(learning_rule,N1,N2,n_dynamic,beta,batch_size)
#             error = csv2array(directory_name,skiplines=2)
#             ax[idx//n_col,idx%n_col].plot(error[1],c=next(color))
#         except:
#             ax[idx//n_col,idx%n_col].plot([0],c=next(color))
#             print(directory_name)
#     ax[idx//n_col,idx%n_col].set_xlabel('Epoch')
#     ax[idx//n_col,idx%n_col].set_ylabel('Test error rate (%)')
#     ax[idx//n_col,idx%n_col].set_xlim([0,200])
#     ax[idx//n_col,idx%n_col].set_ylim([0,20])
#     ax[idx//n_col,idx%n_col].grid(axis='y')
#     ax[idx//n_col,idx%n_col].set_title(r'$\beta = $'+str(beta)+', N2 = '+str(N2))
# fig.suptitle('Test Error (%)\n'+r'$N_1=$'+str(N1)+r', $N_2=$1'+str(N2)+'\n'+r'$\beta=$'+str(beta))
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('main_result_200.png',bbox_inches="tight")
#
# # ###################################################################################################
# #

fig, ax = plt.subplots(2,figsize=(20,40))
labels=['nonspiking_skewsym','stdp','nonspiking_cep','spiking_cep']
for idx,param_setting in enumerate(['old','new']):
    color = iter(colormap(np.linspace(0,1,12)))
    for label in labels:
        directory_name = label+"_"+param_setting
        try:
            error = csv2array(directory_name,skiplines=2)
            ax[idx].plot(error[1],c=next(color))
        except:
            ax[idx].plot([0],c=next(color))
            print(directory_name)
        ax[idx].set_xlabel('Epoch')
        ax[idx].set_ylabel('Test error rate (%)')
        ax[idx].set_xlim([0,300])
        ax[idx].set_ylim([0,20])
        ax[idx].grid(visible=True,axis='y')
        ax[idx].set_title('Parameters = '+param_setting)
fig.suptitle('Test Error (%)')
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('results.png',bbox_inches="tight")


#####################################################################################################33
# learning_rule_li = ['cepalt','skewsym','stdp_rock','stdp_slug','stdp_slow','stdp_med']
#
# batch_size=200
# # for beta in [1.5,0.9,0.4]:
# n_dynamic=8
# fig, ax = plt.subplots(3,3,figsize=(40,40))
# beta=3
# N2=40
# labels=[]
# for idx,N1 in enumerate([40,50,60,70,80,90,100,110,120]):
#     color = iter(colormap(np.linspace(0,1,12)))
#     for learning_rule in learning_rule_li:
#         if idx==0:
#             labels.append(learning_rule)
#         try:
#             directory_name = dir_name2(learning_rule,N1,N2,n_dynamic,beta,batch_size)
#             error = csv2array(directory_name,skiplines=2)
#             ax[idx//n_col,idx%n_col].plot(error[1],c=next(color))
#         except:
#             ax[idx//n_col,idx%n_col].plot([0],c=next(color))
#             print(directory_name)
#     ax[idx//n_col,idx%n_col].set_xlabel('Epoch')
#     ax[idx//n_col,idx%n_col].set_ylabel('Test error rate (%)')
#     ax[idx//n_col,idx%n_col].set_xlim([0,110])
#     ax[idx//n_col,idx%n_col].set_ylim([0,10])
#     ax[idx//n_col,idx%n_col].grid(axis='y')
#     ax[idx//n_col,idx%n_col].set_title('N1 = '+str(N1))
# fig.suptitle('Test Error (%)\n'+r'$N_2=$1'+str(N2)+'\n'+r'$\beta=$'+str(beta))
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('compare_N1.png',bbox_inches="tight")

###################################################################################################

# learning_rule_li = ['cepalt','skewsym','stdp_slug','stdp_slow','stdp_med']
#
# batch_size=200
# # for beta in [1.5,0.9,0.4]:
# n_dynamic=8
# fig, ax = plt.subplots(3,3,figsize=(40,40))
# beta=1
# N1=8*n_dynamic
# labels=[]
# for idx,N2 in enumerate([8*k for k in range(2,9)]):
#     color = iter(colormap(np.linspace(0,1,12)))
#     for learning_rule in learning_rule_li:
#         if idx==0:
#             labels.append(learning_rule)
#         try:
#             directory_name = dir_name2(learning_rule,N1,N2,n_dynamic,beta,batch_size)
#             error = csv2array(directory_name,skiplines=2)
#             ax[idx//n_col,idx%n_col].plot(error[1],c=next(color))
#         except:
#             ax[idx//n_col,idx%n_col].plot([0],c=next(color))
#             print(directory_name)
#     ax[idx//n_col,idx%n_col].set_xlabel('Epoch')
#     ax[idx//n_col,idx%n_col].set_ylabel('Test error rate (%)')
#     ax[idx//n_col,idx%n_col].set_xlim([0,100])
#     ax[idx//n_col,idx%n_col].set_ylim([0,20])
#     ax[idx//n_col,idx%n_col].grid(axis='y')
#     ax[idx//n_col,idx%n_col].set_title('N2 = '+str(N2))
# fig.suptitle('Test Error (%)\n'+r'$N_2=$1'+str(N2)+'\n'+r'$\beta=$'+str(beta))
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('compare_N2_beta='+str(beta)+'.png',bbox_inches="tight")





# # # Universal
# # #################################################################################
# # fig, ax = plt.subplots(6,figsize=(20,40))
# # batch_size=25
# # max_fr=3
# # tau_dynamic=3
# # beta=0.2
# # labels=[]
# # for idx,(batch_size,N2) in enumerate(product(batch_size_li,N2_li)):
# #     step = 12./N2
# #     for activity_type,learning_rule in product(activity_type_li,learning_rule_li):
# #         if idx==0:
# #             labels.append(r''+activity_type+" "+learning_rule)
# #         directory_name = dir_name(activity_type,learning_rule,max_fr,N2,tau_dynamic,beta,batch_size)
# #         error = csv2array(directory_name,skiplines=2)
# #         ax[idx].plot(error[1])
# #     ax[idx].set_xlabel('Epoch')
# #     ax[idx].set_ylabel('Test error rate (%)')
# #     ax[idx].set_xlim([0,30])
# #     ax[idx].set_ylim([0,20])
# #     ax[idx].grid(axis='y')
# #     ax[idx].set_title('step size = '+str(step)+'\n batch size = '+str(batch_size))
# # fig.suptitle(r'Batch size='+str(batch_size)+", max f.r.="+str(max_fr))
# # fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# # fig.savefig('test1.png',bbox_inches="tight")
# # #################################################################################
#
# n_col,n_row=2,2
# fig, ax = plt.subplots(n_row,n_col,figsize=(20,20))
# # batch_size=200
# max_fr=3
# tau_dynamic=3
# beta=0.9
# labels=[]
# for idx,(batch_size,N2) in enumerate(product([25,200],N2_li)):
#     step = 12./N2
#     for activity_type,learning_rule in product(activity_type_li,learning_rule_li):
#         if idx==0:
#             labels.append(r''+activity_type+" "+learning_rule)
#         directory_name = dir_name(activity_type,learning_rule,max_fr,N2,tau_dynamic,beta,batch_size)
#         error = csv2array(directory_name,skiplines=2)
#         ax[idx//n_col,idx%n_col].plot(error[1])
#     ax[idx//n_col,idx%n_col].set_xlabel('Epoch')
#     ax[idx//n_col,idx%n_col].set_ylabel('Test error rate (%)')
#     ax[idx//n_col,idx%n_col].set_xlim([0,100])
#     ax[idx//n_col,idx%n_col].set_ylim([0,25])
#     ax[idx//n_col,idx%n_col].grid(axis='y')
#     ax[idx//n_col,idx%n_col].set_title('step size = '+str(step)+'\n batch size = '+str(batch_size))
# fig.suptitle('Test Error (%)\n'+r'$T_1=36, T_2=12$'+'\n max f.r. = '+str(max_fr)+'\n'+r'$\beta=$'+str(beta))
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('general_beta=02.png',bbox_inches="tight")
#
#
# n_col,n_row=4,2
# fig, ax = plt.subplots(n_row,n_col,figsize=(60,20))
# # batch_size=200
# max_fr=3
# tau_dynamic=3
# beta=0.2
# labels=[]
# for idx,(batch_size,N2) in enumerate(product([25,50,100,200],N2_li)):
#     step = 12./N2
#     for activity_type,learning_rule in product(activity_type_li,learning_rule_li):
#         if idx==0:
#             labels.append(r''+activity_type+" "+learning_rule)
#         directory_name = dir_name(activity_type,learning_rule,max_fr,N2,tau_dynamic,beta,batch_size)
#         error = csv2array(directory_name,skiplines=2)
#         ax[idx//n_col,idx%n_col].plot(error[1])
#     ax[idx//n_col,idx%n_col].set_xlabel('Epoch')
#     ax[idx//n_col,idx%n_col].set_ylabel('Test error rate (%)')
#     ax[idx//n_col,idx%n_col].set_xlim([0,100])
#     ax[idx//n_col,idx%n_col].set_ylim([0,25])
#     ax[idx//n_col,idx%n_col].grid(axis='y')
#     ax[idx//n_col,idx%n_col].set_title('step size = '+str(step)+'\n batch size = '+str(batch_size))
# fig.suptitle('Test Error (%)\n'+r'$T_1=36, T_2=12$'+'\n max f.r. = '+str(max_fr)+'\n'+r'$\beta=$'+str(beta))
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('general_beta=09.png',bbox_inches="tight")
#
#





#
#
# # Plot various levels of discretization for nonspiking cepalt
# n_plots = 12
# fig, ax = plt.subplots(figsize=(20,10))
# color = iter(colormap(np.linspace(0,1,12)))
# step=[5./k for k in range(1,n_plots+1)]
# labels=['step = '+'{:.2f}'.format(step[i]) for i in range(n_plots)]
# disc_test_dir = ["cepalt_nonspiking_"+str(i) for i in range(n_plots)]
# error = [csv2array(disc_test_dir[i],skiplines=2) for i in range(n_plots)]
# for i in range(n_plots):
#     ax.plot(error[i][1],c=next(color))
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Test error rate (%)')
# ax.set_xlim([0,30])
# ax.set_ylim([0,20])
# ax.grid(axis='y')
# # ax.set_title()
# fig.suptitle(r"Nonspiking Discretization schemes ($T_1=45,T_2=15,\beta=0.2$,max f.r.=3.0)")
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('nonspiking_discretization.png',bbox_inches="tight")
#
# # Plot various levels of discretization for nonspiking cepalt
# fig, ax = plt.subplots(figsize=(20,10))
# color = iter(colormap(np.linspace(0,1,12)))
# labels=['step = '+'{:.2f}'.format(step[i]) for i in range(n_plots)]
# disc_test_dir = ["cepalt_spiking_"+str(i) for i in range(n_plots)]
# error = [csv2array(disc_test_dir[i],skiplines=2) for i in range(n_plots)]
# for i in range(n_plots):
#     ax.plot(error[i][1],c=next(color))
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Test error rate (%)')
# ax.set_xlim([0,30])
# ax.set_ylim([0,20])
# ax.grid(axis='y')
# fig.suptitle(r"Spiking Discretization schemes ($T_1=45,T_2=15,\beta=0.2,max f.r.=3.0)")
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('spiking_discretization.png',bbox_inches="tight")
#
#
# n_plots=12
# # Spiking vs nonspiking cepalt for various levels of discretization
# fig, ax = plt.subplots(n_plots//3,3,figsize=(20,20))
# color = iter(colormap(np.linspace(0,1,12)))
# labels=['spiking','nonspiking']
# N2 = [3*i for i in range(1,n_plots+1)]
# for i in range(n_plots):
#     spiking_dir = 'cepalt_spiking_'+str(i)
#     nonspiking_dir = 'cepalt_nonspiking_'+str(i)
#     spiking_train_error, spiking_test_error = csv2array(spiking_dir,skiplines=2)
#     nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir,skiplines=2)
#     ax[i//3,i%3].plot(spiking_test_error)
#     ax[i//3,i%3].plot(nonspiking_test_error)
#     ax[i//3,i%3].set_xlabel('Epoch')
#     ax[i//3,i%3].set_ylabel('Test error rate (%)')
#     ax[i//3,i%3].set_xlim([0,30])
#     ax[i//3,i%3].set_ylim([0,20])
#     ax[i//3,i%3].grid(axis='y')
#     ax[i//3,i%3].set_title('step = '+'{:.2f}'.format(15./N2[i]))
# fig.suptitle(r"Test error for spiking and nonspiking dynamics ($\beta = 0.2,T_1=45,T_2=15,max f.r.=3.0$)",fontsize=20)
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('cepalt_discretization.png')#,bbox_inches="tight")
#
#
# # Cepalt vs skewsym
#
# fig, ax = plt.subplots(figsize=(20,10))
# color = iter(colormap(np.linspace(0,1,12)))
# labels=["spiking cepalt", "nonspiking cepalt", "spiking skewsym","nonspiking skewsym"]
# dirs = ["spiking_cepalt", "nonspiking_cepalt", "spiking_skewsym","nonspiking_skewsym"]
# error = [csv2array(dir,skiplines=2) for dir in dirs]
# for i in range(4):
#     ax.plot(error[i][1])
# ax.set_xlabel('Epoch')
# ax.set_ylabel('Test error rate (%)')
# ax.set_xlim([0,50])
# ax.set_ylim([0,15])
# # ax.set_title()
# fig.suptitle('Cepalt vs skewsym / spiking vs nonspiking \n'+r'$T_1=40, T_2=15$'+'\n step = 1.0 \n max f.r. = 1.0')
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('cepalt_vs_skewsym.png',bbox_inches="tight")
#
# #
# # fig, ax = plt.subplots(2,4,figsize=(20,10))
# # labels=['spiking','nonspiking','stdp']
# # N2 = [3*i for i in range(1,11)]
# # for i in range(8):
# #     spiking_dir = 'skewsym_spiking_'+str(i)
# #     nonspiking_dir = 'skewsym_nonspiking_'+str(i)
# #     stdp_dir = 'stdp_'+str(i)
# #     spiking_train_error, spiking_test_error = csv2array(spiking_dir)
# #     nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
# #     stdp_train_error,stdp_test_error = csv2array(stdp_dir)
# #     ax[i//4,i%4].plot(spiking_test_error)
# #     ax[i//4,i%4].plot(nonspiking_test_error)
# #     ax[i//4,i%4].plot(stdp_test_error)
# #     ax[i//4,i%4].set_xlabel('Epoch')
# #     ax[i//4,i%4].set_ylabel('Test error rate (%)')
# #     ax[i//4,i%4].set_xlim([0,30])
# #     ax[i//4,i%4].set_title(r'$N_1=$'+str(3*N2[i])+', $N_2=$'+str(N2[i])+', dt = '+'{:.2f}'.format(1-(2**(-20/(3*N2[i])))))
# # fig.suptitle("Comparison of three approximately equivalent dynamics")
# # fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# # fig.savefig('skew_error.png',bbox_inches="tight")
#
#
# fig, ax = plt.subplots(2,2,figsize=(20,10))
# labels=['spiking',r'spiking stdp: $\tau=2$',r'spiking stdp: $\tau=1.44$ ',r'spiking stdp: $\tau=1.0$','nonspiking',r'nonspiking stdp: $\tau=2$',r'nonspiking stdp: $\tau=1.44$ ',r'nonspiking stdp: $\tau=1.0$']
# batch_size=[25,50,100,200]
# for i in [0,1,2,3]:
#
#     spiking_dir = 'spiking_skewsym_'+str(i)
#     spiking_stdp_slow_dir = 'spiking_stdp_slow_'+str(i)
#     spiking_stdp_med_dir = 'spiking_stdp_med_'+str(i)
#     spiking_stdp_fast_dir = 'spiking_stdp_fast_'+str(i)
#     nonspiking_dir = 'nonspiking_skewsym_'+str(i)
#     nonspiking_stdp_slow_dir = 'nonspiking_stdp_slow_'+str(i)
#     nonspiking_stdp_med_dir = 'nonspiking_stdp_med_'+str(i)
#     nonspiking_stdp_fast_dir = 'nonspiking_stdp_fast_'+str(i)
#
#     spiking_train_error, spiking_test_error = csv2array(spiking_dir,skiplines=2)
#     spiking_stdp_slow_train_error,spiking_stdp_slow_test_error = csv2array(spiking_stdp_slow_dir,skiplines=2)
#     spiking_stdp_med_train_error,spiking_stdp_med_test_error = csv2array(spiking_stdp_med_dir,skiplines=2)
#     spiking_stdp_fast_train_error,spiking_stdp_fast_test_error = csv2array(spiking_stdp_fast_dir,skiplines=2)
#     nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir,skiplines=2)
#     nonspiking_stdp_slow_train_error,nonspiking_stdp_slow_test_error = csv2array(nonspiking_stdp_slow_dir,skiplines=2)
#     nonspiking_stdp_med_train_error,nonspiking_stdp_med_test_error = csv2array(nonspiking_stdp_med_dir,skiplines=2)
#     nonspiking_stdp_fast_train_error,nonspiking_stdp_fast_test_error = csv2array(nonspiking_stdp_fast_dir,skiplines=2)
#
#     ax[i//2,i%2].plot(spiking_test_error)
#     ax[i//2,i%2].plot(spiking_stdp_slow_test_error)
#     ax[i//2,i%2].plot(spiking_stdp_med_test_error)
#     ax[i//2,i%2].plot(spiking_stdp_fast_test_error)
#     ax[i//2,i%2].plot(nonspiking_test_error)
#     ax[i//2,i%2].plot(nonspiking_stdp_slow_test_error)
#     ax[i//2,i%2].plot(nonspiking_stdp_med_test_error)
#     ax[i//2,i%2].plot(nonspiking_stdp_fast_test_error)
#     ax[i//2,i%2].set_xlabel('Epoch')
#     ax[i//2,i%2].set_ylabel('Test error rate (%)')
#     ax[i//2,i%2].set_xlim([0,30])
#     ax[i//2,i%2].set_title('batch size = '+str(batch_size[i]))
# fig.suptitle("Comparison of batch sizes\n"+r'$T_1=40,T_2=15$'+'\nstep = 1.0\nmax f.r. = 1.0\n'+r'$\beta=1.0$')
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('compare_batch_size.png',bbox_inches="tight")
#
#
# # fig, ax = plt.subplots(2,2,figsize=(20,10))
# # color = iter(colormap(np.linspace(0,1,12)))
# # labels=['spiking','nonspiking',r'stdp: \tau=2',r'stdp: \tau=1.44 ',r'stdp: \tau=1.0']
# # batch_size=[200,100,20,40]
# # for i in [0,1,2,3]:
# #     spiking_dir = 'spiking_skewsym_'+str(i)
# #     nonspiking_dir = 'nonspiking_skewsym_'+str(i)
# #     stdp_slow_dir = 'stdp_slow_'+str(i)
# #     stdp_med_dir = 'stdp_med_'+str(i)
# #     stdp_fast_dir = 'stdp_fast_'+str(i)
# #     spiking_train_error, spiking_test_error = csv2array(spiking_dir,skiplines=2)
# #     nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir,skiplines=2)
# #     stdp_slow_train_error,stdp_slow_test_error = csv2array(stdp_slow_dir,skiplines=2)
# #     stdp_med_train_error,stdp_med_test_error = csv2array(stdp_med_dir,skiplines=2)
# #     stdp_fast_train_error,stdp_fast_test_error = csv2array(stdp_fast_dir,skiplines=2)
# #     ax[i//2,i%2].plot(spiking_test_error)
# #     ax[i//2,i%2].plot(nonspiking_test_error)
# #     ax[i//2,i%2].plot(stdp_slow_test_error)
# #     ax[i//2,i%2].plot(stdp_med_test_error)
# #     ax[i//2,i%2].plot(stdp_fast_test_error)
# #     ax[i//2,i%2].set_xlabel('Epoch')
# #     ax[i//2,i%2].set_ylabel('Test error rate (%)')
# #     ax[i//2,i%2].set_xlim([0,30])
# #     ax[i//2,i%2].set_title('batch size = '+str(batch_size[i]))
# # fig.suptitle(r"Comparison of trace decay rates ($N_1=40,N_2=15,\beta=0.9$)")
# # fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# # fig.savefig('kernel_width_compare.png',bbox_inches="tight")
#
# # fig, ax = plt.subplots(figsize=(20,10))
# # labels=['spiking','nonspiking','stdp: trace decay = 0.9','stdp: trace decay = 0.7','stdp: trace decay = 0.5','stdp: trace decay = 0.4']
# # beta=[0.9,0.7,0.5,0.4]
# # i=1
# # spiking_dir = 'skewsym_spiking_b_'+str(i)
# # nonspiking_dir = 'skewsym_nonspiking_b_'+str(i)
# # stdp_0_dir = 'stdp_0_'+str(i)
# # stdp_1_dir = 'stdp_1_'+str(i)
# # stdp_2_dir = 'stdp_2_'+str(i)
# # stdp_3_dir = 'stdp_3_'+str(i)
# # spiking_train_error, spiking_test_error = csv2array(spiking_dir)
# # nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
# # stdp_0_train_error,stdp_0_test_error = csv2array(stdp_0_dir)
# # stdp_1_train_error,stdp_1_test_error = csv2array(stdp_1_dir)
# # stdp_2_train_error,stdp_2_test_error = csv2array(stdp_2_dir)
# # stdp_3_train_error,stdp_3_test_error = csv2array(stdp_3_dir)
# # ax.plot(spiking_test_error)
# # ax.plot(nonspiking_test_error)
# # ax.plot(stdp_0_test_error)
# # ax.plot(stdp_1_test_error)
# # ax.plot(stdp_2_test_error)
# # ax.plot(stdp_3_test_error)
# # ax.set_xlabel('Epoch')
# # ax.set_ylabel('Test error rate (%)')
# # ax.set_xlim([0,20])
# # # ax.set_title()
# # fig.suptitle(r"Comparison of trace decay rates ($N_1=40,N_2=15,dt=0.3,\beta=$"+str(beta[i])+")")
# # fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# # fig.savefig('decay_compare.png',bbox_inches="tight")
#
#
# # Old version; delete
# ###########################################3
# # fig, ax = plt.subplots(3,3,figsize=(25,20))
# # # fig.tight_layout()
# # labels=['cepalt','skewsym']
# # for idx1,beta in enumerate([1.0,0.5,0.2]):
# #     for idx2,N2 in enumerate([5,10,15]):
# #         i = idx1*3+idx2
# #         cepalt_dir = 'cepalt_'+str(i)
# #         skewsym_dir = 'skewsym_'+str(i)
# #         ceptalt_train_error, cepalt_test_error = csv2array(cepalt_dir)
# #         skewsym_train_error,skewsym_test_error = csv2array(skewsym_dir)
# #         ax[idx1,idx2].plot(cepalt_test_error)
# #         ax[idx1,idx2].plot(skewsym_test_error)
# #         ax[idx1,idx2].set_xlabel('Epoch')
# #         ax[idx1,idx2].set_ylabel('Test error rate (%)')
# #         ax[idx1,idx2].set_xlim([0,20])
# #         ax[idx1,idx2].set_ylim([0,20])
# #         ax[idx1,idx2].set_title(r'$\beta = $' +str(beta)+', $N_1=$'+str(3*N2)+', $N_2=$'+str(N2)+', $dt = $'+'{:.2f}'.format(1-(2**(-20/(3*N2)))) )
# # fig.suptitle(r"CEP vs skewsym for nonspiking dynamics",fontsize=20)
# # fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# # fig.savefig('cepalt_vs_skewsym.png')#,bbox_inches="tight")
# ###################################################
#
# # fig, ax = plt.subplots(figsize=(20,10))
# # N1=[3*N2 for N2 in [3,6,9,12,15]]
# # labels=[r'N_1='+str(N1[i]) for i in range(5)]
# # disc_test_dir = ["disc_test_"+str(i) for i in range(5)]
# # error = [csv2array(disc_test_dir[i]) for i in range(5)]
# # for i in range(5):
# #     ax.plot(error[i][1])
# # ax.set_xlabel('Epoch')
# # ax.set_ylabel('Test error rate (%)')
# # ax.set_xlim([0,20])
# # # ax.set_title()
# # fig.suptitle(r"Discretization schemes")
# # fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# # fig.savefig('discretization.png',bbox_inches="tight")
