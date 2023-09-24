import numpy as np
from matplotlib import pyplot as plt
import csv
import argparse

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

def csv2array(directory):
    with open(directory+'/results.csv','r',newline='') as csv_file:
        csv_reader = csv.reader(csv_file)
        error = np.array(list(csv_reader)).astype('float')
        train_error,test_error = error[:,0],error[:,1]
    return train_error, test_error

#
# fig, ax = plt.subplots(2,5,figsize=(25,10))
# # fig.tight_layout()
# labels=['spiking','nonspiking']
# dt = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
# for i in range(10):
#     spiking_dir = 'cepalt_spiking_'+str(i)
#     nonspiking_dir = 'cepalt_nonspiking_'+str(i)
#     spiking_train_error, spiking_test_error = csv2array(spiking_dir)
#     nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
#     ax[i//5,i%5].plot(spiking_test_error)
#     ax[i//5,i%5].plot(nonspiking_test_error)
#     ax[i//5,i%5].set_xlabel('Epoch')
#     ax[i//5,i%5].set_ylabel('Test error rate (%)')
#     ax[i//5,i%5].set_xlim([0,30])
#     ax[i//5,i%5].set_ylim([0,20])
#     ax[i//5,i%5].set_title('dt = '+str(dt[i]))
# fig.suptitle(r"Test error for spiking and non-spiking dynamics ($\beta = 0.2,N_1=40,N_2=15$)",fontsize=20)
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('cepalt_error.png')#,bbox_inches="tight")


fig, ax = plt.subplots(figsize=(20,10))
N1=[3*Kmax for Kmax in [3,6,9,12,15,18]]
labels=[r'N_1='+str(N1[i]) for i in range(5)]
disc_test_dir = ["cepalt_nonspiking_"+str(i) for i in range(5)]
error = [csv2array(disc_test_dir[i]) for i in range(5)]
for i in range(5):
    ax.plot(error[i][1])
ax.set_xlabel('Epoch')
ax.set_ylabel('Test error rate (%)')
ax.set_xlim([0,20])
# ax.set_title()
fig.suptitle(r"Discretization schemes")
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('discretization.png',bbox_inches="tight")


fig, ax = plt.subplots(2,3,figsize=(20,10))
# fig.tight_layout()
labels=['spiking','nonspiking']
N2 = [3*i for i in range(1,7)]
for i in range(6):
    spiking_dir = 'cepalt_spiking_'+str(i)
    nonspiking_dir = 'cepalt_nonspiking_'+str(i)
    spiking_train_error, spiking_test_error = csv2array(spiking_dir)
    nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
    ax[i//3,i%3].plot(spiking_test_error)
    ax[i//3,i%3].plot(nonspiking_test_error)
    ax[i//3,i%3].set_xlabel('Epoch')
    ax[i//3,i%3].set_ylabel('Test error rate (%)')
    ax[i//3,i%3].set_xlim([0,30])
    ax[i//3,i%3].set_ylim([0,20])
    ax[i//3,i%3].set_title(r'$N_1=$'+str(3*N2[i])+', $N_2=$'+str(N2[i])+', dt = '+'{:.2f}'.format(1-(2**(-20/(3*N2[i])))))
fig.suptitle(r"Test error for spiking and nonspiking dynamics ($\beta = 0.2$)",fontsize=20)
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('cepalt_error_b.png')#,bbox_inches="tight")




fig, ax = plt.subplots(2,4,figsize=(20,10))
labels=['spiking','nonspiking','stdp']

for i in range(8):
    spiking_dir = 'skewsym_spiking_'+str(i)
    nonspiking_dir = 'skewsym_nonspiking_'+str(i)
    stdp_dir = 'stdp_'+str(i)
    spiking_train_error, spiking_test_error = csv2array(spiking_dir)
    nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
    stdp_train_error,stdp_test_error = csv2array(stdp_dir)
    ax[i//4,i%4].plot(spiking_test_error)
    ax[i//4,i%4].plot(nonspiking_test_error)
    ax[i//4,i%4].plot(stdp_test_error)
    ax[i//4,i%4].set_xlabel('Epoch')
    ax[i//4,i%4].set_ylabel('Test error rate (%)')
    ax[i//4,i%4].set_xlim([0,30])
    ax[i//4,i%4].set_title(r'$N_1=$'+str(3*N2[i])+', $N_2=$'+str(N2[i])+', dt = '+'{:.2f}'.format(1-(2**(-20/(3*N2[i])))))
fig.suptitle("Comparison of three approximately equivalent dynamics")
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('skew_error.png',bbox_inches="tight")


# fig, ax = plt.subplots(2,2,figsize=(20,10))
# labels=['spiking','nonspiking','stdp: trace decay = 0.9','stdp: trace decay = 0.7','stdp: trace decay = 0.5','stdp: trace decay = 0.4']
# beta=[0.9,0.7,0.5,0.4]
# for i in range(1):
#     spiking_dir = 'skewsym_spiking_b_'+str(i)
#     nonspiking_dir = 'skewsym_nonspiking_b_'+str(i)
#     stdp_0_dir = 'stdp_0_'+str(i)
#     stdp_1_dir = 'stdp_1_'+str(i)
#     stdp_2_dir = 'stdp_2_'+str(i)
#     stdp_3_dir = 'stdp_3_'+str(i)
#     spiking_train_error, spiking_test_error = csv2array(spiking_dir)
#     nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
#     stdp_0_train_error,stdp_0_test_error = csv2array(stdp_0_dir)
#     stdp_1_train_error,stdp_1_test_error = csv2array(stdp_1_dir)
#     stdp_2_train_error,stdp_2_test_error = csv2array(stdp_2_dir)
#     stdp_3_train_error,stdp_3_test_error = csv2array(stdp_3_dir)
#     ax[i//4,i%4].plot(spiking_test_error)
#     ax[i//4,i%4].plot(nonspiking_test_error)
#     ax[i//4,i%4].plot(stdp_0_test_error)
#     ax[i//4,i%4].plot(stdp_1_test_error)
#     ax[i//4,i%4].plot(stdp_2_test_error)
#     ax[i//4,i%4].plot(stdp_3_test_error)
#     ax[i//4,i%4].set_xlabel('Epoch')
#     ax[i//4,i%4].set_ylabel('Test error rate (%)')
#     ax[i//4,i%4].set_xlim([0,20])
#     ax[i//4,i%4].set_title(r'$\beta=$'+str(beta[i]))
# fig.suptitle(r"Comparison of trace decay rates ($N_1=40,N_2=15,dt=0.3$)")
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('decay_compare.png',bbox_inches="tight")

fig, ax = plt.subplots(figsize=(20,10))
labels=['spiking','nonspiking','stdp: trace decay = 0.9','stdp: trace decay = 0.7','stdp: trace decay = 0.5','stdp: trace decay = 0.4']
beta=[0.9,0.7,0.5,0.4]
i=1
spiking_dir = 'skewsym_spiking_b_'+str(i)
nonspiking_dir = 'skewsym_nonspiking_b_'+str(i)
stdp_0_dir = 'stdp_0_'+str(i)
stdp_1_dir = 'stdp_1_'+str(i)
stdp_2_dir = 'stdp_2_'+str(i)
stdp_3_dir = 'stdp_3_'+str(i)
spiking_train_error, spiking_test_error = csv2array(spiking_dir)
nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
stdp_0_train_error,stdp_0_test_error = csv2array(stdp_0_dir)
stdp_1_train_error,stdp_1_test_error = csv2array(stdp_1_dir)
stdp_2_train_error,stdp_2_test_error = csv2array(stdp_2_dir)
stdp_3_train_error,stdp_3_test_error = csv2array(stdp_3_dir)
ax.plot(spiking_test_error)
ax.plot(nonspiking_test_error)
ax.plot(stdp_0_test_error)
ax.plot(stdp_1_test_error)
ax.plot(stdp_2_test_error)
ax.plot(stdp_3_test_error)
ax.set_xlabel('Epoch')
ax.set_ylabel('Test error rate (%)')
ax.set_xlim([0,20])
# ax.set_title()
fig.suptitle(r"Comparison of trace decay rates ($N_1=40,N_2=15,dt=0.3,\beta=$"+str(beta[i])+")")
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('decay_compare.png',bbox_inches="tight")


fig, ax = plt.subplots(3,3,figsize=(25,20))
# fig.tight_layout()
labels=['cepalt','skewsym']
for idx1,beta in enumerate([1.0,0.5,0.2]):
    for idx2,Kmax in enumerate([5,10,15]):
        i = idx1*3+idx2
        cepalt_dir = 'cepalt_'+str(i)
        skewsym_dir = 'skewsym_'+str(i)
        ceptalt_train_error, cepalt_test_error = csv2array(cepalt_dir)
        skewsym_train_error,skewsym_test_error = csv2array(skewsym_dir)
        ax[idx1,idx2].plot(cepalt_test_error)
        ax[idx1,idx2].plot(skewsym_test_error)
        ax[idx1,idx2].set_xlabel('Epoch')
        ax[idx1,idx2].set_ylabel('Test error rate (%)')
        ax[idx1,idx2].set_xlim([0,20])
        ax[idx1,idx2].set_ylim([0,20])
        ax[idx1,idx2].set_title(r'$\beta = $' +str(beta)+', $N_1=$'+str(3*Kmax)+', $N_2=$'+str(Kmax)+', $dt = $'+'{:.2f}'.format(1-(2**(-20/(3*Kmax)))) )
fig.suptitle(r"CEP vs skewsym for nonspiking dynamics",fontsize=20)
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('cepalt_vs_skewsym.png')#,bbox_inches="tight")


fig, ax = plt.subplots(figsize=(20,10))
N1=[3*Kmax for Kmax in [3,6,9,12,15]]
labels=[r'N_1='+str(N1[i]) for i in range(5)]
disc_test_dir = ["disc_test_"+str(i) for i in range(5)]
error = [csv2array(disc_test_dir[i]) for i in range(5)]
for i in range(5):
    ax.plot(error[i][1])
ax.set_xlabel('Epoch')
ax.set_ylabel('Test error rate (%)')
ax.set_xlim([0,20])
# ax.set_title()
fig.suptitle(r"Discretization schemes")
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('discretization.png',bbox_inches="tight")
