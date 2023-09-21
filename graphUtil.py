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


fig, ax = plt.subplots(2,5,figsize=(22,10))
# fig.tight_layout()
labels=['spiking','nonspiking']
dt = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for i in range(10):
    spiking_dir = 'cepalt_spiking_'+str(i)
    nonspiking_dir = 'cepalt_nonspiking_'+str(i)
    spiking_train_error, spiking_test_error = csv2array(spiking_dir)
    nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
    ax[i//5,i%5].plot(spiking_test_error)
    ax[i//5,i%5].plot(nonspiking_test_error)
    ax[i//5,i%5].set_xlabel('Epoch')
    ax[i//5,i%5].set_ylabel('Test error rate (%)')
    ax[i//5,i%5].set_title('dt = '+str(dt[i]))
fig.suptitle(r"Comparison of spiking and non-spiking dynamics ($\beta = 0.2,N_1=40,N_2=15$)",fontsize=20)
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('cepalt_error.png')#,bbox_inches="tight")


fig, ax = plt.subplots(2,5,figsize=(22,10))
# fig.tight_layout()
labels=['spiking','nonspiking']
N2 = [3*i for i in range(1,11)]
for i in range(10):
    spiking_dir = 'cepalt_spiking_b_'+str(i)
    nonspiking_dir = 'cepalt_nonspiking_b_'+str(i)
    spiking_train_error, spiking_test_error = csv2array(spiking_dir)
    nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
    ax[i//5,i%5].plot(spiking_test_error)
    ax[i//5,i%5].plot(nonspiking_test_error)
    ax[i//5,i%5].set_xlabel('Epoch')
    ax[i//5,i%5].set_ylabel('Test error rate (%)')
    ax[i//5,i%5].set_title(r'$N_1=$'+str(3*N2[i])+', $N_2=$'+str(N2[i])+', dt = '+'{:.2f}'.format(1-(2**(-20/(3*N2[i])))))
fig.suptitle(r"Comparison of spiking and non-spiking dynamics ($\beta = 0.2$)",fontsize=20)
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('cepalt_error_b.png')#,bbox_inches="tight")



#
# fig, ax = plt.subplots(2,5,figsize=(12,5))
# labels=['spiking','nonspiking','stdp']
#
# for i in range(10):
#     spiking_dir = 'skewsym_spiking_'+str(i)
#     nonspiking_dir = 'skewsym_nonspiking_'+str(i)
#     stdp_dir = 'stdp_dir_'+str(i)
#     spiking_train_error, spiking_test_error = csv2array(spiking_dir)
#     nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
#     stdp_train_error,stdp_test_error = csv2array(stdp_dir)
#     ax[i//5,i%5].plot(spiking_test_error)
#     ax[i//5,i%5].plot(nonspiking_test_error)
#     ax[i//5,i%5].plot(stdp_test_error)
#     ax[i//5,i%5].set_xlabel('Epoch')
#     ax[i//5,i%5].set_ylabel('Test error rate (%)')
#     ax[i//5,i%5].set_title('plot '+str(i))
# fig.suptitle("Comparison of three approximately equivalent dynamics")
# fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
# fig.savefig('skew_error.png',bbox_inches="tight")
