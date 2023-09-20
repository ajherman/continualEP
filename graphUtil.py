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


fig, ax = plt.subplots(2,5,figsize=(10,4))
labels=['spiking','nonspiking']

for i in range(10):
    spiking_dir = 'cepalt_spiking_'+str(i)
    nonspiking_dir = 'cepalt_nonspiking_'+str(i)
    spiking_train_error, spiking_test_error = csv2array(spiking_dir)
    nonspiking_train_error,nonspiking_test_error = csv2array(nonspiking_dir)
    ax[i//5,i%5].plot(spiking_test_error)
    ax[i//5,i%5].plot(nonspiking_test_error)
fig.legend(labels, loc='lower right', ncol=len(labels), bbox_transform=fig.transFigure)
fig.savefig('cepalt_error.png',bbox_inches="tight")
