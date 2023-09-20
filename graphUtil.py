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


fig, ax = plt.subplots(2,5)

for i in range(10):
    spiking_dir = 'cepalt_spiking_'+str(i)
    nonspiking_dir = 'cepalt_nonspiking_'+str(i)
    spiking_train_error, spiking_test_error = csv2array(spiking_dir)
    nonspiking_train_error,non_spiking_test_error = csv2array(nonspiking_dir)
    ax.plot[i//5,i%5](spiking_test_error)
    ax.plot[i//5,i%5](nonspiking_test_error)
fig.savefig('cepalt_error.png')

#
# train_error1,test_error1 = csv2array('cepalt_1')
# train_error2,test_error2 = csv2array('cepalt_2')
# train_error3,test_error3 = csv2array('cepalt_3')
# fig, ax = plt.subplots(3)
# ax.plot[0](train_error1)
# ax.plot[1](train_error2)
# ax.plot[2](train_error3)

fig.savefig(args.directory+'/error.png')
# print(errors)
