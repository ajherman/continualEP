from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim
import os, sys
import datetime
from shutil import copyfile
import copy
import csv
import pickle

def rho(x):
    return x.clamp(min = 0).clamp(max = 1)
def rhop(x):
    return ((x >= 0) & (x <= 1)).float()

def train(net, train_loader, epoch, learning_rule):

    net.train()
    loss_tot = 0
    correct = 0
    mps_li = []
    deltas_li = []
    criterion = nn.MSELoss(reduction = 'sum')
    for batch_idx, (data, targets) in enumerate(train_loader):
        if not net.no_reset or batch_idx == 0:
            s = net.initHidden(data.size(0))
            trace = net.initHidden(data.size(0))
            spike = net.initHidden(data.size(0))
            if net.spike_method == 'accumulator':
                error = net.initHidden(data.size(0))
            else:
                error = None

        data, targets = data.to(net.device), targets.to(net.device)

        for i in range(net.ns+1):
            s[i] = s[i].to(net.device)
            if net.update_rule == 'stdp' or net.update_rule == 'nonspikingstdp':
                trace[i] = trace[i].to(net.device)
            spike[i] = spike[i].to(net.device)
            if net.spike_method == 'accumulator':
                error[i] = error[i].to(net.device)

        for i in range(net.ns+1):
            if net.spiking:
                spike[i] = (torch.rand(s[i].size(),device=net.device)<(rho(s[i]))).float()
            else:
                spike[i] = rho(s[i]) # Get Poisson spikes

        with torch.no_grad():
            s[net.ns] = data

            if batch_idx==0:
                s,phase1_data = net.forward(data, net.N1, s=s, spike=spike,error=error,record=True)
                # with open(net.directory+'/phase1_data_'+str(epoch)+'.pkl', 'wb') as f:
                #     pickle.dump(info,f)
            else:
                s,_ = net.forward(data,net.N1,s=s,spike=spike,error=error)

            pred = s[0].data.max(1, keepdim=True)[1]
            loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
            # #*******************************************VF-EQPROP ******************************************#
            # seq = []
            # for i in range(len(s)): seq.append(s[i].clone())
            seq = [x.clone() for x in s]

            beta = net.beta

            if batch_idx==0:
                s, phase2_data = net.forward(data,net.N2, s=s, spike=spike,error=error,trace=trace, target=targets, beta=beta,record=True,update_weights=True)
                # with open(net.directory+'/phase2_data_'+str(epoch)+'.pkl', 'wb') as f:
                #     pickle.dump(info,f)
            else:
                s,info = net.forward(data,net.N2,s=s,spike=spike,error=error,trace=trace,target=targets,beta=beta,update_weights=True)
                # Dw = info['dw']
            #***********************************************************************************************#

            # if batch_idx%500==0:
            #     mps=np.concatenate((mps1,mps2),axis=1)
            #     deltas=np.concatenate((deltas1,deltas2),axis=0)
            #     mps_li.append(mps)
            #     deltas_li.append(deltas)

        loss_tot += loss
        targets_temp = targets.data.max(1, keepdim=True)[1]
        correct += pred.eq(targets_temp.data.view_as(pred)).cpu().sum()

        if (batch_idx + 1)% 100 == 0:
           print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
               epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
               100. * (batch_idx + 1) / len(train_loader), loss.data))


    loss_tot /= len(train_loader.dataset)


    print('\nAverage Training loss: {:.4f}, Training Error Rate: {:.2f}% ({}/{})\n'.format(
       loss_tot,100*(len(train_loader.dataset)- correct.item() )/ len(train_loader.dataset), len(train_loader.dataset)-correct.item(), len(train_loader.dataset),
       ))

    return 100*(len(train_loader.dataset)- correct.item())/ len(train_loader.dataset),{'phase1':phase1_data,'phase2':phase2_data}

def evaluate(net, test_loader, learning_rule=None):
    net.eval()
    loss_tot_test = 0
    correct_test = 0
    criterion = nn.MSELoss(reduction = 'sum')
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            if not net.no_reset or batch_idx==0:
                s = net.initHidden(data.size(0))
                trace = net.initHidden(data.size(0))
                spike = net.initHidden(data.size(0))
                if net.spike_method == 'accumulator':
                    error = net.initHidden(data.size(0))
                else:
                    error = None
            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(net.ns+1):
                    s[i] = s[i].to(net.device)
                    spike[i] = spike[i].to(net.device)
                    if net.spike_method == 'accumulator':
                        error[i] = error[i].to(net.device)
            s[net.ns] = data

            # New!
            for i in range(net.ns+1):
                if net.spiking:
                    spike[i] = (torch.rand(s[i].size(),device=net.device)<(rho(s[i]))).float()
                else:
                    spike[i] = rho(s[i]) # Get Poisson spikes

            s,_ = net.forward(data, net.N1, s=s, spike=spike,error=error)
            loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
            loss_tot_test += loss #(1/2)*((s[0]-targets)**2).sum()
            pred = s[0].data.max(1, keepdim = True)[1]
            targets_temp = targets.data.max(1, keepdim = True)[1]
            correct_test += pred.eq(targets_temp.data.view_as(pred)).cpu().sum()

    loss_tot_test = loss_tot_test / len(test_loader.dataset)
    accuracy = correct_test.item() / len(test_loader.dataset)
    print('\nAverage Test loss: {:.4f}, Test Error Rate: {:.2f}% ({}/{})\n'.format(
        loss_tot_test,100. *(len(test_loader.dataset)- correct_test.item() )/ len(test_loader.dataset), len(test_loader.dataset)-correct_test.item(), len(test_loader.dataset)))
    return 100 *(len(test_loader.dataset)- correct_test.item() )/ len(test_loader.dataset)

def createPath(args):

    if args.action == 'train':
        BASE_PATH = os.getcwd() + '/'

        if args.cep:
            name = 'c-' + args.learning_rule
        else:
            name = args.learning_rule

        if args.discrete:
            name = name + '_disc'
        else:
            name = name + '_cont'
        name = name + '_' + str(len(args.size_tab) - 2) + 'hidden'

        BASE_PATH = BASE_PATH + name

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        BASE_PATH = BASE_PATH + '/' + datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d")

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        files = os.listdir(BASE_PATH)

        if not files:
            BASE_PATH = BASE_PATH + '/' + 'Trial-1'
        else:
            tab = []
            for names in files:
                if not names[-2] == '-':
                    tab.append(int(names[-2] + names[-1]))
                else:
                    tab.append(int(names[-1]))

            BASE_PATH = BASE_PATH + '/' + 'Trial-' + str(max(tab)+1)

        os.mkdir(BASE_PATH)
        filename = 'results'

        #************************************#
        copyfile('plotFunctions.py', BASE_PATH + '/plotFunctions.py')
        #************************************#

        return BASE_PATH, name

    elif (args.action == 'plotcurves'):
        BASE_PATH = os.getcwd() + '/'

        if args.cep:
            name = 'c-' + args.learning_rule
        else:
            name = args.learning_rule

        if args.discrete:
            name = name + '_disc'
        else:
            name = name + '_cont'
        name = name + '_' + str(len(args.size_tab) - 2) + 'hidden'

        BASE_PATH = BASE_PATH + name

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        files = os.listdir(BASE_PATH)

        if not files:
            BASE_PATH = BASE_PATH + '/' + 'Trial-1'
        else:
            tab = []
            for names in files:
                if not names[-2] == '-':
                    tab.append(int(names[-2] + names[-1]))
                else:
                    tab.append(int(names[-1]))

            BASE_PATH = BASE_PATH + '/' + 'Trial-' + str(max(tab)+1)

        os.mkdir(BASE_PATH)
        filename = 'results'

        #********************************************************#
        copyfile('plotFunctions.py', BASE_PATH + '/plotFunctions.py')
        #********************************************************#

        return BASE_PATH, name


    elif (args.action == 'RMSE') or (args.action == 'prop') or (args.action == 'cosRMSE'):
        BASE_PATH = os.getcwd() + '/'

        if args.cep:
            name = 'c-' + args.learning_rule
        else:
            name = args.learning_rule

        if args.discrete:
            name = name + '_disc'
        else:
            name = name + '_cont'
        name = name + '_' + str(len(args.size_tab) - 2) + 'hidden'

        BASE_PATH = BASE_PATH + name

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        if (args.learning_rule == 'vf'):
            BASE_PATH = BASE_PATH + '/theta_' + str(args.angle)

        if not os.path.exists(BASE_PATH):
            os.mkdir(BASE_PATH)

        files = os.listdir(BASE_PATH)

        if not files:
            BASE_PATH = BASE_PATH + '/' + 'Trial-1'
        else:
            tab = []
            for names in files:
                if not names[-2] == '-':
                    tab.append(int(names[-2] + names[-1]))
                else:
                    tab.append(int(names[-1]))

            BASE_PATH = BASE_PATH + '/' + 'Trial-' + str(max(tab)+1)

        os.mkdir(BASE_PATH)
        filename = 'results'

        #********************************************************#
        copyfile('plotFunctions.py', BASE_PATH + '/plotFunctions.py')
        #********************************************************#

        return BASE_PATH, name


def createHyperparameterfile(BASE_PATH, name, args):

    if args.action == 'train':
        learning_rule = args.learning_rule
        if args.cep:
            learning_rule = 'c-' + learning_rule
        hyperparameters = open(BASE_PATH + r"/hyperparameters.txt","w+")
        L = [" TRAINING: list of hyperparameters " + "(" + name + ", " + datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d") + ") \n",
			"- Learning rule: " + learning_rule + "\n",
            "- N1: {}".format(args.N1) + "\n",
            "- N2: {}".format(args.N2) + "\n",
            "- beta: {:.2f}".format(args.beta) + "\n",
            "- batch size: {}".format(args.batch_size) + "\n",
            "- activation function: " + args.activation_function + "\n",
            "- number of epochs: {}".format(args.epochs) + "\n",
            "- learning rates: {}".format(args.lr_tab) + "\n"]

        if not args.discrete:
            L.append("- dt: {:.3f}".format(args.dt) + "\n")

        if args.randbeta > 0:
            L.append("- Probability of beta sign switching: {}".format(args.randbeta) + "\n")

        if args.angle > 0:
            L.append("- Initial angle between forward and backward weights: {}".format(args.angle) + "\n")

        L.append("- layer sizes: {}".format(args.size_tab) + "\n")

        hyperparameters.writelines(L)
        hyperparameters.close()

    elif (args.action == 'plotcurves'):
        hyperparameters = open(BASE_PATH + r"/hyperparameters.txt","w+")
        L = ["NABLA-DELTA CURVES: list of hyperparameters " + "(" + name + ", " + datetime.datetime.now().strftime("cuda" + str(args.device_label)+"-%Y-%m-%d") + ") \n",
            "- Learning rule: " + args.learning_rule + "\n",
            "- N1: {}".format(args.N1) + "\n",
            "- N2: {}".format(args.N2) + "\n",
            "- beta: {:.2f}".format(args.beta) + "\n",
            "- batch size: {}".format(args.batch_size) + "\n",
            "- activation function: " + args.activation_function + "\n"]

        if not args.discrete:
            L.append("- dt: {:.3f}".format(args.dt) + "\n")

        if args.randbeta > 0:
            L.append("- Probability of beta sign switching: {}".format(args.randbeta) + "\n")

        if args.angle > 0:
            L.append("- Initial angle between forward and backward weights: {}".format(args.angle) + "\n")


        L.append("- layer sizes: {}".format(args.size_tab) + "\n")

        hyperparameters.writelines(L)
        hyperparameters.close()
