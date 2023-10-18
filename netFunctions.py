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

def rho(x):
    return x.clamp(min = 0).clamp(max = 1)
def rhop(x):
    return ((x >= 0) & (x <= 1)).float()

def train(net, train_loader, epoch, learning_rule):

    net.train()
    loss_tot = 0
    correct = 0
    criterion = nn.MSELoss(reduction = 'sum')
    for batch_idx, (data, targets) in enumerate(train_loader):
        if not net.no_reset or batch_idx == 0:
            s = net.initHidden(data.size(0))
        trace = net.initHidden(data.size(0))
        spike = net.initHidden(data.size(0))
        data, targets = data.to(net.device), targets.to(net.device)

        if learning_rule == 'stdp':
            for i in range(net.ns+1):
                s[i] = s[i].to(net.device)
                trace[i] = trace[i].to(net.device)
                spike[i] = spike[i].to(net.device)
        else:
            for i in range(net.ns):
                s[i] = s[i].to(net.device)
                trace[i] = trace[i].to(net.device)
                spike[i] = spike[i].to(net.device)

        # New!
        for i in range(net.ns+1):
            if net.spiking:
                spike[i] = net.spike_height*(torch.rand(s[i].size(),device=net.device)<(rho(s[i])*net.max_Q/net.spike_height)).float()
            else:
                spike[i] = rho(s[i])*net.max_Q # Get Poisson spikes

        if learning_rule == 'ep':
            with torch.no_grad():
                s = net.forward(data, s)
                pred = s[0].data.max(1, keepdim=True)[1]
                loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
                #************************************ EQPROP *******************************************#
                seq = []
                for i in range(len(s)): seq.append(s[i].clone())

                if not net.debug_cep:
                    if net.randbeta > 0:
                        signbeta = 2*np.random.binomial(1, net.randbeta, 1).item() - 1
                        beta = signbeta*net.beta
                    else:
                        beta = net.beta

                    s = net.forward(data, s, target = targets, beta = beta, method = 'nograd')
                    if not net.cep:
                        Dw = net.computeGradients(data, s, seq, beta)
                        net.updateWeights(Dw)
                else:
                    s, Dw = net.forward(data, s, target = targets, beta = net.beta, method = 'nograd')
                    with torch.no_grad():
                        for ind, w_temp in enumerate(net.w):
                            if w_temp is not None:
                                w_temp.weight -= net.lr_tab_debug[int(np.floor(ind/2))]*Dw[0][ind]
                                w_temp.bias -= net.lr_tab_debug[int(np.floor(ind/2))]*Dw[1][ind]

                    net.updateWeights(Dw)
                #***********************************************************************************#


        elif learning_rule == 'vf':
            with torch.no_grad():
                s = net.forward(data, s)
                pred = s[0].data.max(1, keepdim=True)[1]
                loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
                #*******************************************VF-EQPROP ******************************************#
                seq = []
                for i in range(len(s)): seq.append(s[i].clone())

                #******************************************FORMER C-VF******************************************#
                if net.randbeta > 0:
                    signbeta = 2*np.random.binomial(1, net.randbeta, 1).item() - 1
                    beta = signbeta*net.beta
                else:
                    beta = net.beta

                s, Dw = net.forward(data, s, target = targets, beta = beta, method = 'nograd')
                #***********************************************************************************************#

                if not net.cep:
                    if not net.former:
                        net.updateWeights(Dw)
                    else:
                        Dw_former = net.computeGradients(data, s, seq, beta)
                        net.updateWeights(Dw_former)
                #########################################################################################

        elif learning_rule == 'stdp':
            with torch.no_grad():
                s[net.ns] = data

                record=batch_idx%500==0

                if record:
                    s,deltas1, mps1 = net.forward(data, s, spike,record=True)
                else:
                    s = net.forward(data,s,spike)

                pred = s[0].data.max(1, keepdim=True)[1]
                loss = (1/(2*s[0].size(0)))*criterion(s[0], targets)
                #*******************************************VF-EQPROP ******************************************#
                seq = []
                for i in range(len(s)): seq.append(s[i].clone())

                beta = net.beta

                if record:
                    s, Dw, deltas2, mps2 = net.forward(data, s, spike, trace=trace, target=targets, beta=beta, method='nograd',record=True)
                else:
                    s,Dw = net.forward(data,s,spike,trace=trace,target=targets,beta=beta,method='nograd')
                #***********************************************************************************************#

                if record:
                    mps=np.concatenate((mps1,mps2),axis=1)
                    deltas=np.concatenate((deltas1,deltas2),axis=0)
                    with open(net.directory+'/deltas.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(deltas)
                    with open(net.directory+'/mps.csv', 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerows(mps)

                # Plots deltas to visualize convergence
                ##########################
                # if batch_idx%500==0:
                #     fig, ax = plt.subplots()
                #     ax.plot(np.arange(net.N1),deltas)
                #     fig.savefig(net.directory+'/deltas_'+str(batch_idx)+'.png')
                #     plt.close()
                ###########################

                if not net.cep:
                    if not net.former:
                        net.updateWeights(Dw)
                    else:
                        Dw_former = net.computeGradients(data, s, seq, beta)
                        net.updateWeights(Dw_former)
                #########################################################################################

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

    return 100*(len(train_loader.dataset)- correct.item())/ len(train_loader.dataset)

def evaluate(net, test_loader, learning_rule=None):
    net.eval()
    loss_tot_test = 0
    correct_test = 0
    criterion = nn.MSELoss(reduction = 'sum')
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            if not net.no_reset or batch_idx==0:
                s = net.initHidden(data.size(0))
            spike = net.initHidden(data.size(0))
            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(net.ns+1):
                    s[i] = s[i].to(net.device)
                    spike[i] = spike[i].to(net.device)

            if learning_rule == 'stdp':
                s[net.ns] = data

            # New!
            for i in range(net.ns+1):
                if net.spiking:
                    spike[i] = net.spike_height*(torch.rand(s[i].size(),device=net.device)<(rho(s[i])*net.max_Q/net.spike_height)).float()
                else:
                    spike[i] = rho(s[i])*net.max_Q # Get Poisson spikes

            s = net.forward(data, s, spike,method = 'nograd')
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

#################################
# Not sure what all of this does!
#################################

###################################################################################
# def compute_nSdSdT(net, data, target):
#
#     beta = net.beta
#     batch_size_temp = data.size(0)
#     s = net.initHidden(batch_size_temp)
#     if net.cuda:
#         for i in range(net.ns):
#             s[i] = s[i].to(net.device)
#
#     net.zero_grad()
#     s, nS = net.forward(data, s, target = target, method = 'nS')
#
#
#     seq = []
#     for i in range(len(s)):
#         seq.append(s[i].clone())
#     with torch.no_grad():
#         s, dS, dT = net.forward(data, s, seq, target = target, beta = beta, method = 'dSdT')
#
#     return nS, dS, dT, seq
#
#
# def compute_nT(net, data, target, wholeProcess = True, diff = True):
#
#     batch_size_temp = data.size(0)
#
#     NT = []
#     for i in range(len(net.w)):
#         if net.w[i] is not None:
#             NT.append(torch.zeros(net.Kmax, net.w[i].weight.size(0), net.w[i].weight.size(1)))
#         else:
#             NT.append(None)
#
#     criterion = nn.MSELoss(reduction = 'sum')
#     if wholeProcess:
#         for K in range(net.Kmax):
#             print(K)
#             s = net.initHidden(batch_size_temp)
#             if net.cuda:
#                 for i in range(net.ns):
#                     s[i] = s[i].to(net.device)
#             net.zero_grad()
#             s = net.forward(data, s, method = 'withgrad', K = K)
#             loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
#             loss.backward()
#
#             for i in range(len(NT)):
#                 if net.w[i] is not None:
#                     NT[i][K, :, :] = net.w[i].weight.grad.clone()
#     else:
#             s = net.initHidden(batch_size_temp)
#             if net.cuda:
#                 for i in range(net.ns):
#                     s[i] = s[i].to(net.device)
#             net.zero_grad()
#             s = net.forward(data, s, method = 'withgrad', K = net.Kmax)
#             loss = (1/(2.0*s[0].size(0)))*criterion(s[0], target)
#             loss.backward()
#
#             for i in range(len(NT)):
#                 if net.w[i] is not None:
#                     NT[i][-1, :, :] = net.w[i].weight.grad.clone()
#
#     if diff:
#         NT = compute_diffT(NT)
#
#     return NT
#
#
# def compute_diffT(NT):
#
#     nT = []
#
#     for i in range(len(NT)):
#         if NT[i] is not None:
#             nT.append(torch.zeros_like(NT[i]))
#         else:
#             nT.append(None)
#
#     for i in range(len(NT)):
#         if NT[i] is not None:
#             for t in range(NT[i].size(0) - 1):
#                 nT[i][t + 1, :, :] = NT[i][t + 1, :, :] - NT[i][t, :, :]
#
#     return nT
#
#
# #*********WATCH OUT: compute cosRelMSE*********#
# def compute_angleGrad(nS, dS, nT, dT):
#     NT = torch.tensor([], device = nT[0].device)
#     DT = torch.tensor([], device = dT[0].device)
#
#     for i in nT:
#         if i is not None:
#             NT = torch.cat((NT, i.sum(0).view(-1,1)) , 0)
#
#     for i in dT:
#         if i is not None:
#             DT = torch.cat((DT, i.sum(0).view(-1,1)) , 0)
#
#     theta_T =(180/np.pi) * np.arccos(torch.mm(NT.t(), DT).item()/(np.sqrt(torch.mm(NT.t(), NT).item()*torch.mm(DT.t(), DT).item())))
#
#     return theta_T
#################################################################################################################

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
