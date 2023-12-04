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

def train(net, train_loader, epoch, learning_rule,save_interval,save_path):

    net.train()
    # Reset arrays if new epoch
    if net.current_batch==0:
        net.loss_tot = []
        net.correct = []
    # mps_li = []
    # deltas_li = []
    criterion = nn.MSELoss(reduction = 'sum')
    if not hasattr(net,'current_batch'):
        net.current_batch=0
    with torch.no_grad():
        # while net.current_batch < len(train_loader):
        #     batch_idx = net.current_batch # Rename all instances
        #     data,targets = train_loader[batch_idx]
        for batch_idx, (data, targets) in enumerate(train_loader):
            if batch_idx < net.current_batch:
                pass # Skip batches until we get to net.current_batch
            else:
                # Init arrays
                if not net.no_reset or batch_idx == 0:
                    s = net.initHidden(data.size(0))
                    trace = net.initHidden(data.size(0))
                    spike = net.initHidden(data.size(0))
                    if net.spike_method == 'accumulator':
                        error = net.initHidden(data.size(0))
                    else:
                        error = None

                # Change data size/shape to increase population
                # Try torch.repeat too
                # data,targets = torch.tile(data,(1,net.M)),torch.tile(targets,(1,net.M))
                data, targets = data.to(net.device), targets.to(net.device)
                #expand_data, expand_targets = torch.tile(data,(1,self.M)), torch.tile(targets,(1,self.M))

                # Put arrays on gpu
                for i in range(net.ns+1):
                    s[i] = s[i].to(net.device)
                    if net.update_rule == 'stdp' or net.update_rule == 'nonspikingstdp':
                        trace[i] = trace[i].to(net.device)
                    spike[i] = spike[i].to(net.device)
                    if net.spike_method == 'accumulator':
                        error[i] = error[i].to(net.device)

                if batch_idx==0:
                    out,s,phase1_data = net.forward(data,net.N1,s=s,spike=spike,error=error,record=True)
                else:
                    out,s,_ = net.forward(data,net.N1,s=s,spike=spike,error=error)

                pred = out.data.max(1, keepdim=True)[1]
                loss = (1/(2*out.size(0)))*criterion(out, targets)
                seq = [x.clone() for x in s]

                beta = net.beta

                if batch_idx==0:
                    out,s, phase2_data = net.forward(data,net.N2, s=s, spike=spike,error=error,trace=trace, target=targets, beta=beta,record=True,update_weights=True)
                else:
                    out,s,info = net.forward(data,net.N2,s=s,spike=spike,error=error,trace=trace,target=targets,beta=beta,update_weights=True)

                net.loss_tot.append(loss)

                targets_temp = targets.data.max(1, keepdim=True)[1]
                net.correct.append(pred.eq(targets_temp.data.view_as(pred)).cpu().sum())

                if (batch_idx + 1)% 100 == 0:
                   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                       epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.data))


                # Moving this into train fn
                ###########################
                net.current_batch = batch_idx
                # batch_idx = net.current_batch
                if batch_idx%save_interval==0:
                    # # Save to csv
                    # csv_path = save_path+"/results.csv"
                    # with open(csv_path,'a+',newline='') as csv_file:
                    #     csv_writer = csv.writer(csv_file)
                    #     csv_writer.writerow([error_train, error_test])

                    #  Increment epoch and save network
                    # net.current_epoch += 1
                    pkl_path = save_path+'/net'
                    with open(pkl_path,'wb') as pkl_file:
                        pickle.dump(net,pkl_file)
                ############################

    # loss_tot /= len(train_loader.dataset)
    mean_loss = torch.mean(torch.stack(net.loss_tot))
    correct = torch.sum(torch.stack(net.correct))

    # print('\nAverage Training loss: {:.4f}, Training Error Rate: {:.2f}% ({}/{})\n'.format(
    #    loss_tot,100*(len(train_loader.dataset)- correct.item() )/ len(train_loader.dataset), len(train_loader.dataset)-correct.item(), len(train_loader.dataset),
    #    ))

    print('\nAverage Training loss: {:.4f}, Training Error Rate: {:.2f}% ({}/{})\n'.format(
       mean_loss,100*(len(train_loader.dataset)- correct.item() )/ len(train_loader.dataset), len(train_loader.dataset)-correct.item(), len(train_loader.dataset),
       ))

    return 100*(len(train_loader.dataset)- correct.item())/ len(train_loader.dataset) #,{'phase1':phase1_data,'phase2':phase2_data}

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
            # data,targets = torch.tile(data,(1,net.M)),torch.tile(targets,(1,net.M))
            if net.cuda:
                data, targets = data.to(net.device), targets.to(net.device)
                for i in range(net.ns+1):
                    s[i] = s[i].to(net.device)
                    spike[i] = spike[i].to(net.device)
                    if net.spike_method == 'accumulator':
                        error[i] = error[i].to(net.device)
            # s[net.ns] = torch.tile(data,(1,net.M))
            #
            # # New!
            # for i in range(net.ns+1):
            #     if net.spiking:
            #         spike[i] = (torch.rand(s[i].size(),device=net.device)<(rho(s[i]))).float()
            #     else:
            #         spike[i] = rho(s[i]) # Get Poisson spikes

            out,s,_ = net.forward(data,net.N1, s=s, spike=spike,error=error)
            # out = torch.mean(torch.reshape(s[0],(s[0].size(0),net.M,-1)),axis=1)

            loss = (1/(2*out.size(0)))*criterion(out, targets)
            loss_tot_test += loss #(1/2)*((s[0]-targets)**2).sum()
            pred = out.data.max(1, keepdim = True)[1]
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
