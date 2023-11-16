from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.nn.functional as F
import pickle


from main import rho, rhop

class SNN(nn.Module):

    def __init__(self, args):
        super(SNN, self).__init__()
        self.N1 = args.N1
        self.N2 = args.N2
        self.dt = args.dt
        self.size_tab = args.size_tab
        self.lr_tab = args.lr_tab
        self.ns = len(args.size_tab) - 1
        self.nsyn = 2*(self.ns - 1) + 1
        self.cep = args.cep
        self.use_bias = args.use_bias
        self.debug_cep = args.debug_cep
        self.no_reset = args.no_reset
        self.no_rhop = args.no_rhop
        self.plain_data = args.plain_data
        self.update_rule = args.update_rule
        self.trace_decay = args.trace_decay
        self.directory = args.directory
        self.current_epoch = 0
        self.spiking = args.spiking
        # self.spike_height = args.spike_height
        self.step = args.step
        # self.max_fr = args.max_fr
        # self.max_Q = args.max_Q
        if args.device_label >= 0:
            device = torch.device("cuda:"+str(args.device_label))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False
        self.device = device
        self.no_clamp = args.no_clamp
        self.beta = args.beta
        self.spike_method = args.spike_method
        self.omega = args.omega
        #*********RANDOM BETA*********#
        self.randbeta = args.randbeta
        #*****************************#
        self.M = args.M

        w = nn.ModuleList([])

        for i in range(self.ns - 1):
            w.append(nn.Linear(self.M*args.size_tab[i + 1], self.M*args.size_tab[i], bias = self.use_bias))
            w.append(nn.Linear(self.M*args.size_tab[i], self.M*args.size_tab[i + 1], bias = False)) # Why default bias = False???
        w.append(nn.Linear(self.M*args.size_tab[-1], self.M*args.size_tab[-2]))

        #By default, reciprocal weights have the same initial values
        for i in range(self.ns - 1):
            w[2*i + 1].weight.data = torch.transpose(w[2*i].weight.data.clone(), 0, 1)

        self.w = w
        self = self.to(device)

    def stepper(self, s=None, spike=None, error=None, trace=None, target=None, beta=0, return_derivatives=False, update_weights=False):
        dsdt = []
        trace_decay = self.trace_decay

        # Calculate dsdt
        if self.spiking:
            dsdt.append(-s[0] + self.w[0](spike[1]))
            if np.abs(beta) > 0:
                dsdt[0] = dsdt[0] + beta*(target-spike[0])
            for i in range(1, self.ns):
                dsdt.append(-s[i] + self.w[2*i](spike[i+1]) + self.w[2*i-1](spike[i-1]))
        else:
            dsdt.append(-s[0] + self.w[0](rho(s[1])))
            if np.abs(beta) > 0:
                dsdt[0] = dsdt[0] + beta*(target-rho(s[0]))
            for i in range(1, self.ns):
                dsdt.append(-s[i] + self.w[2*i](rho(s[i+1])) + self.w[2*i-1](rho(s[i-1])))

        s_old = [x.clone() for x in s]

        # Update s
        if self.no_clamp:
            for i in range(self.ns):
                s[i] = s[i] + self.dt*dsdt[i]
        else:
            for i in range(self.ns):
                s[i] = (s[i] + self.dt*dsdt[i]).clamp(min = 0).clamp(max = 1)
                dsdt[i] = torch.where((s[i] == 0)|(s[i] ==1), torch.zeros_like(dsdt[i], device = self.device), dsdt[i])

        # Traces
        if trace != None:
            if self.update_rule == 'stdp':
                for i in range(self.ns+1):
                    trace[i] = self.trace_decay*(trace[i]+spike[i])
            elif self.update_rule == 'nonspikingstdp':
                for i in range(self.ns+1):
                    trace[i] = self.trace_decay*(trace[i]+rho(s[i]))

        # Get spikes
        for i in range(self.ns+1):
            if self.spike_method == 'poisson':
                spike[i] = (torch.rand(s[i].size(),device=self.device)<rho(s[i])).float()
            elif self.spike_method == 'accumulator':
                omega = self.omega
                spike[i] = torch.floor(omega*(rho(s[i])+error[i]))/omega
                error[i] += rho(s[i])-spike[i]
            elif self.spike_method == 'nonspiking':
                spike[i] = rho(s[i])

        # CEP
        if update_weights:
            dw = self.computeGradients(s, s_old, trace, spike)
            with torch.no_grad():
                self.updateWeights(dw)
        return s,dsdt


    def forward(self, N, s=None, spike=None, error=None, trace=None, seq=None,  beta=0, target=None, record=False, update_weights=False):
        save_data_dict = {'s':[],'spike':[],'w':[]}

        for t in range(N):
            if record: # Store data
                save_data_dict['s'].append([si.detach().cpu().numpy().copy() for si in s])
                save_data_dict['spike'].append([spikei.detach().cpu().numpy().copy() for spikei in spike])
                if beta>0:
                    save_data_dict['w'].append([wi.weight.detach().cpu().numpy().copy() for wi in self.w])

                # for key in save_data_dict.keys():
                #     for x in save_data_dict[key]:
                #         for y in x:
                #             print(type(x))

            s,dsdt = self.stepper(data,s=s,spike=spike,error=error,trace=trace,target=target,beta=beta,update_weights=update_weights)

        #     if record:
        #         delta = [torch.sqrt(torch.mean(dsdt_i**2)).detach().cpu().numpy() for dsdt_i in dsdt]
        #         deltas.append(delta)
        #         for i in range(len(node_list)):
        #             layer,node = node_list[i]
        #             mps[i].append(s[layer][0,node].detach().cpu().numpy())
        # mps = np.array(mps)
        # info['mps'] = mps
        # info['deltas'] = deltas

        return s,save_data_dict

    def initHidden(self, batch_size):
        s = []
        for i in range(self.ns+1):
            s.append(torch.zeros(batch_size, self.M*self.size_tab[i], requires_grad = True))
        return s

    def computeGradients(self, s, seq, trace, spike):
        gradw = []
        gradw_bias = []
        batch_size = s[0].size(0)
        beta = self.beta
        for i in range(self.ns - 1):
            if self.update_rule == 'asym':
                gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(rho(s[i]) - rho(seq[i]), 0, 1), rho(s[i + 1])))
                gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(rho(s[i + 1]) - rho(seq[i + 1]), 0, 1), rho(s[i])))
            elif self.update_rule == 'skew':
                gradw.append((1/(beta*batch_size))*( torch.mm(torch.transpose(rho(s[i]) - rho(seq[i]), 0, 1), rho(s[i + 1])) -  torch.mm(torch.transpose(rho(s[i]),0,1),rho(s[i+1])-rho(seq[i+1])) ))
                gradw.append((1/(beta*batch_size))*( torch.mm(torch.transpose(rho(s[i+1]) - rho(seq[i+1]), 0, 1), rho(s[i])) -  torch.mm(torch.transpose(rho(s[i+1]),0,1),rho(s[i])-rho(seq[i])) ))
            elif self.update_rule == 'skewsym':
                gradw.append((1/(beta*batch_size))*( torch.mm(torch.transpose(rho(s[i]) - rho(seq[i]), 0, 1), rho(s[i + 1])) -  torch.mm(torch.transpose(rho(s[i]),0,1),rho(s[i+1])-rho(seq[i+1])) ))
                gradw.append((1/(beta*batch_size))*( torch.mm(torch.transpose(rho(s[i+1]), 0, 1), rho(s[i]) - rho(seq[i])) -  torch.mm(torch.transpose(rho(s[i+1])-rho(seq[i+1]),0,1),rho(s[i])) ))
            elif self.update_rule == 'stdp':
                gradw.append(((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*( -torch.mm(torch.transpose(trace[i], 0, 1), spike[i + 1]) +  torch.mm(torch.transpose(spike[i],0,1),trace[i+1]) ))
                gradw.append(((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*( -torch.mm(torch.transpose(spike[i+1], 0, 1), trace[i]) +  torch.mm(torch.transpose(trace[i+1],0,1),spike[i]) ))
            elif self.update_rule == 'nonspikingstdp':
                gradw.append(((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*( -torch.mm(torch.transpose(trace[i], 0, 1), rho(s[i + 1])) +  torch.mm(torch.transpose(rho(s[i]),0,1),trace[i+1]) ))
                gradw.append(((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*( -torch.mm(torch.transpose(rho(s[i+1]), 0, 1), trace[i]) +  torch.mm(torch.transpose(trace[i+1],0,1),rho(s[i])) ))
            elif self.update_rule == 'cep':
                gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[i]), 0, 1), rho(s[i+1])) - torch.mm(torch.transpose(rho(seq[i]), 0, 1), rho(seq[i+1]))))
                gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[i+1]), 0, 1), rho(s[i])) - torch.mm(torch.transpose(rho(seq[i+1]), 0, 1), rho(seq[i]))))
            elif self.update_rule == 'cepalt':
                gradw.append((1/(beta*batch_size))*( torch.mm(torch.transpose(rho(s[i]) - rho(seq[i]), 0, 1), rho(s[i + 1])) +  torch.mm(torch.transpose(rho(s[i]),0,1),rho(s[i+1])-rho(seq[i+1])) ))
                gradw.append((1/(beta*batch_size))*( torch.mm(torch.transpose(rho(s[i+1]), 0, 1), rho(s[i]) - rho(seq[i])) +  torch.mm(torch.transpose(rho(s[i+1])-rho(seq[i+1]),0,1),rho(s[i])) ))
            if self.use_bias:
                gradw_bias.append((1/(beta*batch_size))*(s[i] - seq[i]).sum(0))
                gradw_bias.append(None)

        if self.update_rule == 'stdp':
            gradw.append( ((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*(torch.mm(torch.transpose(spike[-2],0,1),trace[-1]) - torch.mm(torch.transpose(trace[-2],0,1), spike[-1]) ))
        elif self.update_rule =='nonspikingstdp':
            gradw.append( ((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*(torch.mm(torch.transpose(rho(s[-2]),0,1),trace[-1]) - torch.mm(torch.transpose(trace[-2],0,1), rho(s[-1])) ))
        elif self.update_rule == 'skewsym':
            gradw.append((1/(beta*batch_size))*(torch.mm(torch.transpose(rho(s[-2]) - rho(seq[-2]), 0, 1), rho(s[-1])) -  torch.mm(torch.transpose(rho(s[-2]),0,1),rho(s[-1])-rho(seq[-1])) ))
        else:
            gradw.append((1/(beta*batch_size))*torch.mm(torch.transpose(rho(s[-2]) - rho(seq[-2]), 0, 1), rho(s[-1])))
        if self.use_bias:
            gradw_bias.append((1/(beta*batch_size))*(s[-1] - seq[-1]).sum(0))

        return  gradw, gradw_bias

    #**************************NEW**************************#
    def updateWeights(self, gradw):
        lr_tab = self.lr_tab
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[0][i]
            if self.use_bias:
                if gradw[1][i] is not None:
                    self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw[1][i]
