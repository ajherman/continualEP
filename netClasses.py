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
        self.current_batch = 0
        self.loss_tot = []
        self.correct = []
        self.spiking = args.spiking
        self.tau_dynamic = args.tau_dynamic
        self.step = args.step
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
        self.blowup_method = args.blowup_method
        if self.blowup_method == 'sum':
            self.act_fn_scl = 1./self.M
        elif self.blowup_method == 'mean':
            self.act_fn_scl = 1.0
        else:
            assert(0)

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
        # self.apply(self._init_weights) # This will break weight symmetry

    def activation(self, x):
        return rho(x)*self.act_fn_scl

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if self.blowup_method == 'mean':
                # torch.nn.init.normal_(module.weight, mean=0.0, std=1./torch.sqrt(3*module.weight.size(1))) # Normal version
                b=1./torch.sqrt(module.weight.size(1))    
                torch.nn.init.uniform_(module.weight, -b,b)  
            elif self.blowup_method == 'sum':
                b=1./torch.sqrt(module.weight.size(1)/self.M)    
                torch.nn.init.uniform_(module.weight, -b,b) 
            else:
                assert(0)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def stepper(self, s=None, spike=None, error=None, trace=None, target=None, beta=0, return_derivatives=False, update_weights=False):
        trace_decay = self.trace_decay
        # Calculate inputs
        input=[]
        if self.spiking:
            input.append(self.w[0](spike[1]))
            if np.abs(beta) > 0:
                input[0] = input[0] + beta*(target-spike[0])
            for i in range(1, self.ns):
                input.append(self.w[2*i](spike[i+1]) + self.w[2*i-1](spike[i-1]))
        else:
            input.append(self.w[0](self.activation(s[1])))
            if np.abs(beta) > 0:
                input[0] = input[0] + beta*(target-self.activation(s[0]))
            for i in range(1, self.ns):
                input.append(self.w[2*i](self.activation(s[i+1])) + self.w[2*i-1](self.activation(s[i-1])) )       

        dsdt = []
        for i in range(self.ns):
            dsdt.append(-s[i] + input[i])

        # # Calculate dsdt
        # if self.spiking:
        #     dsdt.append(-s[0] + self.w[0](spike[1]))
        #     if np.abs(beta) > 0:
        #         dsdt[0] = dsdt[0] + beta*(target-spike[0])
        #     for i in range(1, self.ns):
        #         dsdt.append(-s[i] + self.w[2*i](spike[i+1]) + self.w[2*i-1](spike[i-1]))
        # else:
        #     dsdt.append(-s[0] + self.w[0](self.activation(s[1])))
        #     if np.abs(beta) > 0:
        #         dsdt[0] = dsdt[0] + beta*(target-self.activation(s[0]))
        #     for i in range(1, self.ns):
        #         dsdt.append(-s[i] + self.w[2*i](self.activation(s[i+1])) + self.w[2*i-1](self.activation(s[i-1])))

        s_old = [x.clone() for x in s]

        # Traces
        if trace != None:
            if self.update_rule == 'stdp':
                for i in range(self.ns+1):
                    # trace[i] = self.trace_decay*(trace[i]+spike[i])
                    trace[i] = self.trace_decay*trace[i] + spike[i]*(1-self.trace_decay)**2
                    # trace[i] = self.trace_decay*trace[i] + spike[i]*(1-self.trace_decay)

            elif self.update_rule == 'nonspikingstdp': # Still need to modify this to match above
                for i in range(self.ns+1):
                    # trace[i] = self.trace_decay*(trace[i]+self.activation(s[i]))
                    trace[i] = self.trace_decay*trace[i]+self.activation(s[i])*(1-self.trace_decay)**2
                    # trace[i] = self.trace_decay*trace[i]+self.activation(s[i])*(1-self.trace_decay)

        # Update s
        if self.no_clamp:
            for i in range(self.ns):
                s[i] = s[i] + self.dt*dsdt[i]
        else:
            for i in range(self.ns):
                s[i] = (s[i] + self.dt*dsdt[i]).clamp(min = 0).clamp(max = 1)
                dsdt[i] = torch.where((s[i] == 0)|(s[i] ==1), torch.zeros_like(dsdt[i], device = self.device), dsdt[i])

        # Get spikes
        for i in range(self.ns+1):
            if self.spike_method == 'poisson':
                spike[i] = (torch.rand(s[i].size(),device=self.device)<self.activation(s[i])).float()
            elif self.spike_method == 'accumulator':
                omega = self.omega
                spike[i] = torch.floor(omega*(self.activation(s[i])+error[i]))/omega
                error[i] += self.activation(s[i])-spike[i]
            elif self.spike_method == 'nonspiking':
                spike[i] = self.activation(s[i])
            elif self.spike_method == 'binomial':
                assert(self.omega>=1 and self.omega-np.floor(self.omega)<1e-15)
                omega=int(self.omega)
                spike[i]=torch.distributions.binomial.Binomial(total_count=omega,probs=self.activation(s[i])).sample()/omega
            elif self.spike_method == 'normal': # This should be approximately the same as binomial for large omega
                omega = self.omega
                out = self.activation(s[i])
                spike[i] = torch.normal(out,torch.sqrt(self.activation(s[i])*(1-self.activation(s[i]))/omega))

        # CEP
        if update_weights:
            dw = self.computeGradients(s, s_old, trace, spike)
            with torch.no_grad():
                self.updateWeights(dw)
        return s,dsdt

    def forward(self, data, N, s=None, spike=None, error=None, trace=None, seq=None,  beta=0, target=None, record=False, update_weights=False):
        save_data_dict = {'s':[],'spike':[],'w':[]}

        # Create expanded arrays for data and target
        expand_data = torch.tile(data,(1,self.M))
        if target != None:
            expand_target = torch.tile(target,(1,self.M))
        else:
            expand_target = None

        # Set init values for arrays
        for i in range(self.ns+1):
            if self.spiking:
                if self.spike_method == 'poisson':
                    spike[i] = (torch.rand(s[i].size(),device=self.device)<self.activation(s[i])).float()
                elif self.spike_method == 'accumulator':
                    omega = self.omega
                    spike[i] = torch.floor(omega*(self.activation(s[i])+error[i]))/omega
                    error[i] += self.activation(s[i])-spike[i]
                elif self.spike_method == 'nonspiking':
                    spike[i] = self.activation(s[i])
                elif self.spike_method == 'binomial':
                    assert(self.omega>=1 and self.omega-np.floor(self.omega)<1e-15)
                    omega=int(self.omega)
                    spike[i]=torch.distributions.binomial.Binomial(total_count=omega,probs=self.activation(s[i])).sample()/omega
                elif self.spike_method == 'normal': # This should be approximately the same as binomial for large omega
                    omega = self.omega
                    out = self.activation(s[i])
                    spike[i] = torch.normal(out,torch.sqrt(self.activation(s[i])*(1-self.activation(s[i]))/omega))
                else:
                    print("Invalid spike method")
                    assert(0)
            else:
                spike[i] = self.activation(s[i]) # Get Poisson spikes
        with torch.no_grad():
            s[self.ns] = expand_data

        for t in range(N):
            if record: # Store data
                save_data_dict['s'].append([si.detach().cpu().numpy().copy() for si in s])
                save_data_dict['spike'].append([spikei.detach().cpu().numpy().copy() for spikei in spike])
                if beta>0:
                    save_data_dict['w'].append([wi.weight.detach().cpu().numpy().copy() for wi in self.w])

            # s,dsdt = self.stepper(s=s,spike=spike,error=error,trace=trace,target=target,beta=beta,update_weights=update_weights)
            s,dsdt = self.stepper(s=s,spike=spike,error=error,trace=trace,target=expand_target,beta=beta,update_weights=update_weights)

        out = torch.mean(torch.reshape(s[0],(s[0].size(0),self.M,-1)),axis=1)
        return out,s,save_data_dict

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
        scale_factor = 1/(beta*batch_size)
        for i in range(self.ns - 1):
            if self.update_rule == 'asym':
                gradw.append(scale_factor*torch.mm(torch.transpose(self.activation(s[i]) - self.activation(seq[i]), 0, 1), self.activation(s[i + 1])))
                gradw.append(scale_factor*torch.mm(torch.transpose(self.activation(s[i + 1]) - self.activation(seq[i + 1]), 0, 1), self.activation(s[i])))
            elif self.update_rule == 'skew':
                gradw.append(scale_factor*( torch.mm(torch.transpose(self.activation(s[i]) - self.activation(seq[i]), 0, 1), self.activation(s[i + 1])) -  torch.mm(torch.transpose(self.activation(s[i]),0,1),self.activation(s[i+1])-self.activation(seq[i+1])) ))
                gradw.append(scale_factor*( torch.mm(torch.transpose(self.activation(s[i+1]) - self.activation(seq[i+1]), 0, 1), self.activation(s[i])) -  torch.mm(torch.transpose(self.activation(s[i+1]),0,1),self.activation(s[i])-self.activation(seq[i])) ))
            elif self.update_rule == 'skewsym':
                gradw.append(scale_factor*( torch.mm(torch.transpose(self.activation(s[i]) - self.activation(seq[i]), 0, 1), self.activation(s[i + 1])) -  torch.mm(torch.transpose(self.activation(s[i]),0,1),self.activation(s[i+1])-self.activation(seq[i+1])) ))
                gradw.append(scale_factor*( torch.mm(torch.transpose(self.activation(s[i+1]), 0, 1), self.activation(s[i]) - self.activation(seq[i])) -  torch.mm(torch.transpose(self.activation(s[i+1])-self.activation(seq[i+1]),0,1),self.activation(s[i])) ))
            elif self.update_rule == 'stdp':
                # gradw.append(((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*( -torch.mm(torch.transpose(trace[i], 0, 1), spike[i + 1]) +  torch.mm(torch.transpose(spike[i],0,1),trace[i+1]) ))
                # gradw.append(((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*( -torch.mm(torch.transpose(spike[i+1], 0, 1), trace[i]) +  torch.mm(torch.transpose(trace[i+1],0,1),spike[i]) ))
                gradw.append(scale_factor*( -torch.mm(torch.transpose(trace[i], 0, 1), spike[i + 1]) +  torch.mm(torch.transpose(spike[i],0,1),trace[i+1]) ))
                gradw.append(scale_factor*( -torch.mm(torch.transpose(spike[i+1], 0, 1), trace[i]) +  torch.mm(torch.transpose(trace[i+1],0,1),spike[i]) ))
            elif self.update_rule == 'nonspikingstdp':
                # gradw.append(((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*( -torch.mm(torch.transpose(trace[i], 0, 1), self.activation(s[i + 1])) +  torch.mm(torch.transpose(self.activation(s[i]),0,1),trace[i+1]) ))
                # gradw.append(((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*( -torch.mm(torch.transpose(self.activation(s[i+1]), 0, 1), trace[i]) +  torch.mm(torch.transpose(trace[i+1],0,1),self.activation(s[i])) ))
                gradw.append(scale_factor*( -torch.mm(torch.transpose(trace[i], 0, 1), self.activation(s[i + 1])) +  torch.mm(torch.transpose(self.activation(s[i]),0,1),trace[i+1]) ))
                gradw.append(scale_factor*( -torch.mm(torch.transpose(self.activation(s[i+1]), 0, 1), trace[i]) +  torch.mm(torch.transpose(trace[i+1],0,1),self.activation(s[i])) ))
            elif self.update_rule == 'cep':
                gradw.append(scale_factor*(torch.mm(torch.transpose(self.activation(s[i]), 0, 1), self.activation(s[i+1])) - torch.mm(torch.transpose(self.activation(seq[i]), 0, 1), self.activation(seq[i+1]))))
                gradw.append(scale_factor*(torch.mm(torch.transpose(self.activation(s[i+1]), 0, 1), self.activation(s[i])) - torch.mm(torch.transpose(self.activation(seq[i+1]), 0, 1), self.activation(seq[i]))))
            elif self.update_rule == 'cepalt':
                gradw.append(scale_factor*( torch.mm(torch.transpose(self.activation(s[i]) - self.activation(seq[i]), 0, 1), self.activation(s[i + 1])) +  torch.mm(torch.transpose(self.activation(s[i]),0,1),self.activation(s[i+1])-self.activation(seq[i+1])) ))
                gradw.append(scale_factor*( torch.mm(torch.transpose(self.activation(s[i+1]), 0, 1), self.activation(s[i]) - self.activation(seq[i])) +  torch.mm(torch.transpose(self.activation(s[i+1])-self.activation(seq[i+1]),0,1),self.activation(s[i])) ))
            if self.use_bias:
                gradw_bias.append(scale_factor*(s[i] - seq[i]).sum(0))
                gradw_bias.append(None)

        if self.update_rule == 'stdp':
            # gradw.append( ((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*(torch.mm(torch.transpose(spike[-2],0,1),trace[-1]) - torch.mm(torch.transpose(trace[-2],0,1), spike[-1]) ))
            gradw.append(scale_factor*(torch.mm(torch.transpose(spike[-2],0,1),trace[-1]) - torch.mm(torch.transpose(trace[-2],0,1), spike[-1]) ))
        elif self.update_rule =='nonspikingstdp':
            # gradw.append( ((1-self.trace_decay)**2/(self.trace_decay*beta*batch_size))*(torch.mm(torch.transpose(self.activation(s[-2]),0,1),trace[-1]) - torch.mm(torch.transpose(trace[-2],0,1), self.activation(s[-1])) ))
            gradw.append(scale_factor*(torch.mm(torch.transpose(self.activation(s[-2]),0,1),trace[-1]) - torch.mm(torch.transpose(trace[-2],0,1), self.activation(s[-1])) ))
        elif self.update_rule == 'skewsym':
            gradw.append(scale_factor*(torch.mm(torch.transpose(self.activation(s[-2]) - self.activation(seq[-2]), 0, 1), self.activation(s[-1])) -  torch.mm(torch.transpose(self.activation(s[-2]),0,1),self.activation(s[-1])-self.activation(seq[-1])) ))
        else:
            gradw.append(scale_factor*torch.mm(torch.transpose(self.activation(s[-2]) - self.activation(seq[-2]), 0, 1), self.activation(s[-1])))
        if self.use_bias:
            gradw_bias.append(scale_factor*(s[-1] - seq[-1]).sum(0))

        return  gradw, gradw_bias

    #**************************NEW**************************#
    def updateWeights(self, gradw):
        lr_tab = [lr/self.M for lr in  self.lr_tab]
        for i in range(len(self.w)):
            if self.w[i] is not None:
                self.w[i].weight += lr_tab[int(np.floor(i/2))]*gradw[0][i]
            if self.use_bias:
                if gradw[1][i] is not None:
                    self.w[i].bias += lr_tab[int(np.floor(i/2))]*gradw[1][i]
