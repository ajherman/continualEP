import numpy as np
import matplotlib.pyplot as plt
import os, sys
import pickle
import torch

fontsize = 13

def plot_T(NT, DT, *arg): 
    if not isinstance(NT[0], list):
        args = arg[0]
        toymodel = args.toymodel
        learning_rule = args.learning_rule

        if not toymodel:
            
            if args.learning_rule == 'vf':

                fig = plt.figure()
                #plt.subplots_adjust(hspace = 1)
                plt.rcParams.update({'font.size': fontsize})
                N = int((len(NT) - 1)/2)
                for i in range(N):
                    plt.subplot(len(NT), 1, 2*i +1)
                    for j in range(10):
                        ind_temp0, ind_temp1 = np.random.randint(NT[2*i][0, :, :].size(0)), np.random.randint(NT[2*i][0, :, :].size(1))
                        plt.plot(NT[2*i][:, ind_temp0, ind_temp1].cpu().numpy(), label='NT'+str(2*i) + str(2*i+1)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j))
                        plt.plot(DT[2*i][:, ind_temp0, ind_temp1].cpu().numpy(), label='DT'+ str(2*i) + str(2*i+1)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j),linestyle='--')
                    plt.xlabel('t')
                    plt.title(r'$\Delta_{W_{' + str(2*i) + str(2*i + 1)+r'}}^{\rm EP}$, $-\nabla_{W_{' + str(2*i) + str(2*i + 1)+r'}}^{\rm BPTT}$')
                    plt.grid()
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                    plt.subplot(len(NT), 1, 2*i + 2)
                    for j in range(10):
                        ind_temp0, ind_temp1 = np.random.randint(NT[2*i + 1][0, :, :].size(0)), np.random.randint(NT[2*i + 1][0, :, :].size(1))
                        plt.plot(NT[2*i + 1][:, ind_temp0, ind_temp1].cpu().numpy(), label='NT'+str(2*i + 1) + str(2*i)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j))
                        plt.plot(DT[2*i + 1][:, ind_temp0, ind_temp1].cpu().numpy(), label='DT'+ str(2*i + 1) + str(2*i)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j),linestyle='--')
                    plt.xlabel('t')
                    plt.title(r'$\Delta_{W_{' + str(2*i + 1) + str(2*i)+r'}}^{\rm EP}$, $-\nabla_{W_{' + str(2*i + 1) + str(2*i)+r'}}^{\rm BPTT}$')      
                    plt.grid()
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                plt.subplot(len(NT), 1, len(NT))
                for i in range(10):
                    ind_tempx, ind_temp1 = np.random.randint(NT[-1][0, :, :].size(0)), np.random.randint(NT[-1][0, :, :].size(1))
                    plt.plot(NT[-1][:, ind_tempx, ind_temp1].cpu().numpy(), label='NT' + str(len(NT))+'['+str(ind_tempx)+str(ind_temp1)+']',color='C'+str(i))
                    plt.plot(DT[-1][:, ind_tempx, ind_temp1].cpu().numpy(), label='DT'+ str(len(NT))+'['+str(ind_tempx)+str(ind_temp1)+']',color='C'+str(i),linestyle='--')
                plt.xlabel('t')
                plt.title(r'$\Delta_{W_{'+ str(N - 1) +r'x}}^{\rm EP}$, $-\nabla_{W_{'+ str(N - 1) +r'x}}^{\rm BPTT}$')
                plt.grid()
                plt.subplots_adjust(hspace = 0.5)
                fig.tight_layout()
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            else:
                fig = plt.figure()
                plt.rcParams.update({'font.size': fontsize})
                N = int((len(NT) - 1)/2) + 1 
                for i in range(N - 1):
                    plt.subplot(N, 1, 1 + i)
                    for j in range(10):
                        ind_temp0, ind_temp1 = np.random.randint(NT[2*i][0, :, :].size(0)), np.random.randint(NT[2*i][0, :, :].size(1))
                        plt.plot(NT[2*i][:, ind_temp0, ind_temp1].cpu().numpy(), 
                                label='NT'+str(2*i) + str(2*i+1)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j))
                        plt.plot(DT[2*i][:, ind_temp0, ind_temp1].cpu().numpy(), 
                                label='DT'+ str(2*i) + str(2*i+1)+'['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j),linestyle='--')
                    plt.xlabel('t')
                    plt.title(r'$\Delta_{W_{' + str(i) + str(i + 1)+r'}}^{\rm EP}$, $-\nabla_{W_{' + str(i) + str(i + 1)+r'}}^{\rm BPTT}$')
                    plt.grid()
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


                plt.subplot(N, 1, N)
                for i in range(10):
                    ind_tempx, ind_temp1 = np.random.randint(NT[-1][0, :, :].size(0)), np.random.randint(NT[-1][0, :, :].size(1))
                    plt.plot(NT[-1][:, ind_tempx, ind_temp1].cpu().numpy(), 
                            label='NT' + str(len(NT))+'['+str(ind_tempx)+str(ind_temp1)+']',color='C'+str(i))
                    plt.plot(DT[-1][:, ind_tempx, ind_temp1].cpu().numpy(), 
                            label='DT'+ str(len(NT))+'['+str(ind_tempx)+str(ind_temp1)+']',color='C'+str(i),linestyle='--')
                plt.xlabel('t')
                plt.title(r'$\Delta_{W_{'+ str(N - 1) +r'x}}^{\rm EP}$, $-\nabla_{W_{'+ str(N - 1) +r'x}}^{\rm BPTT}$')
                plt.grid()
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

                plt.subplots_adjust(hspace = 0.5)
                fig.tight_layout()
        

        else:
            fig = plt.figure(figsize = (5, 8))     
            plt.rcParams.update({'font.size': fontsize})      
            plt.subplot(3, 1, 1)
            for j in range(5):
                ind_temp0, ind_temp1 = np.random.randint(NT[1][0, :, :].size(0)), np.random.randint(NT[1][0, :, :].size(1))
                plt.plot(NT[1][:, ind_temp0, ind_temp1].cpu().numpy(), 
                        label='NT01['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j))
                plt.plot(DT[1][:, ind_temp0, ind_temp1].cpu().numpy(), 
                        label='DT01['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j),linestyle='--')
            plt.xlabel('t')
            plt.title(r'$\Delta_{W_{01}}^{\rm EP}$, $-\nabla_{W_{01}}^{\rm BPTT}$')
            plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            
            plt.subplot(3, 1, 2)
            for j in range(5):
                ind_temp0, ind_temp1 = np.random.randint(NT[2][0, :, :].size(0)), np.random.randint(NT[2][0, :, :].size(1))
                plt.plot(NT[2][:, ind_temp0, ind_temp1].cpu().numpy(), 
                        label='NT0x['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j))
                plt.plot(DT[2][:, ind_temp0, ind_temp1].cpu().numpy(), 
                        label='DT0x['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j),linestyle='--')
            plt.xlabel('t')
            plt.title(r'$\Delta_{W_{0x}}^{\rm EP}$, $-\nabla_{W_{0x}}^{\rm BPTT}$')
            plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))         


            plt.subplot(3, 1, 3)
            for j in range(5):
                ind_temp0, ind_temp1 = np.random.randint(NT[4][0, :, :].size(0)), np.random.randint(NT[4][0, :, :].size(1))
                plt.plot(NT[4][:, ind_temp0, ind_temp1].cpu().numpy(), 
                        label='NT11['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j))
                plt.plot(DT[4][:, ind_temp0, ind_temp1].cpu().numpy(), 
                        label='DT11['+str(ind_temp0)+str(ind_temp1)+']',color='C'+str(j),linestyle='--')
            plt.xlabel('t')
            plt.title(r'$\Delta_{W_{1x}}^{\rm EP}$, $-\nabla_{W_{1x}}^{\rm BPTT}$')
            plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            plt.subplots_adjust(hspace = 0.5)
            fig.tight_layout()       

    else:
        NT_conv, NT_fc = NT[0], NT[1]     
        DT_conv, DT_fc = DT[0], DT[1]
        
        n_conv = len(NT_conv)
        n_fc = len(NT_fc)

        fig = plt.figure(figsize = (5, 10))
        #fig = plt.figure(figsize = (5, 8))
        #plt.rcParams.update({'font.size': 17})
        plt.rcParams.update({'font.size': 10})
        
        for i in range(n_fc):
            plt.subplot(n_conv + n_fc, 1, 1 + i)
            for j in range(5):
                ind_temp0, ind_temp1 = np.random.randint(NT_fc[i][0, :].size(0)), np.random.randint(NT_fc[i][0, :].size(1))
                plt.plot(NT_fc[i][:, ind_temp0, ind_temp1].cpu().numpy(), color='C'+str(j))
                plt.plot(DT_fc[i][:, ind_temp0, ind_temp1].cpu().numpy(), color='C'+str(j),linestyle = '--')
            plt.xlabel('t')
            plt.title(r'$\Delta_{W^{\rm fc}_{' + str(i) + str(i + 1)+r'}}^{\rm EP}$, $-\nabla_{W^{\rm fc}_{' + str(i) + str(i + 1)+r'}}^{\rm BPTT}$', fontsize = 10)
            plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))


        for i in range(n_conv):
            plt.subplot(n_conv + n_fc, 1, 1 + n_fc + i)
            for j in range(5):
                ind_temp0, ind_temp1, ind_temp2, ind_temp3 = np.random.randint(NT_conv[i][0, :].size(0)), np.random.randint(NT_conv[i][0, :].size(1)), np.random.randint(NT_conv[i][0, :].size(2)), np.random.randint(NT_conv[i][0, :].size(3))
                plt.plot(NT_conv[i][:, ind_temp0, ind_temp1, ind_temp2, ind_temp3].cpu().numpy(), color='C'+str(j))
                plt.plot(DT_conv[i][:, ind_temp0, ind_temp1, ind_temp2, ind_temp3].cpu().numpy(), color='C'+str(j),linestyle = '--')
            plt.xlabel('t')
            plt.title(r'$\Delta_{W^{\rm conv}_{' + str(i) + str(i + 1)+r'}}^{\rm EP}$, $-\nabla_{W^{\rm conv}_{' + str(i) + str(i + 1)+r'}}^{\rm BPTT}$', fontsize = 10)
            plt.grid()
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        fig.tight_layout()
        plt.subplots_adjust(hspace = 1)      


def plot_S(nS, dS):

    if not (len(nS[-1].size()) >= 4):
        fig = plt.figure(figsize = (5, 5))
        #plt.figure(figsize = (5, 8))
        plt.rcParams.update({'font.size': fontsize})  
        for i in range(len(nS)):       
            plt.subplot(len(nS), 1, 1 + i)
            for j in range(5):
                if (i > 0):
                    n = np.random.randint(nS[i].size(2))
                    plt.plot(nS[i][:, 0, n].cpu().numpy(),label='nS'+str(i)+'['+str(n)+']',color='C'+str(j))
                    plt.plot(dS[i][:, 0, n].cpu().numpy(), label='dS'+str(i)+'['+str(n)+']',color='C'+str(j),linestyle='--')
                else:
                    plt.plot(nS[i][:, 0, j].cpu().numpy(),label='nS'+str(i)+'['+str(j)+']',color='C'+str(j))
                    plt.plot(dS[i][:, 0, j].cpu().numpy(), label='dS'+str(i)+'['+str(j)+']',color='C'+str(j),linestyle='--')
      
            plt.xlabel('t')
            plt.title(r'$\Delta_{s_{' + str(i) +r'}}^{\rm EP}$, $-\nabla_{s_{' + str(i) +r'}}^{\rm BPTT}$')   
            plt.grid()    
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.subplots_adjust(hspace = 0.5)
            fig.tight_layout()

    else:        
        #plt.figure(figsize = (5, 8))
        fig = plt.figure(figsize = (5, 10))
        #plt.subplots_adjust(hspace = 0.7)
        #plt.rcParams.update({'font.size': 17})
        plt.rcParams.update({'font.size': 10})
        for i in range(len(nS)):       
            plt.subplot(len(nS), 1, 1 + i)
            for j in range(8):
                #Last classifier layer
                if (i == 0):
                    plt.plot(nS[i][:, 0, j].cpu().numpy(),label='nS'+str(i)+'['+str(j)+']',color='C'+str(j))
                    plt.plot(dS[i][:, 0, j].cpu().numpy(), label='dS'+str(i)+'['+str(j)+']',color='C'+str(j),linestyle='--')  
                     
                #Middle classifier layers         
                elif ((i > 0) & (len(nS[i].size()) < 4)):
                    n = np.random.randint(nS[i].size(2))
                    plt.plot(nS[i][:, 0, n].cpu().numpy(),label='nS'+str(i)+'['+str(n)+']',color='C'+str(j))
                    plt.plot(dS[i][:, 0, n].cpu().numpy(), label='dS'+str(i)+'['+str(n)+']',color='C'+str(j),linestyle='--')
                    
                #Conv layers
                elif (len(nS[i].size()) >= 4):
                    n, m, p = np.random.randint(nS[i].size(2)), np.random.randint(nS[i].size(3)), np.random.randint(nS[i].size(4))
                    plt.plot(nS[i][:, 0, n, m, p].cpu().numpy(),label='nS'+str(i)+'['+str(n)+str(m)+str(p) + ']',color='C'+str(j))
                    plt.plot(dS[i][:, 0, n, m, p].cpu().numpy(), label='dS'+str(i)+'['+str(n)+str(m)+str(p) + ']',color='C'+str(j),linestyle='--')                

            plt.title(r'$\Delta_{s_{' + str(i) +r'}}^{\rm EP}$, $-\nabla_{s_{' + str(i) +r'}}^{\rm BPTT}$', fontsize = 10)   
            plt.grid()    
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))  
        plt.xlabel('t', fontsize = 10)
        plt.subplots_adjust(hspace = 0.5)
        fig.tight_layout()
                     
        
def compute_nTdT(NT, DT):

    if not isinstance(NT[0], list):
        nT = []
        dT = []

        for i in range(len(NT)):
            if NT[i] is not None:
                nT.append(torch.zeros_like(NT[i]))
                dT.append(torch.zeros_like(DT[i]))
            else:
                nT.append(None)
                dT.append(None)            

        for i in range(len(NT)):
            if NT[i] is not None:
                for t in range(NT[i].size(0) - 1):
                    nT[i][t + 1, :, :] = NT[i][t + 1, :, :] - NT[i][t, :, :]
                    dT[i][t + 1, :, :] = DT[i][t + 1, :, :] - DT[i][t, :, :]
        
        return nT, dT


    else:

        nT_conv = []
        nT_fc = []    
        dT_conv = []
        dT_fc = []

        NT_conv = NT[0]
        NT_fc = NT[1]
        DT_conv = DT[0]
        DT_fc = DT[1]    
        
        for i in range(len(NT_fc)):
                nT_fc.append(torch.zeros_like(NT_fc[i]))
                dT_fc.append(torch.zeros_like(DT_fc[i]))
                
        for i in range(len(NT_conv)):
                nT_conv.append(torch.zeros_like(NT_conv[i]))
                dT_conv.append(torch.zeros_like(DT_conv[i]))   

        for i in range(len(NT_fc)):
                for t in range(NT_fc[i].size(0) - 1):
                    nT_fc[i][t + 1, :, :] = NT_fc[i][t + 1, :, :] - NT_fc[i][t, :, :]
                    dT_fc[i][t + 1, :, :] = DT_fc[i][t + 1, :, :] - DT_fc[i][t, :, :]
                    
        for i in range(len(NT_conv)):
                for t in range(NT_conv[i].size(0) - 1):
                    nT_conv[i][t + 1, :, :] = NT_conv[i][t + 1, :, :] - NT_conv[i][t, :, :]
                    dT_conv[i][t + 1, :, :] = DT_conv[i][t + 1, :, :] - DT_conv[i][t, :, :]
        
        return [nT_conv, nT_fc], [dT_conv, dT_fc]        
        
def compute_Hist(nS, dS, NT, DT):
    hist_S_mean = []
    hist_T_mean = []
    hist_S_std = []
    hist_T_std = []
            
      
    for i in range(len(dS)):
        err = torch.where(((dS[i]**2).sum(0) == 0 )& ((nS[i]**2).sum(0) == 0), torch.zeros_like(dS[i][0, :]),
                            torch.sqrt(torch.div(((nS[i] - dS[i])**2).sum(0), torch.max( (nS[i]**2).sum(0),(dS[i]**2).sum(0)))))
        hist_S_mean.append(err.mean().item())
        hist_S_std.append(err.std().item())
        
                
        del err
    
    if not (len(nS[-1].size()) >= 4):
        for i in range(len(DT)):
            if NT[i] is not None:
            
                err = torch.where(((DT[i]**2).sum(0) == 0 )& ((NT[i]**2).sum(0) == 0), torch.zeros_like(DT[i][0, :]),
                                    torch.sqrt(torch.div(((NT[i] - DT[i])**2).sum(0), torch.max((NT[i]**2).sum(0),(DT[i]**2).sum(0)))))
                hist_T_mean.append(err.mean().item())
                hist_T_std.append(err.std().item())

                del err
                
    else:
        NT_conv, NT_fc = NT
        DT_conv, DT_fc = DT
        for i in range(len(DT_fc)):
            if NT_fc[i] is not None: 
                err = torch.where(((DT_fc[i]**2).sum(0) == 0 )&((NT_fc[i]**2).sum(0) == 0), torch.zeros_like(DT_fc[i][0, :]),
                                    torch.sqrt(torch.div(((NT_fc[i] - DT_fc[i])**2).sum(0), torch.max((NT_fc[i]**2).sum(0),(DT_fc[i]**2).sum(0)))))
                hist_T_mean.append(err.mean().item())
                hist_T_std.append(err.std().item())
                
                
                del err       
                 
        for i in range(len(DT_conv)):
            print('size: {}'.format(DT_conv[i].size()))
            if NT_conv[i] is not None: 
                err = torch.where(((DT_conv[i]**2).sum(0) == 0 )&((NT_conv[i]**2).sum(0) == 0), torch.zeros_like(DT_conv[i][0, :]),
                                    torch.sqrt(torch.div(((NT_conv[i] - DT_conv[i])**2).sum(0), torch.max((NT_conv[i]**2).sum(0),(DT_conv[i]**2).sum(0)))))
                hist_T_mean.append(err.mean().item())
                hist_T_std.append(err.std().item())
                     
                del err                

            
    return [hist_S_mean, hist_S_std] , [hist_T_mean, hist_T_std]        

def plot_Hist(hist_S, hist_T, NT, args):
    ind = []
    ind_names = []
    fig = plt.figure(figsize = (5, 5))
    plt.rcParams.update({'font.size': fontsize})
    for i in range(len(hist_S[0])):
        plt.bar(i, hist_S[0][i], width = 1, label = r'$s^{'+str(i)+'}$', alpha = 0.5)
        ind.append(i)
        ind_names.append(r'$s^{'+str(i)+'}$')

    if not isinstance(NT[0], list):

        if (args.learning_rule == 'ep') & (not args.toymodel):
            for i in range(len(hist_T[0]) - 1):
                plt.bar(len(hist_S[0]) + 1 + i, hist_T[0][i], 
                        width = 1, label = r'$W_{'+str(i) + str(i + 1) +'}$', alpha = 0.4)
                ind.append(len(hist_S[0]) + 1 + i)
                ind_names.append(r'$W_{'+str(i) + str(i + 1) +'}$')
            
            plt.bar(len(hist_S[0]) + len(hist_T[0]), hist_T[0][-1], 
                    width = 1, label = r'$W_{'+str(len(hist_T[0]) - 1) +'x}$', alpha = 0.4)
            ind.append(len(hist_S[0]) + len(hist_T[0]))
            ind_names.append(r'$W_{'+str(len(hist_T[0]) - 1) +'x}$')


        elif (args.learning_rule == 'vf') & (not args.toymodel):
            for i in range(int((len(hist_T[0]) - 1)/2)):
                plt.bar(len(hist_S[0]) + 1 + 2*i, hist_T[0][2*i], 
                        width = 1, label = r'$W_{'+str(i) + str(i + 1) +'}$', alpha = 0.4)
                ind.append(len(hist_S[0]) + 1 + 2*i)
                ind_names.append(r'$W_{'+str(i) + str(i + 1) +'}$')

                plt.bar(len(hist_S[0]) + 1 + 2*i + 1, hist_T[0][2*i + 1], 
                        width = 1, label = r'$W_{'+str(i + 1) + str(i) +'}$', alpha = 0.4)
                ind.append(len(hist_S[0]) + 1 + 2*i + 1)
                ind_names.append(r'$W_{'+str(i + 1) + str(i) +'}$')
            
            plt.bar(len(hist_S[0]) + len(hist_T[0]), hist_T[0][-1], 
                    width = 1, label = r'$W_{'+str(int((len(hist_T[0]) - 1)/2)) +'x}$', alpha = 0.4)
            ind.append(len(hist_S[0]) + len(hist_T[0]))
            ind_names.append(r'$W_{'+str(int((len(hist_T[0]) - 1)/2)) +'x}$')


        elif (args.learning_rule == 'vf') & (args.toymodel):

            ind_W = ['00', '01', '0x', '11', '1x', '10'] 
                
            for i in range(len(hist_T[0])):
                plt.bar(len(hist_S[0]) + 1 + i, hist_T[0][i], 
                        width = 1, label = r'$W_{'+ ind_W[i] +'}$', alpha = 0.4)
                ind.append(len(hist_S[0]) + 1 + i)
                ind_names.append(r'$W_{'+ ind_W[i] +'}$')                    
    else:
        nconv = len(NT[0])
        nfc = len(NT[1]) 
        for i in range(nfc):
            plt.bar(len(hist_S[0]) + 1 + i, hist_T[0][i], width = 1, label = r'$W_{'+str(i) + str(i + 1) +r'}^{\rm fc}$', alpha = 0.4)
            ind.append(len(hist_S[0]) + 1 + i)
            ind_names.append(r'$W_{'+str(i) + str(i + 1) +r'}^{\rm fc}$')   

        for i in range(nconv - 1):
            plt.bar(len(hist_S[0]) + 1 + nfc + i, hist_T[0][nfc + i], width = 1, label = r'$W_{'+str(i) + str(i + 1) +r'}^{\rm conv}$', alpha = 0.4)
            ind.append(len(hist_S[0]) + 1 + nfc + i)
            ind_names.append(r'$W_{'+str(i) + str(i + 1) +r'}^{\rm conv}$')            
            
        plt.bar(len(hist_S[0]) + nfc + nconv, hist_T[0][nfc + nconv - 1], width = 1, label = r'$W_{'+str(len(hist_T[0]) - 1)+r'x}^{\rm conv}$', alpha = 0.4)
        ind.append(len(hist_S[0]) + nfc + nconv)
        ind_names.append(r'$W_{'+str(nconv - 1)+r'x}^{\rm conv}$')   

    fig.tight_layout()
    plt.xticks(ind, ind_names, fontsize = fontsize)    
    plt.grid() 

def plot_results(what, *arg):
    if what == 'error':     
        error_train_tab = arg[0]
        error_test_tab = arg[1]
        epochs = len(error_train_tab)
        plt.figure(figsize=(5, 5))
        plt.plot(np.linspace(1, epochs, epochs), error_train_tab, label = 'training error')
        plt.plot(np.linspace(1, epochs, epochs), error_test_tab, label = 'test error')
        plt.legend(loc = 'best')
        plt.xlabel('Epochs')
        plt.ylabel('Error rate (%)')
        plt.grid()


if __name__ == '__main__':
    BASE_PATH = os.getcwd() + '/results' 
    infile = open(BASE_PATH,'rb')
    results_dict = pickle.load(infile)
    infile.close()

    if 'nS' in results_dict:
        nS = results_dict['nS']
        dS = results_dict['dS']
        nT = results_dict['nT']
        dT = results_dict['dT']
        args = results_dict['args'] 
        #nT, dT = compute_nTdT(NT, DT)    
        hist_S, hist_T = compute_Hist(nS, dS, nT, dT)
        plot_Hist(hist_S, hist_T, nT, args)   
        plt.show()
        plot_S(nS, dS)
                  
        plot_T(nT, dT, args)                                 
        plt.show()
    
    if 'error_train_tab' in results_dict:
        plot_results('error', results_dict['error_train_tab'], results_dict['error_test_tab'])
        plt.title('EP')

    if 'error_train_bptt_tab' in results_dict:
        plot_results('error', results_dict['error_train_bptt_tab'], results_dict['error_test_bptt_tab'] )
        plt.title('BPTT')
	
    plt.show()   
    
        



    

