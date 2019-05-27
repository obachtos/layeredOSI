import os
import numpy as np
import matplotlib.pyplot as plt
import style;reload(style); import style as sty
import output; reload(output); import output as out

# import functions; reload(functions); from functions import *



class plot_class():

    def OSI(self, arr):

        F1 = np.sum(arr * np.exp(1j*2.*self.angles), axis=1)
        nOSV = np.sum(arr, axis=1)

        F1[nOSV==0] = 0.
        nOSV[nOSV==0] = 1.

        OSVs_pop = F1/nOSV

        POs = np.angle(OSVs_pop)        
        POs[POs<0] = POs[POs<0] + 2.*np.pi
        POs = POs/(2.*np.pi)*180.

        OSIs = np.abs(OSVs_pop)
        OSIs[np.isnan(OSIs)] = 0.

        return OSIs, POs, F1

    def plotTuningCurve(self, angles, rates, title=None, rect=True, save_folder=None):

        tmp = np.int(np.ceil(np.sqrt(rates.shape[0])))
        sbplts = tmp, tmp

        f, axs_rec = plt.subplots(sbplts[0],sbplts[1], figsize=(12,12), sharex=True)#,sharey=True)
        axs = axs_rec.flatten()
        # axs2 = []

        angles = np.concatenate((angles/np.pi * 180., [180.]))
        rates = np.concatenate((rates, rates[:,0,:].reshape((rates.shape[0],1,rates.shape[2]))), axis=1)

        for i in range(np.min([np.prod(sbplts), rates.shape[0]])):

            #style.hideAxes(axs[i])

            tmp = rates[i,:,1]
            mrkr = '-'
            if rect and tmp[0] is not None:
                tmp[tmp<0.] = 0.
                axs[i].plot(angles, tmp, '-', c=sty.color_cycle[1], label='linearization')
                mrkr = '--'

            axs[i].plot(angles, rates[i,:,0], '-', c=sty.color_cycle[0], label='spiking sim.')
            if rates[i,0,1] is not None:
                axs[i].plot(angles, rates[i,:,1], mrkr, c=sty.color_cycle[1], label='linearization')
            if rates[i,0,2] is not None:
                axs[i].plot(angles, rates[i,:,2], '-', c=sty.color_cycle[2], label='siegert')

            # axs[i].tick_params('y', color=style.color_cycle[0])

            # axs2.append(axs[i].twinx())
            # axs2[i].plot(angles, rates[i,:,1], c=style.color_cycle[1], label='output')#, label='linearization')
            # axs2[i].tick_params('y', color=style.color_cycle[1])

            plt.setp(axs[i], xticks=np.arange(0,182,30))

            axs[i].set_ylim([0., axs[i].get_ylim()[1]])

        axs_rec[-1,0].set_xlabel('stim. angle [$^\circ$] ')
        axs_rec[-1,0].set_ylabel(r'$\nu$ [kHz]')

        axs[-1].legend(loc='best')

        if title is not None:
            plt.suptitle(title)

        if save_folder is not None:
            plt.savefig(save_folder + title + '_tuning_curve.png')
            # plt.savefig(save_folder + title + 'linear_comparison.pdf')

    def vsPlot(self, spk, sieg, lin, title=None, save_folder=None):

        f, axs_rec = plt.subplots(4,3, figsize=(12,12))#$, sharex=True)#,sharey=True)
        axs = axs_rec.flatten()

        for i in range(3):
            if   i == 0: 
                xdata = spk
                ydata = sieg
            elif i == 1:
                xdata = sieg
                ydata = lin
            elif i == 2:
                xdata = spk
                ydata = lin

            if xdata is None or ydata is None:
                continue

            for p, pop in enumerate(self.net.populations):
                if p%2 == 0:
                    ax = axs_rec[p/2, i]
                    style.hideAxes(ax)

                ax.plot(xdata[p].flatten(), ydata[p].flatten(), 
                        '.', c=sty.pop_colors[p], ms=4)

                mx = np.max([np.max(xdata[p]), np.max(ydata[p])])

                ax.plot([0., mx], [0., mx], c='k', lw=1.)

        axs_rec[-1,0].set_xlabel('spiking') 
        axs_rec[-1,1].set_xlabel('siegert') 
        axs_rec[-1,2].set_xlabel('spiking') 
        for i in range(4):
            axs_rec[i,0].set_ylabel('siegert')
            axs_rec[i,1].set_ylabel('linear')
            axs_rec[i,2].set_ylabel('linear')

        plt.suptitle(title)

        plt.tight_layout()

        if save_folder is not None:
            plt.savefig(save_folder + title + '_VS.png')
            # plt.savefig(save_folder + title + 'linF1_vs_spkgF1.pdf')

    def plotPrediction(self, lin_point, size='full'):

        if size == 'full':
            sizes = self.n_neurons
        elif size == 'small':
            sizes = self.n_neurons_small

        save_folder=self.model_def_path+'/analysis/linearization_%s/'%lin_point
        
        try:    os.mkdir(save_folder)
        except: pass

        ### collect first n neurons (temporary purpose) ###
        ### for tuning curves                           ###
        n = 3**2

        # spiking
        spk_rates_sp = np.zeros((len(self.net.populations), n, len(self.angles)))
        spk_rates_th = np.zeros((len(self.net.populations), n, len(self.angles)))
        for p, pop in enumerate(self.net.populations):
            spk_rates_sp[p,:,:] = self.v_sp[p][:n,:]
            spk_rates_th[p,:,:] = self.v_th[p][:n,:]

        # siegert
        sieg_rates_sp  = np.zeros(spk_rates_sp.shape)
        if self.sieg_rates_sp is not None:
            for p, pop in enumerate(self.net.populations):  
                for a in range(len(self.angles)):
                    sieg_rates_sp[p,:,a] = self.sieg_rates_sp[p][:n,a]
        else:
            sieg_rates_sp = sieg_rates_sp * np.nan

        sieg_rates_th  = np.zeros(spk_rates_sp.shape)
        if self.sieg_rates_th is not None:
            for p, pop in enumerate(self.net.populations):  
                for a in range(len(self.angles)):
                    sieg_rates_th[p,:,a] = self.sieg_rates_th[p][:n,a]
        else:
            sieg_rates_th = sieg_rates_th * np.nan

        # linear
        dlin_rates_th = np.zeros(spk_rates_sp.shape)
        lin_rates_th = np.zeros(spk_rates_sp.shape)        
        for p, pop in enumerate(self.net.populations):  
            for a in range(len(self.angles)):      
                dlin_rates_th[p,:,a] = self.dlin_rates_th[p][:n,a]
                lin_rates_th[p,:,a] = self.lin_rates_th[p][:n,a]
        

        ### calculate OSIs and POs ###
        self.spk_F1s = []
        self.lin_OSIs = []
        self.lin_POs = []
        self.lin_F1s = []
        self.sieg_OSIs = []
        self.sieg_POs = []
        self.sieg_F1s = []

        for p in range(len(self.net.populations)):
            OSI_tmp, PO_tmp, F1s_tmp = self.OSI(self.lin_rates_th[p])
            self.lin_OSIs.append(OSI_tmp)
            self.lin_POs.append(PO_tmp)
            self.lin_F1s.append(F1s_tmp)
            
            OSI_tmp, PO_tmp, F1s_tmp = self.OSI(self.sieg_rates_th[p])
            self.sieg_OSIs.append(OSI_tmp)
            self.sieg_POs.append(PO_tmp)
            self.sieg_F1s.append(F1s_tmp)

            self.spk_F1s.append(self.OSI(self.v_th[p])[-1])


        ### VS. plots ###
        # sp rates 
        self.vsPlot(self.v_sp, self.sieg_rates_sp, self.lin_rates_sp, 
                        title='rates_sp', save_folder=save_folder)

        # th rates 
        self.vsPlot(self.v_th, self.sieg_rates_th, self.lin_rates_th, 
                        title='rates_th', save_folder=save_folder)


        # PO
        self.vsPlot(self.POs, self.sieg_POs, self.lin_POs, 
                        title='POs', save_folder=save_folder)        

        # OSI
        self.vsPlot(self.OSIs, self.sieg_OSIs, self.lin_OSIs, 
                        title='OSIs', save_folder=save_folder)    

        # F1
        # self.vsPlot(self.spk_F1s, self.sieg_F1s, self.lin_F1s, 
        #                 title='F1s', save_folder=save_folder) 


        ### plot tuning curves (spontaneous)
        # for p, pop in enumerate(self.net.populations):
        #     rates = np.dstack((spk_rates_sp[p,:,:], lin_rates_th[p,:,:]*np.nan, sieg_rates_sp[p,:,:]))
            
        #     self.plotTuningCurve(self.angles, rates, rect=True, title='sp_' + pop, 
        #                             save_folder=save_folder)

        ### plot tuning curves (thalamic)
        for p, pop in enumerate(self.net.populations):
            rates = np.dstack((spk_rates_th[p,:,:], lin_rates_th[p,:,:], sieg_rates_th[p,:,:])) 

            self.plotTuningCurve(self.angles, rates, rect=True, title='th_' + pop, 
                                    save_folder=save_folder)
