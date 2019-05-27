import os
import numpy as np
import matplotlib.pyplot as plt
import style;reload(style); import style as sty
import output; reload(output); import output as out

import functions; reload(functions); from functions import *



class plot_EV_class():

    def plotEVs(self, which=(1,1)):

        save_folder = self.model_def_path+'/analysis/linearization_EVs/'

        try:    os.mkdir(save_folder)
        except: pass
 

        h = plt.figure(figsize=(5,4))

        def my_outlier(es, n_comp=100):
            return np.abs(es) > 1.1 * np.median(np.abs(es[:n_comp]))

        if self.evs is not None and which[0]:

            idxs = my_outlier(self.evs, 100)

            # bulk
            plt.plot(np.real(self.evs[np.logical_not(idxs)]), 
                     np.imag(self.evs[np.logical_not(idxs)]),
                     '.', ms=.5, alpha=.2, c=sty.color_cycle[0])

            # exceptional
            plt.plot(np.real(self.evs[idxs]), 
                     np.imag(self.evs[idxs]),
                     '.', ms=10., c=sty.color_cycle[1])

        if self.Qevs is not None and which[1]:
            idxs = np.abs(self.Qevs) > 1e-10
            plt.plot(np.real(self.Qevs[idxs]), np.imag(self.Qevs[idxs]), 'kx', ms=8.)

        # circle
        xs = np.linspace(-1,1,100)
        ys = np.sqrt(1-xs**2.)
        plt.plot(xs, ys, 'k-', lw=1., alpha=.5)
        plt.plot(xs, -ys, 'k-', lw=1., alpha=.5)

        # lines
        ax = plt.gca()
        xlims, ylims = ax.get_xlim(), ax.get_ylim()
        plt.plot([0., 0.], [-1000., 1000.], 'k--', lw=1, zorder=-10)
        plt.plot([-1000., 1000.], [0., 0.], 'k--', lw=1, zorder=-10)
        plt.xlim(xlims)
        plt.ylim(ylims)

        out.cleanAxis(ax)

        ax.set_aspect('equal', 'datalim')

        plt.xlabel(r'$\mathfrak{R}(\lambda)$')
        plt.ylabel(r'$\mathfrak{I}(\lambda)$')

        h.tight_layout()

        if which == (1,1):
            ext = ''
        elif which == (1,0):
            ext = '_W'
        elif which == (0,1):
            ext = '_Q'
        else:
            ext = '_this_is_empty'
        if save_folder is not None:
            plt.savefig(save_folder + 'EVs'+ext+'.png')
            plt.savefig(save_folder + 'EVs'+ext+'.pdf')

    def plotEVecs(self, source='std'):

        if source == 'std':
            evs, evecs = self.evs, self.evecs
            pre_str = ''

            arr = np.concatenate((self.evs.real.reshape((-1,1)), self.evs.imag.reshape((-1,1))), axis=1)
            idxs = np.where(is_outlier(arr, thresh=3.1))[0]

        elif source == 'Q':
            evs, evecs = self.Qevs, self.Qevecs
            pre_str = 'Q'

            idxs = np.arange(8)

        save_folder = self.model_def_path+'/analysis/linearization_EVs/'        
        try:    os.mkdir(save_folder)
        except: pass

        # idxs = np.where(np.abs(evs) >= 1.)[0]
        # rnd_idxs = np.random.randint(idxs.size+1, evs.size, size=8)

        ### magnitudes ###
        # exceptional
        h = plt.figure(figsize=(6,4))
        for i in range(idxs.size):
            plt.plot(np.abs(evecs[:,idxs[i]]), '-', 
                     alpha=.3,
                     color=sty.color_cycle[i])

            I = 0
            plt_idxs = []
            mns = []
            for p in range(8):
                mn = np.mean(np.abs(evecs[I:I+self.n_neurons[p],idxs[i]]))

                plt_idxs.append(I)
                plt_idxs.append(I+self.n_neurons[p])
                mns.append(mn)
                mns.append(mn)

                I += self.n_neurons[p]

            plt.plot(plt_idxs, mns, '-', color=sty.color_cycle[i],
                     label=str(idxs[i]) + ' %.2f'%np.abs(evs[idxs[i]]))


        # plt.legend(loc='best')        
        # plt.xlabel('neuron ID')
        # plt.ylabel('|' + pre_str + 'EVec|')
        plt.ylabel(r'|$\Psi$|')
        # plt.title('exceptional EVs')

        xticks = np.concatenate(([0], np.cumsum(self.n_neurons)[:-1])) + self.n_neurons/2.
        xticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
        plt.xticks(xticks, xticklabels, rotation=60)

        plt.setp(plt.gca(), yticklabels=[])

        out.cleanAxis(plt.gca())
    
        h.tight_layout()

        if save_folder is not None:
            plt.savefig(save_folder + pre_str + 'EVecs_mag1.png')
            plt.savefig(save_folder + pre_str + 'EVecs_mag1.pdf')

        # regular
        # plt.figure()
        # for i in range(rnd_idxs.size):
        #     plt.plot(np.abs(evecs[:,rnd_idxs[i]]), '-', 
        #              alpha=.1,
        #              color=sty.color_cycle[i])

        #     I = 0
        #     plt_idxs = []
        #     mns = []
        #     for p in range(8):
        #         mn = np.mean(np.abs(evecs[I:I+self.n_neurons[p],rnd_idxs[i]]))

        #         plt_idxs.append(I)
        #         plt_idxs.append(I+self.n_neurons[p])
        #         mns.append(mn)
        #         mns.append(mn)

        #         I += self.n_neurons[p]

        #     plt.plot(plt_idxs, mns, '-', color=sty.color_cycle[i],
        #              label=str(rnd_idxs[i]) + ' %.2f'%np.abs(evs[rnd_idxs[i]]))

        # plt.legend(loc='best')        
        # plt.xlabel('neuron ID')
        # plt.ylabel('|' + pre_str + 'EVec|')
        # plt.title('regular EVs')

        # out.cleanAxis(plt.gca())
        # if save_folder is not None:
        #     plt.savefig(save_folder + pre_str + 'EVecs_mag2.png')
        #     plt.savefig(save_folder + pre_str + 'EVecs_mag2.pdf')


        ### phases ###
        # exceptional
        plt.figure()
        for i in range(idxs.size):
            plt.plot(np.angle(evecs[:,idxs[i]]), '-', 
                     alpha=.2,
                     color=sty.color_cycle[i])

            I = 0
            plt_idxs = []
            mns = []
            for p in range(8):
                mn = np.mean(np.angle(evecs[I:I+self.n_neurons[p],idxs[i]]))

                plt_idxs.append(I)
                plt_idxs.append(I+self.n_neurons[p])
                mns.append(mn)
                mns.append(mn)

                I += self.n_neurons[p]

            plt.plot(plt_idxs, mns, '-', color=sty.color_cycle[i],
                     label=str(idxs[i]) + ' %.2f'%np.abs(evs[idxs[i]]))

        plt.legend(loc='best')        
        plt.xlabel('neuron ID')
        plt.ylabel('<' + pre_str + 'EVec')
        plt.title('exceptional EVs')

        out.cleanAxis(plt.gca())

        if save_folder is not None:
            plt.savefig(save_folder + pre_str + 'EVecs_ang1.png')
            plt.savefig(save_folder + pre_str + 'EVecs_ang1.pdf')

        # regular
        # plt.figure()
        # for i in range(rnd_idxs.size):
        #     plt.plot(np.angle(evecs[:,rnd_idxs[i]]), '-', 
        #              alpha=.1,
        #              color=sty.color_cycle[i])

        #     I = 0
        #     plt_idxs = []
        #     mns = []
        #     for p in range(8):
        #         mn = np.mean(np.angle(evecs[I:I+self.n_neurons[p],rnd_idxs[i]]))

        #         plt_idxs.append(I)
        #         plt_idxs.append(I+self.n_neurons[p])
        #         mns.append(mn)
        #         mns.append(mn)

        #         I += self.n_neurons[p]

        #     plt.plot(plt_idxs, mns, '-', color=sty.color_cycle[i],
        #              label=str(rnd_idxs[i]) + ' %.2f'%np.abs(evs[rnd_idxs[i]]))

        # plt.legend(loc='best')        
        # plt.xlabel('neuron ID')
        # plt.ylabel('<' + pre_str + 'EVec')
        # plt.title('regular EVs')

        # out.cleanAxis(plt.gca())

        # if save_folder is not None:
        #     plt.savefig(save_folder + pre_str + 'EVecs_ang2.png')
        #     plt.savefig(save_folder + pre_str + 'EVecs_ang2.pdf')


    def plotEVdecomp(self, inv_type='exact'):

        save_folder = self.model_def_path+'/analysis/linearization_EVs/'
        try:    os.mkdir(save_folder)
        except: pass

        ### plot rate approximation

        if inv_type == 'exact':
            exten = ''
            title = r'true inverse $(1-W)^{-1}$'

            if self.dlin_rates_th is not None:
                dlin_rates = self.dlin_rates_th
            else:
                dlin_rates = [np.zeros((n,self.angles.size)) for n in self.n_neurons]

        elif inv_type == 'approx':
            exten = '_apprx'
            title = r'approximation $(1 + W)$'

            if self.dlin_rates_apprx is not None:
                dlin_rates = self.dlin_rates_apprx
            else:
                dlin_rates = [np.zeros((n,self.angles.size)) for n in self.n_neurons]

        plt.figure(figsize=(6,4))

        tmp = np.concatenate(dlin_rates)
        tmp = np.mean(tmp, axis=1) # mean over angles
        plt.plot(tmp, label=r'$d\nu$ linearizatioin')
        plt.plot(np.repeat(self.dlin_rates_EV, self.n_neurons), label=r'$d\nu$ EV baseline')

        # plt.title(title)
        plt.legend(loc='best')

        plt.ylabel(r'$\nu$ [Hz]')

        xticks = np.concatenate(([0], np.cumsum(self.n_neurons)[:-1])) + self.n_neurons/2.
        xticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
        plt.xticks(xticks, xticklabels, rotation=60)

        out.cleanAxis(plt.gca())

        plt.savefig(save_folder + 'dv_lin_VS_dv_EV'+exten+'.png')
        plt.savefig(save_folder + 'dv_lin_VS_dv_EV'+exten+'.pdf')





        def extr(*arr):
            arr = np.concatenate(arr)
            return np.array([np.min(arr), np.max(arr)])


        lim_sc = 0.1


        ### plot EV scaling ###


        f, axs = plt.subplots(3,1,figsize=(6,6))
        # plt.suptitle(title)


        ### xi
        xs, ys = np.real(self.xi), np.imag(self.xi)

        axs[0].plot([0., 0.], [1e5, -1e5], '-k', alpha=.5, lw=1., zorder=-1e3)
        axs[0].plot([1e5, -1e5], [0., 0.], '-k', alpha=.5, lw=1., zorder=-1e3)

        axs[0].plot(xs, ys, 'o', c=sty.color_cycle[4], label=r'$\xi$')
        for i in range(8):
            axs[0].text(xs[i], ys[i], str(i), color=sty.color_cycle[4], fontsize=12)
        
        xlims, tmp = extr(xs), lim_sc * np.diff(extr(xs))[0]
        xlims[0] -= tmp
        xlims[1] += tmp
        
        ylims, tmp = extr(ys), lim_sc * np.diff(extr(ys))[0]
        ylims[0] -= tmp
        ylims[1] += tmp

        axs[0].set_xlim(xlims)
        axs[0].set_ylim(ylims)

        axs[0].legend(loc='best')

        ### lambda

        xs, ys = np.real(self.Qevs_tmp), np.imag(self.Qevs_tmp)

        axs[1].plot([0., 0.], [1e5, -1e5], '-k', alpha=.5, lw=1., zorder=-1e3)
        axs[1].plot([1e5, -1e5], [0., 0.], '-k', alpha=.5, lw=1., zorder=-1e3)


        axs[1].plot(xs, ys, 'o', c=sty.color_cycle[1], label=r'$\tilde\lambda$')
        for i in range(8):
            axs[1].text(xs[i], ys[i], str(i), color=sty.color_cycle[1], fontsize=12)
        
        xlims, tmp = extr(xs), lim_sc * np.diff(extr(xs))[0]
        xlims[0] -= tmp
        xlims[1] += tmp
        
        ylims, tmp = extr(ys), lim_sc * np.diff(extr(ys))[0]
        ylims[0] -= tmp
        ylims[1] += tmp

        axs[1].set_xlim(xlims)
        axs[1].set_ylim(ylims)

        axs[1].legend(loc='best')

        ### xi * lambda

        xs, ys = np.real(self.mul),np.imag(self.mul)

        axs[2].plot([0., 0.], [1e5, -1e5], '-k', alpha=.5, lw=1., zorder=-1e3)
        axs[2].plot([1e5, -1e5], [0., 0.], '-k', alpha=.5, lw=1., zorder=-1e3)

        axs[2].plot(xs, ys, 'o', c=sty.color_cycle[2], label=r'$\xi\cdot\tilde\lambda$')
        for i in range(8):
            axs[2].text(xs[i], ys[i], str(i), color=sty.color_cycle[2], fontsize=12)
        
        xlims, tmp = extr(xs), lim_sc * np.diff(extr(xs))[0]
        xlims[0] -= tmp
        xlims[1] += tmp
        
        ylims, tmp = extr(ys), lim_sc * np.diff(extr(ys))[0]
        ylims[0] -= tmp
        ylims[1] += tmp

        axs[2].set_xlim(xlims)
        axs[2].set_ylim(ylims)

        axs[2].legend(loc='best')


        # axs[1].set_ylabel('imag')
        # axs[2].set_xlabel('real')
        axs[2].set_xlabel(r'$\mathfrak{R}(\lambda)$')
        axs[1].set_ylabel(r'$\mathfrak{I}(\lambda)$')
        out.cleanAxis(axs[0])
        out.cleanAxis(axs[1])
        out.cleanAxis(axs[2])

        plt.savefig(save_folder + 'EV_scaling'+exten+'.png')
        plt.savefig(save_folder + 'EV_scaling'+exten+'.pdf')




        h = plt.figure(figsize=(6,4))
        ax = plt.gca()

        p = 0
        for i in range(8):
            plt.plot(np.repeat(np.real(self.rhos[:,i]), self.n_neurons), 
                     label='mode %i'%i, c=sty.color_cycle[i])


        plt.ylabel(r'$\nu$ [Hz]')
        xticks = np.concatenate(([p], np.cumsum(self.n_neurons)[:-1])) + self.n_neurons/2.
        xticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
        plt.xticks(xticks, xticklabels, rotation=60)


        # ax.legend(loc='best')
        out.cleanAxis(ax)


    
        h.tight_layout()

        plt.savefig(save_folder + 'mode_decomposition.png')
        plt.savefig(save_folder + 'mode_decomposition.pdf')