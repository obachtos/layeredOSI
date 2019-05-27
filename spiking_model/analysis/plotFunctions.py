import socket
import numpy as np
import matplotlib.pyplot as plt

import style; reload(style)

import functions; reload(functions); from functions import *

def plotRaster(rec_spike_GIDs, spike_GIDs, spike_times, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    plt.figure()

    dT = 100.

    angle_0 = 0
    angle_90 = int((angle_0 + np.round(len(sim.angles)/2.))%len(sim.angles))

    #for all populations
    times = []
    senders = []
    meanPos = []
    cI = 0
    for p in range(len(net.populations)):   

        minGID = np.min(rec_spike_GIDs[p])
        maxGID = np.max(rec_spike_GIDs[p])

        frac = 1.
        high_bound = int(minGID + frac*(maxGID-minGID))
        
        # filter for GIDs
        spike_times_0_filtered = spike_times[angle_0][p][spike_GIDs[angle_0][p] < high_bound]
        spike_GIDs_0_filtered = spike_GIDs[angle_0][p][spike_GIDs[angle_0][p] < high_bound]
        spike_times_90_filtered = spike_times[angle_90][p][spike_GIDs[angle_90][p] < high_bound]
        spike_GIDs_90_filtered = spike_GIDs[angle_90][p][spike_GIDs[angle_90][p] < high_bound]
        # filter for times
        senders_0 = spike_GIDs_0_filtered[spike_times_0_filtered > sim.t_measure-dT]
        times_0 = spike_times_0_filtered[spike_times_0_filtered > sim.t_measure-dT]
        senders_90 = spike_GIDs_90_filtered[spike_times_90_filtered > sim.t_measure-dT]
        times_90 = spike_times_90_filtered[spike_times_90_filtered > sim.t_measure-dT]
        
        times_0 = times_0 - sim.t_measure + dT
        times_90 = times_90 - sim.t_measure + 2.*dT

        nGID = maxGID - minGID+ 1
        meanPos.append(cI + nGID/2.)
        
        times.append(np.concatenate((times_0, times_90)))
        senders.append(np.concatenate((senders_0, senders_90))-minGID+cI)
        
        cI += nGID
        
    for p, pop in enumerate(net.populations):
        
        plt.plot(times[p], cI-senders[p], '.', ms = 1.5, color=style.pop_colors[p])
        
    yticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
    plt.setp(plt.gca(), yticks=cI-meanPos, yticklabels=yticklabels)

    plt.xlabel('time [ms]')
    xticklabels = ['0','50','100','150','0','50','100','150','']
    plt.setp(plt.gca(), xticklabels=xticklabels)

    plt.plot([dT, dT], plt.gca().get_ylim(), c = 'k', lw = 1, ls='--')

    plt.title('Raster plot of activity')

    if save_folder is not None:
        plt.savefig(save_folder + 'raster.png')


def plotRateHist(rates, pop_rates, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    f, axs = plt.subplots(2,2)#,sharex=True,sharey=True)
    axs = axs.flatten()
    for p, pop in enumerate(net.populations):
        if p%2 == 0:
            ax = axs[p/2]
            ax.set_title(pop[:-1])
        
        ax.hist(rates[p].flatten(), bins = 30, histtype='step', 
                    color=style.pop_colors[p], lw=2, normed=True)

        ax.plot([pop_rates[p], pop_rates[p]], ax.get_ylim(), 
                    color=style.pop_colors[p], lw=2, ls = '--')
        
        
    plt.suptitle('Firing rates')
    ax.set_xlabel('firing rate [Hz]')
    ax.set_ylabel('#neurons')


    if save_folder is not None:
        plt.savefig(save_folder+'rates_hist.png')

def plotRates(pop_rates, save_folder=None, typ = 'arb.'):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    plt.figure()

    ticks= np.arange(len(net.populations))
    for p in range(len(net.populations)):
        plt.bar(ticks[p]+.1, pop_rates[p], width=0.8, color=style.pop_colors[p])

    xticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
    plt.setp(plt.gca(), xticks=ticks+0.5, xticklabels=xticklabels)
    #plt.xlabel('subpopulation')
    plt.xticks(rotation=60)
    plt.ylabel('mean firing rate [Hz]')
    if typ == 'sp':
        plt.title('spontaneuos population firing rates')
        file_name = 'rates.png'
    elif typ == 'ev':
        plt.title('evoked population firing rates')
        file_name = 'evoked_rates.png'
    else:
        plt.title('population firing rates')
        file_name = 'rates.png'


    if save_folder is not None:
        plt.savefig(save_folder+file_name)

def cvStatistics(rates, pop_rates, CVs, attrs, thr=5, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    # # print '6-2'
    # ### plot CV population mean over angles
    # f, axs = plt.subplots(2,2)#,sharex=True,sharey=True)
    # axs = axs.flatten()
    # for p, pop in enumerate(net.populations):
    #     if p%2 == 0:
    #         ax = axs[p/2]
    #         ax.set_title(pop[:-1])

    #     angle_CVs = np.zeros(len(attrs['angles']))
    #     angle_CVs_std = np.zeros(len(attrs['angles']))
    #     for a in range(len(attrs['angles'])):
    #         indis = N_spikes[p][:,a]>thr
    #         if np.sum(indis) == 0:
    #             continue

    #         angle_CVs[a] = np.mean(CVs[p][indis,a])
    #         angle_CVs_std[a] = np.std(CVs[p][indis,a])
        
    #     ticks = np.arange(len(attrs['angles']))
        
    #     if p%2==0:
    #         offset=.1
    #     else:
    #         offset=.5
    #     ax.bar(ticks+offset, angle_CVs, width=0.4, yerr=angle_CVs_std, color=style.pop_colors[p])

    #     if p%2 == 0:

    #         xticklabels = np.arange(0,180,15)

    #         ax.set_xticks(ticks+0.5)
    #         ax.set_xticklabels(xticklabels, rotation=60)

    #         # plt.setp(ax, xticks=ticks+0.5, xticklabels=xticklabels)
    #         # #plt.xlabel('subpopulation')
    #         # ax.xticks(rotation=60)
        
    # plt.suptitle('CVs')
    # axs[2].set_xlabel(r'angle [$^\circ$]')
    # axs[2].set_ylabel('CV')
    # f.tight_layout()

    # if save_folder is not None:
    #     plt.savefig(save_folder+'CVs_over_angle.png')


    ## plot CV mean over populations and angles
    plt.figure()

    ticks = np.arange(len(net.populations))
    for p in range(len(net.populations)):

        N_spikes = rates[p]*sim.t_measure*1e-3
        if np.sum(N_spikes>thr) > 0:
            mean_CV = np.mean(CVs[p][N_spikes>thr].flatten())
        else:
            mean_CV = 0

        plt.bar(ticks[p]+.1, mean_CV, width=0.8, color=style.pop_colors[p])

    xticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
    plt.setp(plt.gca(), xticks=ticks+0.5, xticklabels=xticklabels)
    #plt.xlabel('subpopulation')
    plt.xticks(rotation=60)
    plt.ylabel('CV')
    plt.title('mean CV over all angles')


    if save_folder is not None:
        plt.savefig(save_folder+'CVs.png')


def PSTH(rec_spike_GIDs, spike_times, bin_size=5, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    f, axs = plt.subplots(2,2,sharex=True)
    # f, axs = plt.subplots(4,2,sharex=True,figsize=(8,10))#,sharey=True)
    axs = axs.flatten()

    dT = 50.

    angle_0 = 0
    angle_90 = int((angle_0 + np.round(len(sim.angles)/2.))%len(sim.angles))


    #for all populations
    for p, pop in enumerate(net.populations):

        # filter for times
        times_0 = spike_times[angle_0][p][spike_times[angle_0][p] > sim.t_measure-dT]
        times_90 = spike_times[angle_90][p][spike_times[angle_90][p] > sim.t_measure-dT]
        
        times_0 = times_0 - sim.t_measure + dT
        times_90 = times_90 - sim.t_measure + 2.*dT
        
        times = np.concatenate((times_0, times_90))


        if p%2 == 0:
            ax = axs[p/2]
            ax.set_title(pop[:-1])
        # ax = axs[p]
        # ax.set_title(pop)

        nbins = int(np.round(2*dT/bin_size))

        if times.shape[0] != 0:
            bins, foo, foo = ax.hist(times, weights=np.ones(times.shape)/rec_spike_GIDs[p].size, 
                        range=(0, 2*dT), bins=nbins , histtype='step', 
                        color=style.pop_colors[p], lw=1.5)


        if p%2 != 0:
            ax.set_xlim([0, 2*dT])

            xticklabels = ['0','50','100','150','0','50','100','150','']
            ax.set_xticklabels(xticklabels)

            tmp = ax.get_ylim()
            ax.plot([dT, dT], tmp, c = 'k', lw = 1, ls='--')
            ax.set_ylim(tmp)
        
        
    plt.suptitle('PSTH')
    axs[2].set_xlabel('time [ms]')
    axs[2].set_ylabel('% of neurons')


    if save_folder is not None:
        plt.savefig(save_folder+'PSTH.png')
