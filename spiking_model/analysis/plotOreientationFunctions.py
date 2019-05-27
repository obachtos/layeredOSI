import os, socket
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.mstats import zscore

import style; reload(style)


def plotOSIHist(OSIs, mean_OSIs, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    f, axs = plt.subplots(2,2)#,sharex=True,sharey=True)
    axs = axs.flatten()
    for p, pop in enumerate(net.populations):
        if p%2 == 0:
            ax = axs[p/2]
            ax.set_title(pop[:-1])


        ax.hist(OSIs[p], range=(0,1), bins = 50, histtype='step', color=style.pop_colors[p], lw=2)

        ylims = ax.get_ylim()
        ax.plot(mean_OSIs[p]*np.ones(2), ylims, c=style.pop_colors[p], ls='--', lw=2)
        ax.set_ylim(ylims)

        ax.set_xlim([0,1.])
        
        style.hideAxes(ax)

    plt.suptitle('OSIs')
    axs[2].set_xlabel('OSI')
    axs[2].set_ylabel('#neurons')

    if save_folder is not None:
        plt.savefig(save_folder+'OSI_hist.png')
        
            
    fig = plt.figure()
    plt.boxplot(OSIs)
    xticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
    plt.setp(plt.gca(), xticklabels=xticklabels)

    plt.xlabel('subpopulation')
    plt.ylabel('OSI')
    plt.title('OSI distribution')


    if save_folder is not None:
        plt.savefig(save_folder+'OSI_box.png')


def plotExpOSI(expOSIs, mean_expOSIs, mean_OSIs, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)


    f, axs = plt.subplots(2,2, sharex=True)#,sharey=True)
    axs = axs.flatten()

    for p, pop in enumerate(net.populations):


        if p%2 == 0:
            ax = axs[p/2]
            ax.set_title(pop[:-1])
        
        ax.hist(expOSIs[p], bins = 30, histtype='step', 
                    color=style.pop_colors[p], lw=2, normed=False)

        ax.plot([mean_expOSIs[p], mean_expOSIs[p]], ax.get_ylim(), 
                    color=style.pop_colors[p], lw=2, ls = '--')

        style.hideAxes(ax)

    plt.suptitle('expected OSIs')
    ax.set_xlabel('OSI')
    ax.set_ylabel('#neurons')


    if save_folder is not None:
        plt.savefig(save_folder+'expOSI_hist.png')


    plt.figure()
    ticks = np.arange(len(net.populations))
    plt.bar(ticks+.05, mean_OSIs, width=0.3, color='g', label = 'mean OSI', alpha = .7)
    plt.bar(ticks+.35, mean_expOSIs, width=0.3, color='c', label = 'expected OSI', alpha = .7)
    plt.bar(ticks+.65, mean_OSIs-mean_expOSIs, width=0.3, color='y', 
                label = 'mean - expected', alpha = .7)

    xticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
    plt.setp(plt.gca(), xticks=ticks+0.5, xticklabels=xticklabels)
    plt.xlabel('population')
    plt.ylabel('OSI')
    plt.title('expected OSI')
    plt.legend()

    if save_folder is not None:
        plt.savefig(save_folder+'expectOSI.png')
        

def plotPOHist(POs, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    f, axs = plt.subplots(2,2)#,sharex=True,sharey=True)
    axs = axs.flatten()
    for p, pop in enumerate(net.populations):
        if p%2 == 0:
            ax = axs[p/2]
            ax.set_title(pop[:-1])
        
        ax.hist(POs[p].flatten(), range=(0,180), bins = 50, histtype='step', color=style.pop_colors[p], lw=2)

        ax.set_xlim([0,180.])
        
        style.hideAxes(ax)

    plt.suptitle('POs')

    axs[2].set_xlabel('PO')
    axs[2].set_ylabel('#neurons')

    if save_folder is not None:
        plt.savefig(save_folder+'PO_hist.png')

def plotPOio(POs, pref_angles, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    f, axs = plt.subplots(2,2)#,sharex=True,sharey=True)
    axs = axs.flatten()

    for i, p in enumerate([2,6]):
        ax = axs[i]

        ax.plot(pref_angles[p]/np.pi * 180., POs[p], '.', c=style.pop_colors[p])

        style.hideAxes(ax)

    plt.suptitle('POs')

    axs[0].set_xlabel('input PO')
    axs[0].set_ylabel('ouput PO')

    if save_folder is not None:
        plt.savefig(save_folder+'POio.png')


def plotMeanTV(proj_tuning_vectors, attrs, save_folder=save_folder):

    fig = plt.figure()

    eff_connectivity = np.zeros((8,10))
    for p_target in range(8):
        for p in range(9):
            eff_connectivity[p_target, p] = np.mean(np.abs(proj_tuning_vectors[p_target][:,p]))

    # nm = LogNorm(vmin=1e-3, vmax=eff_connectivity.max())
    # nm = PowerNorm(.5, vmin=0, vmax=eff_connectivity.max())
    nm = None

    im = ax.imshow(eff_connectivity, cmap='viridis', interpolation='none', norm=nm)

    ax = plt.gca()
    ax.xaxis.set_label_position('top') 
    ax.xaxis.tick_top()
    ax.yaxis.set_label_position('left') 
    ax.yaxis.tick_left()

    plt.xlabel('presynaptic')
    plt.ylabel('postsynaptic')
    yticklabels = ['L2/3e','L2/3i','L4e','L4i','L5e','L5i','L6e','L6i']
    xticklabels = yticklabels + ['bg', 'th']
    plt.xticks(np.arange(10), xticklabels, rotation=60)
    plt.setp(plt.gca(), yticks=np.arange(8), yticklabels=yticklabels)

    cb_ax = sty.get_cb_axis(ax)
    cb = plt.colorbar(im, cax=cb_ax, orientation='horizontal')#, ticks=cb_ticks)
    cb.solids.set_edgecolor("face")    
    cb.set_label('input current [pA]')

    if save_folder is not None:
        plt.savefig(save_folder+'meanTVs.png')


def plotInividualTuning(rates, rec_spike_GIDs, save_folder=None, N=30):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    for p, pop in enumerate(net.populations):

        folder = 'tuning_curves/'+ pop + '_individual'
        if not os.path.isdir(save_folder + folder):
            os.makedirs(save_folder + folder)
                
        for i in range(min([len(rec_spike_GIDs[p]), N])):   
        #connection_GIDs = np.load(save_folder + '/analysis/connection_GIDs.npy')       
        #for GID in connection_GIDs[p,:]:
        #   i = np.where(rec_spike_GIDs[p] == GID)[0][0]
                
                
            fig = plt.figure()
            plt.plot(sim.angles, rates[p][i, :], 'o-', lw=1.5)
            plt.savefig(save_folder + folder + '/tc_%i.png'%i)

            plt.close(fig)


def plotAverageTuning(rates, POs, attrs, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    folder = 'tuning_curves/'
    folder_norm = 'tuning_curves_norm/'
    if not os.path.isdir(save_folder + folder):
        os.makedirs(save_folder + folder)
    if not os.path.isdir(save_folder + folder_norm):
        os.makedirs(save_folder + folder_norm)


        add_CI_plot = False

    # standard version
    for p, pop in enumerate(net.populations):
        #rates: n x (#angles)
        #POs: n

        maxima = np.max(rates[p], axis=1).reshape((-1,1))
        maxima[maxima==0] = 1. #value doesnt matter, hole row is 0 anyway
        norm_rates = rates[p]/maxima

        ang, pref = np.meshgrid(attrs['angles'], POs[p])
        new_angles = (ang + (90. - pref))%180.
        
        plt.figure()    
        plt.scatter(new_angles, rates[p], c=style.pop_colors[p])
        plt.xlabel(r'stim. angle centered arround PO [$^\circ$]')
        plt.ylabel('firing rate [Hz]')
        plt.xlim(0,180)
        plt.ylim(0, plt.gca().get_ylim()[1])
        plt.title(pop) 
        
        if save_folder is not None:
            plt.savefig(save_folder + folder + pop + '_scatter.png')
            plt.close()

        nbins = 60
        lims = np.linspace(0,180, nbins+1)
        
        means, norm_means = np.zeros(nbins), np.zeros(nbins) 
        stds, norm_stds = np.zeros(nbins), np.zeros(nbins)
        for i in range(nbins):
            idx = np.logical_and(new_angles >= lims[i], new_angles < lims[i+1])
            if np.sum(idx) > 0:
                means[i] = np.mean(rates[p][idx])
                stds[i] = np.std(rates[p][idx])
                norm_means[i] = np.mean(norm_rates[idx])
                norm_stds[i] = np.std(norm_rates[idx])

        plt.figure()
        plt.plot((lims[:-1]+lims[1:])/2., means, color=style.pop_colors[p], lw=2)
        plt.plot((lims[:-1]+lims[1:])/2., means+stds, color=style.pop_colors[p], lw=2, alpha=.6)
        plt.plot((lims[:-1]+lims[1:])/2., means-stds, color=style.pop_colors[p], lw=2, alpha=.6)
        
        plt.xlabel(r'stim. angle centered arround PO [$^\circ$]')
        plt.ylabel('firing rate [Hz]')
        plt.xlim(0,180)
        plt.ylim(0,plt.gca().get_ylim()[1])
        plt.title(pop)
            
        if save_folder is not None:
            plt.savefig(save_folder + folder + pop + '_mean.png')
            plt.close()

        


def plotInputRates(conn_target_neurons, connectivity, rates, rec_spike_GIDs, 
                   POs, OSIs, pref_angles, attrs, tuning_FFTs, n=np.inf, save_folder=None):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    # os.system('cp ../simulation/derive_params.py ./derive_params.py')
    import derive_params as der; reload(der)
    # os.system('rm ./derive_params.py')

    # sanity_check
    for p, pop in enumerate(net.populations):
        if not np.all(conn_target_neurons[p] == rec_spike_GIDs[p][:conn_target_neurons[p].size]):
            raise UserWarning('sanity check failed in plotInputRates for '+pop)

    I = 0
    # target populations
    for p, pop in enumerate(net.populations):

        if save_folder is not None:
            folder = save_folder + 'input_rates/'+pop
            if not os.path.isdir(folder):
                os.makedirs(folder)


        # target neurons
        GIDs = conn_target_neurons[p][:np.min([n, conn_target_neurons[p].size])]
        for i, gid in enumerate(GIDs):

            inh_input = np.zeros(sim.angles.shape)
            exc_input = np.zeros(sim.angles.shape)

            # source populations
            N = 0
            for p2, pop2 in enumerate(net.populations):
            
                idx = np.arange(N,N+rec_spike_GIDs[p2].size)
                input_rate = np.sum(rates[p2] * np.array(connectivity[I+i,idx].todense()).reshape((-1,1)).astype(bool), 
                                    axis=0)

                if p==0 and p2==2:
                    exc_input += 2*input_rate
                elif p2 in [0, 2, 4, 6]:
                    exc_input += input_rate
                elif p2 in [1, 3, 5, 7]:
                    inh_input += input_rate

                N += rec_spike_GIDs[p2].size
            

            ext_input = der.K_bg[p] * net.bg_rate * np.ones(sim.angles.size)

            if der.K_th_neurons[p] != 0:
                if p%2 ==0:
                    m = attrs['m'][0]
                else:
                    m = attrs['m'][1]

                rate_th = der.K_th_neurons[p] * attrs['th_rate'] * \
                    (1. + m * np.cos(2.*(sim.angles/180.*np.pi-pref_angles[p][i])))

                ext_input += rate_th


            all_input = exc_input + net.g * inh_input + ext_input

            fig = plt.figure(figsize=(6,5))
            ax1 = plt.gca()

            ext_angles = np.concatenate((sim.angles, [180.]))
            ax1.plot(ext_angles, np.concatenate((exc_input, [exc_input[0]]))/1000., 
                     '-', c='steelblue', lw=2, label='exc. input')
            ax1.plot(ext_angles, np.concatenate((inh_input, [inh_input[0]]))/1000., 
                     '-', c='indianred', lw=2, label='inh. input')

            ax1.set_xlabel('stim. angle [$^\circ$] ')
            ax1.set_ylabel(r'$\nu_{in}$ [kHz]')

            plt.setp(ax1, xticks=np.arange(0,182,30))

            # plt.legend(loc='lower left')

            ax2 = ax1.twinx()

            idx = np.where(rec_spike_GIDs[p] == gid)[0][0]
            ax2.plot(ext_angles, np.concatenate((rates[p][idx,:], [rates[p][idx,0]])), 
                    label='output rate', color='olive', lw=2, ls = '-')
            ax2.plot([POs[p][idx], POs[p][idx]], ax2.get_ylim(), 
                    label='OSI=%.2f'%OSIs[p][idx], color='olive', lw=2, ls = '--')

            tmp = tuning_FFTs[p][i,:].copy()
            tmp[2:] = 0
            reduced = np.fft.irfft(tmp)
            ax2.plot(ext_angles, np.concatenate((reduced, [reduced[0]])), 
                    label='F0+F1', color='olive', lw=1, ls = '-')


            ax2.set_ylabel(r'$\nu_{out}$ [Hz]', color='olive')
            ax2.tick_params(axis='y', colors='olive')
            ax2.yaxis.label.set_color('olive')

            ax1.spines['top'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.xaxis.set_ticks_position('bottom')
            ax2.xaxis.set_ticks_position('bottom')

            plt.title('OSI: %.2f'%OSIs[p][idx])

            fig.tight_layout()


            if save_folder is not None:
                # plt.savefig(folder + '/%i_%.2f.pdf'%(gid,OSIs[p][idx]))
                plt.savefig(folder + '/%i_%.2f.png'%(gid,OSIs[p][idx]))

            plt.close(fig)

        I += len(conn_target_neurons[p])


def plotTuningFFTs(tuning_FFTs, rec_spike_GIDs, save_folder=None):

    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)


    ### F1 stength
    f, axs = plt.subplots(2,2)#,sharex=True)#,sharey=True)
    axs = axs.flatten()
    for p, pop in enumerate(net.populations):
        if p%2 == 0:
            ax = axs[p/2]
            ax.set_title(pop[:-1])
            
        ax.hist(np.abs(tuning_FFTs[p][:,1]), bins = 50, histtype='step', 
                    color=style.pop_colors[p], lw=2, normed=True)
        
        style.hideAxes(ax)

    plt.suptitle('F1 of tuning curves')
    axs[2].set_xlabel('F1')
    axs[2].set_ylabel('#neurons')

    if save_folder is not None:
        plt.savefig(save_folder+'F1_hist.png')

