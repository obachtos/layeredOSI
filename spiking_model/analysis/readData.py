import sys, os
import numpy as np
import h5py

from scipy.sparse import coo_matrix

def readData(exp_folder, lowMemory=False):

    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    hdf5_file = h5py.File(exp_folder + '/hdf5s/data.hdf5', 'r')
        
    n_neurons = hdf5_file.attrs['n_neurons']#np.rint(net.full_scale_n_neurons * net.area).astype(int)
    N = np.sum(n_neurons)

    attrs = {}
    for key in  hdf5_file.attrs.keys():
        attrs[key] = hdf5_file.attrs[key]

    rec_spike_GIDs = []
    for  pop in net.populations:
        rec_spike_GIDs.append(np.array(hdf5_file['rec_spike_GIDs_'+pop]))

    if 'connectivity_Is' in hdf5_file and not lowMemory:
        connectivity = coo_matrix((np.array(hdf5_file['connectivity_weights']), 
                                   (np.array(hdf5_file['connectivity_Is']), 
                                    np.array(hdf5_file['connectivity_Js']))), shape=(N, N)).asformat('csr')

    else:
        connectivity = None

    conn_target_neurons = []
    for p, pop in enumerate(net.populations):
        if 'conn_target_neurons_' + pop in hdf5_file:
            conn_target_neurons.append(np.array(hdf5_file['conn_target_neurons_' + pop]))
        else:
            conn_target_neurons.append(np.array([]))

    spike_GIDs = [[np.array([], dtype=np.uint32) for pop in net.populations] for ang in sim.angles]
    spike_times = [[np.array([], dtype=np.float32) for pop in net.populations] for ang in sim.angles]

    if not lowMemory:
        #for all angels
        for a, stim_angle in enumerate(sim.angles):
            angle_group = hdf5_file['angle_id_%i'%a]
            
            #for all populations
            for p, pop in enumerate(net.populations):
            
                if sim.record_cortical_spikes:
                    # get data from file
                    spike_GIDs[a][p] = np.array(angle_group['spikes/'+pop+'/senders'], dtype=np.uint32)
                    spike_times[a][p] = np.array(angle_group['spikes/'+pop+'/times'], dtype=np.float32)*sim.dt
                    
                    # discard first part of simulation
                    spike_GIDs[a][p] = spike_GIDs[a][p][spike_times[a][p] > sim.t_trans]
                    spike_times[a][p] = spike_times[a][p][spike_times[a][p] > sim.t_trans] -sim.t_trans
                    
                if sim.record_voltage:
                    raise NotImplementedError

    pref_angles = []
    for p, pop in enumerate(net.populations):
        if 'pref_angles_' + pop in hdf5_file:
            pref_angles.append(np.array(hdf5_file['pref_angles_' + pop]))
        else:
            pref_angles.append(np.array([]))

    phases = []
    for p, pop in enumerate(net.populations):
        if 'phases_' + pop in hdf5_file:
            phases.append(np.array(hdf5_file['phases_' + pop]))
        else:
            phases.append(np.array([]))


    return n_neurons, attrs, rec_spike_GIDs, spike_GIDs, spike_times, \
            conn_target_neurons, connectivity, pref_angles, phases


def readAnalysisData(exp_folder):
    
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)

    hdf5_analysis = h5py.File(exp_folder + '/hdf5s/analysis.hdf5', 'r')

    attrs = {}
    for key in  hdf5_analysis.attrs.keys():
        attrs[key] = hdf5_analysis.attrs[key]

    pop_rates = np.array(hdf5_analysis['pop_rates'])
    pop_rates_angles = np.array(hdf5_analysis['pop_rates_angles'])
    if 'evoked_rates' in hdf5_analysis:
        evoked_rates = np.array(hdf5_analysis['evoked_rates'])
    else:
        evoked_rates = None
    mean_OSIs = np.array(hdf5_analysis['mean_OSIs'])
    mean_expOSIs = np.array(hdf5_analysis['mean_expOSIs'])
    
    rates, CVs, POs, OSIs, expOSIs = [], [], [], [], []
    FFTs_mean, FFTs_std, F0s, F1s, FFTs_pop = [], [], [], [], []
    tuning_FFTs, proj_tuning_vectors, sgnl_tuning_vectors = [], [], []
    for pop in net.populations:

        rates.append(np.array(hdf5_analysis[pop]['rates']))
        CVs.append(np.array(hdf5_analysis[pop]['CVs']))
        POs.append(np.array(hdf5_analysis[pop]['POs']))
        OSIs.append(np.array(hdf5_analysis[pop]['OSIs']))
        expOSIs.append(np.array(hdf5_analysis[pop]['expOSIs']))

        if 'proj_tuning_vectors_' + pop in hdf5_analysis:
            proj_tuning_vectors.append(np.array(hdf5_analysis['proj_tuning_vectors_'+pop]))
        else:
            proj_tuning_vectors.append(np.array([]))

        if 'sgnl_tuning_vectors_' + pop in hdf5_analysis:
            sgnl_tuning_vectors.append(np.array(hdf5_analysis['sgnl_tuning_vectors_'+pop]))
        else:
            sgnl_tuning_vectors.append(np.array([]))

        try:
            FFTs_mean.append(np.array(hdf5_analysis[pop]['FFTs_mean']))
            FFTs_std.append(np.array(hdf5_analysis[pop]['FFTs_std']))
            FFTs_pop.append(np.array(hdf5_analysis[pop]['FFTs_pop']))
            F0s.append(np.array(hdf5_analysis[pop]['F0s']))
            F1s.append(np.array(hdf5_analysis[pop]['F1s']))

        except KeyError:
            # print '- no FFTs found in analysis HDF5 -'

            FFTs_mean.append(np.array([]))
            FFTs_std.append(np.array([]))
            FFTs_pop.append(np.array([]))
            F0s.append(np.array([]))
            F1s.append(np.array([]))

        try:
            tuning_FFTs.append(np.array(hdf5_analysis[pop]['tuning_FFTs']))
        except KeyError:
            tuning_FFTs.append(np.array([]))

    return attrs, pop_rates, pop_rates_angles, evoked_rates, mean_OSIs, rates, CVs, \
            POs, OSIs, expOSIs, mean_expOSIs, FFTs_mean, FFTs_std, FFTs_pop, F0s, F1s, \
            tuning_FFTs, proj_tuning_vectors, sgnl_tuning_vectors
