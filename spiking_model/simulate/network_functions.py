import nest
import numpy as np
# import scipy as sp
# import scipy.sparse
import os, time, sys

import scipy.sparse

import network_params as net; reload(net)
import sim_params as sim; reload(sim)
import derive_params as der; reload(der)

import functions; reload(functions); from functions import *

######################################################
# Create nodes
######################################################
def initNEST():

    nest.ResetKernel()
    # set global kernel parameters
    nest.SetKernelStatus(
        {'overwrite_files': True,
         'resolution': sim.dt,
         'total_num_virtual_procs': sim.n_vp})

    # Set random seeds
    nest.SetKernelStatus({'grng_seed': sim.master_seed})
    nest.SetKernelStatus({'rng_seeds': range(sim.master_seed + 1, sim.master_seed + sim.n_vp + 1)})

    seeds = range(sim.master_seed + sim.n_vp + 1, sim.master_seed + 2*sim.n_vp + 1)

    pyrngs = [np.random.RandomState(s) for s in seeds]
                        
    pyrng_gen = np.random.RandomState(sim.master_seed + 2*sim.n_vp + 1)

    return pyrngs, pyrng_gen


######################################################
# Create nodes
######################################################
def create_nodes(pyrngs, pyrng_gen, th_rate, stim_angle, m, Istr):

    record_voltage = sim.record_voltage and \
                            (stim_angle == 0 or not sim.record_voltage_only_0_angle)

    ### Neurons
    neuron_GIDs = nest.Create(net.neuron_model, der.n_total, params=net.model_params)

    #### Initialize membrane potentials locally ####
    ### drawn from normal distribution with mu=Vm0_mean, sigma=Vm0_std
    node_info   = nest.GetStatus(neuron_GIDs)
    local_nodes = [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]
    for gid, vp in local_nodes: 
       nest.SetStatus([gid], {'V_m': pyrngs[vp].normal(net.Vm0_mean, net.Vm0_std)})
    
    ### GIDs for neurons in layers
    neuron_GIDs = np.split(neuron_GIDs, np.cumsum(der.n_neurons))[:-1]
    neuron_GIDs = map(list, neuron_GIDs)

    ### input currents
    for p, pop in enumerate(net.populations):
        if Istr is None: 
            Istr = 0.
        nest.SetStatus(neuron_GIDs[p], {'I_e': Istr * sim.I_input[p]})


    #### External input ####
    ### One poisson generator per population.
    ext_poisson_params = [{'rate': net.bg_rate * in_degree} for in_degree in der.K_bg.copy()]
    ext_poisson = nest.Create('poisson_generator', der.n_populations, params=ext_poisson_params) 

    ### Thalamic neurons: Poisson generators

    th_GIDs, pref_angles= [], []
    for p, pop in enumerate(net.populations):
        
        if pop[-1] == 'e': m_tmp = m[0]
        elif pop[-1] == 'i': m_tmp = m[1]

        if der.K_th_neurons[p]==0 or th_rate==0:

            th_GIDs.append(())
            pref_angles.append(np.array([]))

            continue
        
        pref_angles_p = pyrng_gen.uniform(0, np.pi, der.n_neurons[p])
        rates = der.K_th_neurons[p] * th_rate * (1. + m_tmp*np.cos(2.*(stim_angle-pref_angles_p)))
        
        par_dicts = [{'start': sim.th_start, 
                      'stop': sim.th_start + sim.th_duration,
                      'rate': rate} for rate in rates]

        # create one poisson generator per neuron (no filtering for local, pgs are on every process)
        th_GIDs.append(nest.Create('poisson_generator', der.n_neurons[p], params=par_dicts))

        pref_angles.append(pref_angles_p)
            
        
    # Devices
    if sim.record_cortical_spikes:
        spike_detector_dict = [{'to_file': False, 
                                'to_memory': True} for population in net.populations]
        spike_detectors = nest.Create('spike_detector', der.n_populations, params=spike_detector_dict)
    else: 
        spike_detectors = ()

    if record_voltage:
        multimeter_dict = [{'to_file': False, 
                            'to_memory': True, 
                            'record_from': ['V_m']} for population in net.populations]
        multimeters = nest.Create('multimeter', der.n_populations, params=multimeter_dict)
    else: 
        multimeters = ()

    return neuron_GIDs, ext_poisson, th_GIDs, pref_angles, spike_detectors, multimeters



###################################################
# Connect
###################################################
def connect_network(neuron_GIDs, ext_poisson, th_GIDs):
     
    # pfusch
    K_scaled = der.K_scaled.copy()
    K_scaled[0, 2] = net.f_K_L4e_to_L23e * K_scaled[0, 2]


    if net.neuron_model == 'iaf_psc_delta':
        
        tau_m, C_m = [net.model_params[key] for key in ['tau_m', 'C_m']]
        
        PSP_mat = np.empty((8,8))
        PSP_mat[:, 0:7:2] = net.PSP_e
        PSP_mat[:, 1:8:2] = net.g*net.PSP_e
        PSP_mat[0,2] = net.f_L4e_to_L23e * PSP_mat[0,2]

        weights_ext = net.PSP_ext
        weights_th = net.PSP_th
        weights_mat = PSP_mat

    elif net.neuron_model == 'iaf_psc_exp':
        
        tau_m, tau_syn_ex, tau_syn_in, C_m = [net.model_params[key] for \
                                key in ['tau_m', 'tau_syn_ex', 'tau_syn_in', 'C_m']]

        PSC_e = PSC(net.PSP_e, tau_m, tau_syn_ex, C_m) # excitatory (presynaptic)
        
        PSC_ext = PSC(net.PSP_ext, tau_m, tau_syn_ex, C_m)  # external poisson
        PSC_th = PSC(net.PSP_th, tau_m, tau_syn_ex, C_m)   # thalamus
        
        PSP_i = net.PSP_e * net.g    # IPSP from EPSP
        PSC_i = PSC(PSP_i, tau_m, tau_syn_in, C_m)    # inhibitory (presynaptic)
        
        PSC_L4e_to_L23e = PSC(net.f_L4e_to_L23e*net.PSP_e, tau_m, tau_syn_ex, C_m) # synapses from L4e to L23e
        
        # create PSC array, shape of conn_probs
        PSC_neurons = np.empty(net.conn_probs.shape)
        PSC_neurons[:, 0:7:2] = PSC_e
        PSC_neurons[:, 1:8:2] = PSC_i        
        PSC_neurons[0, 2] =  PSC_L4e_to_L23e

        weights_ext = PSC_ext
        weights_th = PSC_th
        weights_mat = PSC_neurons

    # Connect target populations...
    for target_index, target_pop in enumerate(net.populations):
        target_GIDs = neuron_GIDs[target_index] # transform indices to GIDs of target population

        # ...to source populations
        for source_index, source_pop in enumerate(net.populations):
            source_GIDs = neuron_GIDs[source_index]    # transform indices to GIDs of source population
            n_synapses  = K_scaled[target_index, source_index]  # connection probability
            if not n_synapses == 0:


                if net.conn_rule == 'fixed_total_number':
                    conn_dict = {'rule': net.conn_rule,
                                 'N':    n_synapses}
                elif net.conn_rule == 'fixed_indegree':
                    conn_dict = {'rule': net.conn_rule,
                                 'autapses': net.autapses,
                                 'multapses': net.multapses,
                                 'indegree': int(round(float(n_synapses)/der.n_neurons[target_index]))}
                    
                if source_pop[-1] == 'e':
                    weight_dict = net.weight_dict_exc.copy()
                    mean_delay  = net.delay_e
                elif source_pop[-1] == 'i':
                    weight_dict = net.weight_dict_inh.copy()
                    mean_delay  = net.delay_i
                else:
                    print('No weight dictionary defined for this neuron type!')

                mean_weight = weights_mat[target_index, source_index]
                std_weight = abs(mean_weight * net.PSC_rel_sd)
                weight_dict['mu'] = mean_weight
                weight_dict['sigma'] = std_weight

                std_delay = mean_delay * net.delay_rel_sd 
                delay_dict = net.delay_dict.copy()
                delay_dict['mu'] = mean_delay
                delay_dict['sigma'] = std_delay

                syn_dict = net.syn_dict.copy()
                syn_dict['weight'] = weight_dict
                syn_dict['delay'] = delay_dict
                
                nest.Connect(source_GIDs, target_GIDs, conn_dict, syn_dict)
        
        # ...to background
        nest.Connect([ext_poisson[target_index]], target_GIDs, 
                        conn_spec={'rule': 'all_to_all'}, 
                        syn_spec={'weight': weights_ext}) 
        
        # ...to thalamic population
        if len(th_GIDs[target_index]) != 0:
            conn_dict_th = {'rule': 'one_to_one'}
            syn_dict_th = {'weight': weights_th}
            
            nest.Connect(th_GIDs[target_index], target_GIDs, conn_dict_th, syn_dict_th)


###################################################
# Connect recorders
###################################################
def connect_recorders(spike_detectors, multimeters, neuron_GIDs):        

    # Connect populations...
    rec_spike_GIDs, rec_voltage_GIDs = [], []
    for p, pop in enumerate(net.populations):
        target_GIDs = neuron_GIDs[p] # transform indices to GIDs of target population

        # ...to spike detector
        if sim.record_cortical_spikes:
            rec_spike_GIDs.append(target_GIDs[:der.n_neurons_rec_spike[p]])
            nest.Connect(list(rec_spike_GIDs[-1]), [spike_detectors[p]], 'all_to_all')            
        else:
            rec_spike_GIDs.append(np.array([]))

        # ...to multimeter
        if len(multimeters) != 0 :
            rec_voltage_GIDs.append(target_GIDs[:der.n_neurons_rec_voltage[p]])
            nest.Connect([multimeters[p]], list(rec_voltage_GIDs[-1]), 'all_to_all')
        else:
            rec_voltage_GIDs.append(np.array([]))

    return rec_spike_GIDs, rec_voltage_GIDs


###################################################
# save connectivity
###################################################
def read_connectivity(neuron_GIDs):

    # N = 0
    # for p, pop in enumerate(net.populations):
    #     print pop, ': ', len(neuron_GIDs[p])
    #     N += len(neuron_GIDs[p])
    # print 'total: ', N, '\n'

    conn_target_neurons = []
    Is = []
    Js = []
    weights = []
    for p, pop in enumerate(net.populations):

        conn_target_neurons.append(np.array(neuron_GIDs[p][:der.n_rec_per_pop[p]]))

        # print '\nreading conns to ' + pop
        connections = nest.GetConnections(target=list(conn_target_neurons[p]))
        if len(connections) != 0:
            conns = np.vstack(connections)
        else:
            Is.append(np.array([]))# targets
            Js.append(np.array([])) #sources
            weights.append(np.array([]))


        conns = conns[conns[:,0] <= neuron_GIDs[-1][-1], :] # could also be spike generator etc...
        
        # print conns[:5,:]
        # print conns[-5:,:]

        # print 'total # of connections'
        # print ' ', np.sum(der.K_scaled[p,:]), conns.shape[0]

        # convert GIDs to matrix indecies
        Is.append(np.searchsorted(conn_target_neurons[p], conns[:,1]) + np.sum(der.n_rec_per_pop[:p]))# targets
        Js.append(conns[:, 0] - neuron_GIDs[0][0]) #sources

        weights.append(nest.GetStatus(np.split(conns.flatten(), conns.shape[0]), 'weight'))

    Is = np.concatenate(Is)
    Js = np.concatenate(Js)
    weights = np.concatenate(weights)

    ij = np.concatenate((Is.reshape((-1,1)), Js.reshape((-1,1))), axis=1)
    I = 0
    for p_target, pop_target in enumerate(net.populations):
        J = 0
        for p_source, pop_source in enumerate(net.populations):
            
            ijs_tmp = ij[ij[:,0] >= I,:]
            ijs_tmp = ijs_tmp[ijs_tmp[:,0] < I+len(neuron_GIDs[p_target]),:]
            ijs_tmp = ijs_tmp[ijs_tmp[:,1] >= J,:]
            ijs_tmp = ijs_tmp[ijs_tmp[:,1] < J+len(neuron_GIDs[p_source]),:]

            # print pop_target + ' <- ' + pop_source
            # print ' ', der.K_scaled[p_target, p_source], ijs_tmp.shape[0]

            # if der.K_scaled[p_target, p_source] != ijs_tmp.shape[0]:
            #     print '=========================================================================================='
            #     print '=========================================================================================='
            #     print '=========================================================================================='

            J += len(neuron_GIDs[p_source])
        I += len(neuron_GIDs[p_target])

    connectivity = [Is, Js, weights]

    return conn_target_neurons, connectivity


