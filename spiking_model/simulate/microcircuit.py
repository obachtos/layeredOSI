import nest
import numpy as np
import os, time, errno, sys
import h5py

import network_params as net; reload(net)
import sim_params as sim; reload(sim)
import derive_params as der; reload(der)
import functions; reload(functions)
import network_functions; reload(network_functions); from network_functions import *

t0 = time.time()

# create folder for data if not existing
def try_folder_creation(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
try_folder_creation(sim.data_dir)


# process arguments
if len(sys.argv) <= 1:
    cond = 'stim'
    Iidx = 0
elif len(sys.argv) <= 2:
    cond = sys.argv[1]
    Iidx = 0
else:
    cond = sys.argv[1]
    Iidx = int(sys.argv[1])

Istr = sim.I_strengths[Iidx]


# set pars according to condition
if cond == 'stim':
    th_rate = net.th_rate
    m = sim.m
    save_connectivity = sim.save_connectivity
    is_control = False
elif cond == 'spon':
    th_rate = net.bg_rate
    m = (.0, .0)
    save_connectivity = False
    is_control = True


print '----------------------------'
print '----------- ' + cond + ' -----------'
print '----------------------------'


for j, stim_angle in enumerate(sim.angles):
                            
    print('\n========== Simulating angle ' + str(stim_angle) + ' (#' + str(j+1) + ' of ' + \
            str(len(sim.angles)) + ') ==========')
    print(sim.data_dir)
    print('\na=%.1f, t_sim=%.1f, rule='%\
            (net.area, sim.t_sim,) + net.conn_rule + ',')
    print('v_th=%iHz, mE=%.2f, mI=%.2f'%(th_rate, m[0], m[1]))
    
    # convert degrees to radians
    stim_angle = stim_angle/180.*np.pi

    specs_dir = functions.get_output_path(th_rate, is_control, m, Istr)
    
    try_folder_creation(specs_dir + '/hdf5s')

    # save parameters
    os.system('cp network_params.py ' + specs_dir + '/network_params.py')
    os.system('cp sim_params.py ' + specs_dir + '/sim_params.py')

    ###################################
    ######### initialize NEST #########
    ###################################
    pyrngs, pyrng_gen = initNEST()

    ################################
    ######### Create nodes #########
    ################################
    t_create_0 = time.time()   
    print('\nCreate nodes')
    sys.stdout.flush()

    neuron_GIDs, ext_poisson, th_GIDs, pref_angles, spike_detectors, multimeters, \
                 = create_nodes(pyrngs, pyrng_gen, th_rate, stim_angle, m, Istr)

    t_create = time.time()-t_create_0
    print(' creation took %.2fs'%t_create)
    sys.stdout.flush()

    ###########################
    ######### Connect #########
    ###########################

    t_connect_0 = time.time() 
    print('\nConnecting')
    sys.stdout.flush()

    connect_network(neuron_GIDs, ext_poisson, th_GIDs)
    rec_spike_GIDs, rec_voltage_GIDs = connect_recorders(spike_detectors, 
                                                         multimeters, neuron_GIDs)

    t_connect = time.time()-t_connect_0
    print(' connecting took %.2fs'%t_connect)
    sys.stdout.flush()

    #####################################
    ######### Read Connectivity #########
    #####################################

    if save_connectivity and j==0:

        t_connect_0 = time.time()    
        print('\nReading connectivity')
        sys.stdout.flush()
            
        conn_target_neurons, connectivity = read_connectivity(neuron_GIDs)

        t_read_connect = time.time() - t_connect_0
        print(' reading took %.2fs'%t_read_connect)
        sys.stdout.flush()

    ################################
    ######### Saving Infos #########
    ################################

    # Set data path and create hdf5 file
    if j==0: 
        mode = 'w'
    else:
        mode = 'r+'

    hdf5_file = h5py.File(specs_dir + '/hdf5s/data.hdf5', mode)

    if j==0: # not for each angle

        hdf5_file.attrs['area'] = net.area
        hdf5_file.attrs['sim_time'] = sim.t_sim
        hdf5_file.attrs['PSP_e'] = net.PSP_e
        hdf5_file.attrs['g'] = net.g
        hdf5_file.attrs['n_neurons'] = der.n_neurons
        hdf5_file.attrs['bg_rate'] = net.bg_rate
        hdf5_file.attrs['th_rate'] = th_rate
        hdf5_file.attrs['K_per_neuron'] = der.K_scaled/der.n_neurons.reshape((8,1))
        hdf5_file.attrs['K_bg'] = der.K_bg
        hdf5_file.attrs['K_th'] = der.K_th_neurons
        hdf5_file.attrs['m'] = m
        hdf5_file.attrs['conn_rule'] = net.conn_rule
        hdf5_file.attrs['angles'] = sim.angles
        hdf5_file.attrs['contr'] = is_control # is control? (bool)
        hdf5_file.attrs['compression'] = sim.compress_records

        for p, pop in enumerate(net.populations):
            hdf5_file.create_dataset('GIDs_' + pop, data=neuron_GIDs[p])

        if sim.record_cortical_spikes:
            for p, pop in enumerate(net.populations):
                hdf5_file.create_dataset('rec_spike_GIDs_' + pop, data=rec_spike_GIDs[p])

        if len(multimeters) != 0 :
            for p, pop in enumerate(net.populations):
                hdf5_file.create_dataset('rec_voltage_GIDs_' + pop, data=rec_voltage_GIDs[p])
                
        for p, pop in enumerate(net.populations):
            if len(pref_angles[p]) != 0:
                hdf5_file.create_dataset('pref_angles_'+pop, data=pref_angles[p])

    # connectivity (only for first angle)
    if j==0 and save_connectivity:

            hdf5_file.create_dataset('connectivity_Is', data=connectivity[0], 
                            compression=sim.compress_records)
            hdf5_file.create_dataset('connectivity_Js', data=connectivity[1], 
                            compression=sim.compress_records)
            hdf5_file.create_dataset('connectivity_weights', data=connectivity[2], 
                            compression=sim.compress_records)

            for p, pop in enumerate(net.populations):
                hdf5_file.create_dataset('conn_target_neurons_' + pop, 
                                         data=conn_target_neurons[p])

            connectivity = None
            conn_target_neurons = None

            hdf5_file.attrs['t_read_connect'] = t_read_connect
    
    angle_group = hdf5_file.create_group('angle_id_%i'%j)
    angle_group.attrs['angle'] = stim_angle    

    hdf5_file.close() # reopen after every simulation block     

    ############################
    ######### Simulate #########
    ############################

    ### set new seed for each angle. otherwise identical spike trains for m=0 (incl controls)
    angle_seed = sim.master_seed+569*j
    nest.SetKernelStatus({'grng_seed': angle_seed})
    nest.SetKernelStatus({'rng_seeds': range(angle_seed + 1, angle_seed + sim.n_vp + 1)})

    t_simulate_0 = time.time()
    print('\nSimulating (angle #' + str(j+1) + ' of ' + str(len(sim.angles)) + ')\n')
    sys.stdout.flush()

    nest.Simulate(sim.t_sim)

    ######### Save spike and voltage info #########
    hdf5_file = h5py.File(specs_dir + '/hdf5s/data.hdf5','r+')
    angle_group = hdf5_file['angle_id_%i'%j]
    for p, pop in enumerate(net.populations):
        
        if sim.record_cortical_spikes:
            data = nest.GetStatus([spike_detectors[p]], 'events')[0]

            senders = data["senders"]
            times = data["times"]

            senders = senders.astype(np.uint32)
            times = (times/sim.dt).astype(np.uint32)

            angle_group.create_dataset('spikes/'+pop+'/senders', data=senders, 
                            maxshape=(None,), compression=sim.compress_records)
            angle_group.create_dataset('spikes/'+pop+'/times', data=times, 
                            maxshape=(None,), compression=sim.compress_records)

        if len(multimeters) != 0:
            data = nest.GetStatus([multimeters[p]], 'events')[0]

            senders = data["senders"]
            voltages = data["V_m"]
            times = data["times"]

            senders = senders.astype(np.uint32)
            times = (times/sim.dt).astype(np.uint32)
            voltages = voltages.astype(np.float16)

            angle_group.create_dataset('voltages/'+pop+'/senders', data=senders, 
                                maxshape=(None,), compression=sim.compress_records)
            angle_group.create_dataset('voltages/'+pop+'/times', data=times, 
                                maxshape=(None,), compression=sim.compress_records)
            angle_group.create_dataset('voltages/'+pop+'/measurements', data=voltages, 
                                maxshape=(None,), compression=sim.compress_records)

    hdf5_file.close() 


    t_simulate = time.time() - t_simulate_0
    print('\nSimulation of network with area = %.2f mm for %.1f ms took %.2f s'%\
                        (net.area, sim.t_sim, t_simulate))
    sys.stdout.flush()

    hdf5_file = h5py.File(specs_dir + '/hdf5s/data.hdf5','r+')
    angle_group = hdf5_file['angle_id_%i'%j]

    angle_group.attrs['t_create'] = t_create
    angle_group.attrs['t_connect'] = t_connect
    angle_group.attrs['t_simulate'] = t_simulate   

    hdf5_file.close()

 
total_time = time.time()-t0
print('\nTotal simulation time: ' + timeConversion(total_time))

hdf5_file = h5py.File(specs_dir + '/hdf5s/data.hdf5','r+')

hdf5_file.attrs['total_time'] = total_time

hdf5_file.close()

