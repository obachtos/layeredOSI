import sys, os, time, socket
import h5py
import numpy as np

import itertools
from multiprocessing import Queue, Process

import readData; reload(readData); from readData import *
from functions import calcOS


data_folder = '../../data/noIStim/'
max_proc = 1


if data_folder[-1] != '/':
    data_folder += '/'

print 'Analyzing data from experiment ' + data_folder + ':\n'
experiments = os.listdir(data_folder)
sys.path.append(os.path.abspath(data_folder))



# for exp_folder in experiments:
def doExperiment(exp_folder, info):
    if max_proc==1: 
        print 'Analyzing experiment %i/%i "'%info + exp_folder + '"'

    experiment_str = exp_folder
    exp_folder = data_folder + exp_folder

    sys.path[-1] = os.path.abspath(exp_folder)

    wd = os.getcwd()    
    os.chdir(exp_folder)
    sys.path.append('.')
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)
    os.chdir(wd)

    if max_proc==1:
        sys.stdout.write('  Reading data...')
        sys.stdout.flush()

    n_neurons, attrs, rec_spike_GIDs, spike_GIDs, spike_times,\
         conn_target_neurons, connectivity, pref_angles, phases = \
                            readData(exp_folder)

    if max_proc==1:
        sys.stdout.write('done\n  Calculating stuff...\n')
        sys.stdout.flush()

    ### creat hdf5 file for calculated quantities
    if os.path.isfile(exp_folder + '/hdf5s/analysis.hdf5'):
        os.system('rm ' + exp_folder + '/hdf5s/analysis.hdf5')
    hdf5_analysis = h5py.File(exp_folder + '/hdf5s/analysis.hdf5', 'w')

    pop_groups = []
    for p, pop in enumerate(net.populations):
        pop_groups.append(hdf5_analysis.create_group(pop))

    if not os.path.isdir(exp_folder + '/analysis'):
        os.makedirs(exp_folder + '/analysis')

    ############################################
    #### calculate individual neuron rates #####
    ############################################
    if max_proc==1:
        sys.stdout.write('    -> neuron rates')
        sys.stdout.flush()
    t0 = time.time()

    rates = [np.zeros((len(rec_spike_GIDs[p]), len(sim.angles)), dtype=float) for p in range(len(net.populations))]
    for p in range(len(net.populations)):   
        for a in range(len(sim.angles)):
            
            count = np.bincount(spike_GIDs[a][p]-rec_spike_GIDs[p][0])
            rates[p][:count.size,a] = count/sim.t_measure*1e3

        pop_groups[p].create_dataset('rates', data=rates[p])

    ts0 = time.time()-t0

    ######################################
    ##### calculate population rates #####
    ######################################
    if max_proc==1:
        sys.stdout.write(' %.1fs\n'%ts0)
        sys.stdout.write('    -> population rates')
        sys.stdout.flush()
    t1 = time.time()

    pop_rates = np.empty((len(net.populations),1))
    pop_rates_angles = np.empty((len(net.populations), len(sim.angles)))
    with open(exp_folder + '/analysis/rates.txt', 'w') as f:

        for p in range(len(net.populations)):   
            pop_rates[p] = np.mean(rates[p].flatten())

            # for a in range(len(sim.angles)):
            pop_rates_angles[p,:] = np.mean(rates[p], axis=0)

            f.write(str(pop_rates[p,0])+'\n')

    hdf5_analysis.create_dataset('pop_rates', data=pop_rates)
    hdf5_analysis.create_dataset('pop_rates_angles', data=pop_rates_angles)

    ts1 = time.time()-t1    

    #####################
    ### calculate CVs ###
    #####################
    if max_proc==1:
        sys.stdout.write(' %.1fs\n'%ts1)
        sys.stdout.write('    -> CVs')
        sys.stdout.flush()
    t2 = time.time()

    for p in range(len(net.populations)):
        CVs = np.empty((len(rec_spike_GIDs[p]), len(sim.angles)), dtype=float) 
        CVs[:,:] = np.nan

        for a in range(len(sim.angles)):

            GIDs = np.unique(spike_GIDs[a][p])

            idxs = np.argsort(spike_GIDs[a][p], kind='mergesort')
            ISIs = np.diff(np.concatenate(([np.nan], spike_times[a][p][idxs])))
            ISIs = np.split(ISIs, np.where(np.diff(spike_GIDs[a][p][idxs])!=0)[0]+1)

            idxs = np.array(map(lambda arr: arr.size > 1, ISIs))
            GIDs = GIDs[idxs]
            ISIs = list(itertools.compress(ISIs, idxs))

            idxs = map(lambda GID: np.where(rec_spike_GIDs[p]==GID)[0][0], GIDs)
            CVs[idxs, a] = map(lambda ISI_arr: np.std(ISI_arr[1:])/np.mean(ISI_arr[1:]), ISIs)

        pop_groups[p].create_dataset('CVs', data=CVs)


        if max_proc==1:
            sys.stdout.write('\r    -> CVs %i%%'%np.round((p+1)/8.*100.))
            sys.stdout.flush()

    ts2 = time.time()-t2


    #########################
    ##### calculate OSV #####
    #########################
    if max_proc==1:
        sys.stdout.write('\r    -> CVs %.1fs    \n'%ts2)
        sys.stdout.write('    -> OSVs')
        sys.stdout.flush()
    t3 = time.time()


    mean_OSIs, POs, OSIs, OSVs = calcOS(rates, attrs['angles'])

    for p in range(len(net.populations)):
        pop_groups[p].create_dataset('POs', data=POs[p])
        pop_groups[p].create_dataset('OSIs', data=OSIs[p])

    hdf5_analysis.create_dataset('mean_OSIs', data=mean_OSIs)

    ts3 = time.time()-t3


    #########################
    ##### calculate TVs #####
    #########################
    if max_proc==1:
        sys.stdout.write('\r    -> OSVs %.1fs    \n'%ts3)
        sys.stdout.write('    -> tuning vectors')
        sys.stdout.flush()
    t3b = time.time()

    if connectivity is not None:

        PSPs = np.ones((8,8)) * attrs['PSP_e']
        PSPs[0,2] *= 2.
        PSPs[:,1::2] *= net.g 

        sgnl_tuning_vectors = []
        for p, pop in enumerate(net.populations):
            sgnl_tuning_vectors.append(net.model_params['C_m'] * np.mean(rates[p], axis=1) * OSVs[p])

            hdf5_analysis.create_dataset('sgnl_tuning_vectors_'+pop, data=sgnl_tuning_vectors[p])

        proj_tuning_vectors = [np.zeros((len(rec_spike_GIDs[p]), len(net.populations)+1), dtype=complex) for p in range(len(net.populations))]
        I = 0
        for p, pop in enumerate(net.populations):
            J = 0

            for p_source, pop_source in enumerate(net.populations):

                conn = connectivity[I:I+len(rec_spike_GIDs[p]),
                                     J:J+len(rec_spike_GIDs[p_source])]
                proj_tuning_vectors[p][:,p_source] = PSPs[p,p_source] * conn.astype(np.bool).dot(sgnl_tuning_vectors[p_source])

                J += len(rec_spike_GIDs[p_source])

            #thalamus
            if pref_angles[p].size != 0:
                proj_tuning_vectors[p][:,-1] = net.model_params['C_m'] * net.PSP_ext \
                                                * attrs['th_rate'] * attrs['K_th'][p]\
                                                * attrs['m'][0]/2. * np.exp(1j*2.*pref_angles[p])

            I += len(rec_spike_GIDs[p])

            hdf5_analysis.create_dataset('proj_tuning_vectors_'+pop, data=proj_tuning_vectors[p])

    ts3b = time.time()-t3b

    ##################################
    ##### calculate expected OSV #####
    ##################################
    if max_proc==1:
        sys.stdout.write(' %.1fs\n'%ts3b)
        sys.stdout.write('    -> expected OSVs')
        sys.stdout.flush()
    t4 = time.time()

    def expectedOSI(count, N=1e3):

        lamb = count/sim.angles.size

        ns = np.random.poisson(lamb, (int(N), sim.angles.size))

        sum_ns = np.sum(ns, axis=1)
        idx_0 = (sum_ns == 0.)
        sum_ns[idx_0] = 1.
        
        OSI_N = np.abs(np.sum(ns * np.exp(2.*np.pi*1j*sim.angles/180.), axis=1)) / sum_ns
        OSI_N[idx_0] = 0

        return np.mean(OSI_N)

    mean_expOSIs = np.empty(len(net.populations))
    for p, pop in enumerate(net.populations):

        tmp = np.sum(rates[p], axis=1) * (sim.t_measure/1e3)
        count = tmp.astype(np.int)

        expOSIs = np.zeros(rates[p].shape[0])
        for i in np.where(count != 0)[0]:
            expOSIs[i] =  expectedOSI(count[i])

        pop_groups[p].create_dataset('expOSIs', data=expOSIs)

        mean_expOSIs[p] = np.mean(expOSIs)

        if max_proc==1:
            sys.stdout.write('\r    -> expected OSVs %i%%'%np.round((p+1)/8.*100.))
            sys.stdout.flush()

    hdf5_analysis.create_dataset('mean_expOSIs', data=mean_expOSIs)
     
    ts4 = time.time()-t4

    ################################
    ##### calculate tuning FFT #####
    ################################
    if max_proc==1:
        sys.stdout.write('\r    -> FFTs %.1fs     \n'%ts4) 
        sys.stdout.write('    -> tuning FFTs')
        sys.stdout.flush()
    t6 = time.time()

    for p, pop in enumerate(net.populations):

        tuning_FFTs = np.fft.rfft(rates[p], axis=1)#, norm='ortho')
        pop_groups[p].create_dataset('tuning_FFTs', data=tuning_FFTs)

    ts6 = time.time()-t6


    if max_proc==1:
        sys.stdout.write('\r    -> tuning FFTs %.1fs     \n'%ts6) 

        sys.stdout.write('  done\n\n')
        sys.stdout.flush()

    hdf5_analysis.close()

    queue.put((experiment_str,))



print 'Analysing %i experiment(s)'%len(experiments)
i, j = 0, 0
queue = Queue()

### start processes
while i < len(experiments):

    if i >= max_proc:
        stuff = queue.get()

        j += 1
        if max_proc > 1:
            print '\tdone with #%i: '%j + stuff[0]


    p = Process(target=doExperiment, args=(experiments[i], (i+1, len(experiments))))
    p.start()

    i += 1

### wait for remaining processe to finish
while j < len(experiments):
    stuff = queue.get()

    j += 1
    if max_proc > 1:
        print '\tdone with #%i: '%j + stuff[0]

    i += 1

