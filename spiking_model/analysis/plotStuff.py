
import sys, os, socket
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Queue, Queue, Process

import readData; reload(readData); from readData import *
import plotFunctions; reload(plotFunctions); from plotFunctions import *
import plotOreientationFunctions; reload(plotOreientationFunctions); from plotOreientationFunctions import *

import functions; reload(functions); from functions import *




save_plots = True
show_plots = False

data_folder = '../../data/noIStim'
max_proc = 1






def remondis(strng):
    if max_proc==1:
        sys.stdout.write(strng)
        sys.stdout.flush()

    if show_plots:
        plt.show()
    else:
        plt.close('all')

if data_folder[-1] != '/':
    data_folder += '/'

experiments = os.listdir(data_folder) 

def doExperiment(exp_folder, info):
    if max_proc==1:
        print 'Analyzing experiment %i/%i "'%info + exp_folder + '"'

    experiment_str = exp_folder
    exp_folder = data_folder + exp_folder

    clearPath()
    sys.path.append(os.path.abspath(exp_folder))
    
    if not os.path.isdir(exp_folder + '/analysis'):
        os.makedirs(exp_folder + '/analysis')

    wd = os.getcwd()    
    os.chdir(exp_folder)
    sys.path.append('.')
    import network_params as net; reload(net)
    import sim_params as sim; reload(sim)
    os.chdir(wd)


    if max_proc ==1:
        sys.stdout.write('\n  Reading raw data...')
        sys.stdout.flush()

    n_neurons, attrs, rec_spike_GIDs, spike_GIDs, spike_times, conn_target_neurons, \
                            connectivity, pref_angles, phases = \
                                readData(exp_folder, lowMemory=False)

    if max_proc ==1:
        sys.stdout.write('done\n  Reading analysis data...')
        sys.stdout.flush()

    attrs_ana, pop_rates, pop_rates_angles, evoked_rates, mean_OSIs, rates, CVs, POs, OSIs, \
            expOSIs, mean_expOSIs, FFTs_mean, FFTs_std, FFTs_pop, F0s, F1s, \
            tuning_FFTs, proj_tuning_vectors, sgnl_tuning_vectors\
             = readAnalysisData(exp_folder)

    if max_proc ==1:         
        sys.stdout.write('done\n  Creating plots...')
        sys.stdout.flush()

    if show_plots: plt.ion()
    else: plt.ioff()

    if save_plots:
        save_folder = exp_folder + '/analysis/'
    else:
        save_folder = None


    ### standard analysis
    plotRaster(rec_spike_GIDs, spike_GIDs, spike_times, save_folder=save_folder)
    remondis('1')

    plotRates(pop_rates, save_folder=save_folder)
    remondis('.2')
        
    plotRateHist(rates, pop_rates, save_folder=save_folder)
    remondis('.5')

    cvStatistics(rates, pop_rates, CVs, attrs, save_folder=save_folder)
    remondis('.7')
    
    PSTH(rec_spike_GIDs, spike_times, bin_size=1, save_folder=save_folder)
    remondis('.8')
  
    ### orientation related analysis ###
    plotOSIHist(OSIs, mean_OSIs, save_folder=save_folder)
    remondis('.10')
    
    plotExpOSI(expOSIs, mean_expOSIs, mean_OSIs, save_folder=save_folder)
    remondis('.11')
    
    plotPOHist(POs, save_folder=save_folder)
    remondis('.12')

    plotPOio(POs, pref_angles, save_folder=save_folder)
    remondis('.12b')

    plotAverageTuning(rates, POs, attrs, save_folder=save_folder)
    remondis('.13')

    plotMeanTV(proj_tuning_vectors, attrs, save_folder=save_folder)
    remondis('.13b')
        
    # plt.ioff()
    plotInividualTuning(rates, rec_spike_GIDs, save_folder, N=20)
    remondis('.14')

    ### connectivity and full recording required
    plotInputRates(conn_target_neurons, connectivity, rates, rec_spike_GIDs, 
                   POs, OSIs, pref_angles, attrs, tuning_FFTs,
                   n=10, save_folder=save_folder)#np.inf
    remondis('.15')

    plotTuningFFTs(tuning_FFTs, rec_spike_GIDs, save_folder=save_folder)
    remondis('.16')

    tv_folder = save_folder + 'tuning_vectors/'
    try: os.makedirs(tv_folder)
    except: pass

    if show_plots: plt.ion()
    
    if not show_plots: plt.close('all')

    if max_proc ==1:
        sys.stdout.write('...done\n\n')
        sys.stdout.flush()

    queue.put((experiment_str,))


print 'Plotting data from experiment ' + data_folder + ':'
print '(%i conditions)\n'%len(experiments)



os.system('cp ../simulate/derive_params.py ./derive_params.py')

i, j = 0, 0
queue = Queue()
while i < len(experiments):

    if i >= max_proc:
        stuff = queue.get()

        j += 1
        if max_proc > 1:
            print '\tdone with #%i: '%j + stuff[0]

    p = Process(target=doExperiment, args=(experiments[i], (i+1, len(experiments))))
    p.start()

    i += 1

while j < len(experiments):
    stuff = queue.get()

    j += 1
    if max_proc > 1:
        print '\tdone with #%i: '%j + stuff[0]

    i += 1

os.system('rm ./derive_params.py')
