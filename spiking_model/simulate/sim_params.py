import numpy as np
import socket

###################################################
###         Simulation parameters        ###        
###################################################

simulation_name = 'noIStim'


#### timing ####
t_trans = .2 * 1e3        # ms; transitional period in order to reach equilibrium
t_measure = 100. * 1e3# 0.#       # ms; time actually measured
t_sim = t_measure + t_trans    # ms; simulated time 
dt = 0.1            # ms; simulation step; default is 0.1 ms. (resolution of kernel)
t_block = 10. * 1e3       # ms; simulation block length

th_start    = 0.    # onset of thalamic input (ms)
th_duration = t_sim    # duration of thalamic input (ms)


#### orientation related ####
angles = np.array(range(0, 180, 15), dtype=int)

mE = .3
mI = .3
m = (mE, mI)


I_input = np.zeros(8)
I_strengths = [None]

# del7Scan
# I_input = np.array([26.45704451,   
#                     0.10353471,  
#                     -4.63026653,  
#                     -3.24050413, 
#                     -29.48241865,
#                     -3.93512304, 
#                     -17.22549621,  
#                     -3.28171574])
# I_strengths = np.linspace(0., 3., 13)

# supScan
# I_input = np.array([73.41525152,  
#                     175.87155524,  
#                     15.16011982,    
#                     7.318257, 
#                     96.86754428,
#                     137.3768862,   
#                     131.19768462,   
#                     62.32482988])
# I_strengths = np.linspace(0., 2., 9)

# read connectivity matrix from network
save_connectivity = True
n_targets = np.inf#100# # np.inf for all

# Cortical spikes
record_cortical_spikes = True 
frac_rec_spike = 1.

# Cortical voltages
record_voltage = False
record_voltage_only_0_angle = False
frac_rec_voltage = 0.01




# master seed for random number generators
# actual seeds will be master_seed ... master_seed + 2*n_vp
#  ==>> different master seeds must be spaced by at least 2*n_vp + 1
master_seed = 123456    # changes rng_seeds and grng_seed

n_vp = 12


compress_records = 'gzip'# None/'gzip'


# path under which data folder is ceated
data_dir = '../../data/' + simulation_name

if data_dir[-1] != '/':
    data_dir += '/'

# extension for each spec folder
path_extension = ''


