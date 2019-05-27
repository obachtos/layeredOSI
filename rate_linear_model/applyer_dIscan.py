import sys, time, os, errno
import numpy as np

import analytics as ana; reload(ana)

import output; reload(output); import output as out


t0 = time.time()

max_proc = 12

# del7Scan
scan_title = 'del7Scan'
I0 = np.array([26.45704451,   
               0.10353471,  
               -4.63026653,  
               -3.24050413, 
               -29.48241865,
               -3.93512304, 
               -17.22549621,  
               -3.28171574]) * 1e3 # pA -> nA
max_frac = 3.
n_steps =  13  #total number of array jobs

# # supScan
# scan_title = 'supScan'
# I0 = np.array([73.41525152,  
#                175.87155524,  
#                15.16011982,    
#                7.318257, 
#                96.86754428,
#                137.3768862,   
#                131.19768462,   
#                62.32482988]) * 1e3 # pA -> nA
# max_frac = 2.
# n_steps =  9  #total number of array jobs


K = int(sys.argv[1])
sv_prfx = str(K)


experiment = 'noIStim/a1.0_t100.2_30Hz_mE0.30_mI0.30_K'
c_experiment = 'noIStim/a1.0_t100.2_8Hz_c_mE0.00_mI0.00_K'

data_folder = '../data/'
save_folder = '../data/' + scan_title + '_%i_%i/'%(int(max_frac), n_steps)

try:
    os.makedirs(save_folder)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

FRACs = np.linspace(0., max_frac, n_steps)
frac = FRACs[K]

obj = ana.analytics(data_folder + experiment, data_folder + c_experiment)

obj.I_input_override = frac * I0


### siegert and linear predictions
obj.createSiegTuning('th', max_proc=max_proc, save_loc=save_folder, save_prefix=sv_prfx)


# load unstimulated rate model for linearization point
lin_point = 'sieg_lin0'

obj.loadSiegTuning('lin0', save_prefix='')
obj.dF_load(lin_point, save_prefix='')

# calc. linear prediction
obj.createLinTuning('th', method='bicg', lin_point=lin_point, max_proc=max_proc, 
                    save_loc=save_folder, save_prefix=sv_prfx)


print('completely done (%s)'%out.timeConversion(time.time()-t0))