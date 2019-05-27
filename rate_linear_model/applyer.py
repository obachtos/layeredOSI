import socket
import scipy
import numpy as np

from multiprocessing import Queue, Process

import matplotlib.pyplot as plt

import output; reload(output); import output as out
import functions; reload(functions); from functions import Q_mul, S_mul

import analytics as ana; reload(ana)

plt.close('all')



experiment = 'noStim/a1.0_t100.2_30Hz_mE0.30_mI0.30_K'
c_experiment = 'noStim/a1.0_t100.2_8Hz_c_mE0.00_mI0.00_K'

data_folder = '../data/'

max_proc_sieg = 12
max_proc_lin = 6

obj = ana.analytics(data_folder + experiment, data_folder + c_experiment)


### siegert and linear predictions


# 'spk_sp'#'spk_th'#'sieg_sp'#'sieg_th'#
sv_prfx = '' #'_Idel7'#

obj.createSiegTuning('sp', max_proc=max_proc, save_prefix=sv_prfx)
obj.createSiegTuning('th', max_proc=max_proc, save_prefix=sv_prfx)
obj.createSiegTuning('lin0', max_proc=max_proc, save_prefix=sv_prfx)

# obj.loadSiegTuning('sp', save_prefix=sv_prfx)
# obj.loadSiegTuning('th', save_prefix=sv_prfx)
# obj.loadSiegTuning('lin0', save_prefix=sv_prfx)

lin_point = 'sieg_lin0' # 'sieg_sp' #
obj.dF(lin_point, save_prefix=sv_prfx)
# obj.dF_load(lin_point, save_prefix=sv_prfx)

#'inv'  'spsolve'  'bicg'
obj.createLinTuning('sp', method='bicg', lin_point=lin_point, max_proc=max_proc_lin, save_prefix=sv_prfx)
obj.createLinTuning('th', method='bicg', lin_point=lin_point, max_proc=max_proc_lin, save_prefix=sv_prfx)

# obj.loadLinTuning('sp', lin_point=lin_point, save_prefix=sv_prfx)
# obj.loadLinTuning('th', lin_point=lin_point, save_prefix=sv_prfx)


obj.plotPrediction(lin_point)





#####################################################
################# Eigenvector story #################
#####################################################
print '\n##### Eigenvector story #####'

obj.EVs(100)#'scipy eig'#'scipy eigvals'#'load'#int
# obj.EVs('load')
obj.QEVs()
obj.prep_decomp(lin_point)

try:    os.mkdir(obj.model_def_path+'/analysis/linearization/')
except: pass

obj.plotEVs()
obj.plotEVecs()
obj.plotEVecs('Q')

obj.plotEVdecomp(inv_type='exact')



#######################################
############# B & M story #############
#######################################
print '\n##### B & M story #####'

obj.prep_separation()
obj.calculate_separation()

splt_pts = np.cumsum(obj.n_neurons)[:-1]

save_folder = obj.model_def_path+'/analysis/linearization_EVs/'
try:    os.mkdir(save_folder)
except: pass

obj.plotSepInput(save_folder)
obj.plotSepOutput(save_folder)
obj.plotSepProjections(save_folder)


print 'mean absolte values of terms for B&M separation:'
print 'db\t', np.mean(np.abs(obj.db))

print 'Q dv_M\t', np.mean(np.abs(Q_mul(obj.q, obj.dv_M, obj.n_neurons)))
print 'S dv_B\t', np.mean(np.abs(S_mul(obj.dFdv, obj.q, obj.dv_B, obj.n_neurons)))

print 'Q dv_B\t', np.mean(np.abs(Q_mul(obj.q, obj.dv_B, obj.n_neurons)))
print 'S_dv_M\t', np.mean(np.abs(S_mul(obj.dFdv, obj.q, obj.dv_M, obj.n_neurons)))




#######################################
############# rescue story ############
#######################################
print '\n##### rescue story #####'

save_folder = obj.model_def_path+'/analysis/linearization_EVs/'
try:    os.mkdir(save_folder)
except: pass

obj.delete_mode7(lin_point, save_folder)
obj.supPop5(lin_point)



