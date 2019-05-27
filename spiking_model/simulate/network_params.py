import numpy as np


###################################################
###         Network parameters        ###        
###################################################

# area of network in mm^2; scales numbers of neurons
# use 1 for the full-size network (77,169 neurons)
area    = 1.

# Whether to scale number of synapses K linearly: K = K_full_scale * area.
# When scale_K_linearly is false, K is derived from the connection probabilities and
# scaled neuron numbers according to eq. (1) of the paper. In first order 
# approximation, this corresponds to K = K_full_scale * area**2.
# Note that this produces different dynamics compared to the original model.
scale_K_linearly  = True

layers  = ['L23', 'L4', 'L5', 'L6']
types = ['e', 'i'] 
populations = [layer + typus for layer in layers for typus in types]


full_scale_n_neurons = np.array( \
                        [20683,    # layer 2/3 e
                         5834,    # layer 2/3 i
                         21915,    # layer 4 e
                         5479,    # layer 4 i
                         4850,    # layer 5 e
                         1065,    # layer 5 i
                         14395,    # layer 6 e
                         2948])    # layer 6 i

# mean EPSP amplitude (mV) for all connections except L4e->L2/3e
PSP_e = .15
# factor for mean EPSP amplitude (mv) of L4e->L2/3e connections
f_L4e_to_L23e = 2.
# standard deviation of PSC amplitudes relative to mean PSC amplitudes
PSC_rel_sd = 0.1 
# IPSP amplitude relative to EPSP amplitude
g = -4.

# probabilities for >=1 connection between neurons in the given populations
# columns correspond to source populations; rows to target populations
# i. e. conn_probs[post, pre] = conn_prob[target, source]

# source      2/3e    2/3i    4e      4i      5e      5i      6e      6i       
conn_probs = np.array(
            [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.    , 0.0076, 0.    ],
             [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.    , 0.0042, 0.    ],
             [0.0077, 0.0059, 0.0497, 0.135 , 0.0067, 0.0003, 0.0453, 0.    ],
             [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.    , 0.1057, 0.    ],
             [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
             [0.0548, 0.0269, 0.0257, 0.0022, 0.06  , 0.3158, 0.0086, 0.    ],
             [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364, 0.001 , 0.0034, 0.0005, 0.0277, 0.008 , 0.0658, 0.1443]])
         
# factor for K from L4e to L2/3e
f_K_L4e_to_L23e = 1.
         
# mean dendritic delays for excitatory and inhibitory transmission (ms)
delay_e = 1.5   # ms, excitatory synapses
delay_i = 0.75  # ms, inhibitory synapses
# standard deviation relative to mean delays
delay_rel_sd = 0.5 

# Synapse dictionaries
# default connection dictionary
# conn_rule = 'fixed_total_number'
conn_rule = 'fixed_indegree'
autapses = False # only for 'fixed_indegree'
multapses = False

# weight distribution of connections between populations
weight_dict_exc = {'distribution': 'normal_clipped', 'low': 0.0} 
weight_dict_inh = {'distribution': 'normal_clipped', 'high': 0.0} 
# delay distribution of connections between populations
delay_dict  = {'distribution': 'normal_clipped', 'low': 0.1} 
# default synapse dictionary
syn_dict = {'model': 'static_synapse'}


###################################################
###          Single-neuron parameters        ###        
###################################################

neuron_model = 'iaf_psc_delta' # 'iaf_psc_exp' #

Vm0_mean = -58.0    # mean of initial membrane potential (mV)
Vm0_std = 10.0         # std of initial membrane potential (mV)

# neuron model parameters
model_params = {'tau_m': 10.,        # membrane time constant (ms)
                't_ref': 2.,        # absolute refractory period (ms)
                'E_L': -65.,        # resting membrane potential (mV)
                'V_th': -50.,        # spike threshold (mV)
                'C_m': 250.,        # membrane capacitance (pF)
                'V_reset': -65.        # reset potential (mV)
                } 

if neuron_model == 'iaf_psc_exp':
    model_params['tau_syn_ex'] = 0.5# excitatory synaptic time constant (ms)
    model_params['tau_syn_in'] = 0.5# inhibitory synaptic time constant (ms)

###################################################
###           Stimulus parameters        ###        
###################################################
 
# rate of background Poisson input at each external input synapse (spikes/s)
bg_rate = 8.       # Hz
th_rate = 30.

PSP_ext = 0.15 #      # mean EPSP amplitude (mV) for external input

# in-degrees for background input
K_bg_0 = np.array([
        1600,    # 2/3e
        1500,    # 2/3i
        2100,    # 4e
        1900,    # 4i
        2000,    # 5e
        1900,    # 5i
        2900,    # 6e
        2100])    # 6i
        

# optional additional thalamic input (Poisson)
# Set n_th to 0 to avoid this input.
# For producing Potjans & Diesmann (2012) Fig. 10, n_th = 902 was used.
# Note that the thalamic rate here reproduces the simulation results
# shown in the paper, and differs from the rate given in the text. 
n_th = 902    # size of thalamic population
PSP_th = 0.15    # mean EPSP amplitude (mV) for thalamic input

# connection probabilities for thalamic input
conn_probs_th = np.array(
        [0.0,        # 2/3e
         0.0,        # 2/3i    
         0.0983,    # 4e
         0.0619,    # 4i
         0.0,        # 5e
         0.0,        # 5i
         0.0512,    # 6e
         0.0196])    # 6i


# mean delay of thalamic input (ms)
delay_th = 1.5  
# standard deviation relative to mean delay of thalamic input
delay_th_rel_sd = 0.5  


