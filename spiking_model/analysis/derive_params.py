import numpy as np

# Import specific moduls
import network_params as net; reload(net)
import sim_params as sim; reload(sim)
# from functions import *


def getControls(parameters):

    tmpPars = []
    for pars in parameters:
        if pars[:-2] not in tmpPars:
            tmpPars.append(pars[:-2])
            
    newPars = []
    for pars in tmpPars:
        newPars += [pars + (None, None)]
    
    return newPars


# Scale size of network
n_neurons      = np.rint(net.full_scale_n_neurons * net.area).astype(int)
n_populations   = len(net.populations)
n_layers        = len(net.layers)
n_types      = len(net.types)
n_total      = np.sum(n_neurons)
matrix_shape    = np.shape(net.conn_probs)  # shape of connection probability matrix

# Scale synapse numbers
if net.scale_K_linearly:
    n_outer_full    = np.outer(net.full_scale_n_neurons, net.full_scale_n_neurons)
    K_full_scale    = np.log(1. - net.conn_probs   ) / np.log(1. - 1. / n_outer_full)
    K_scaled        = np.int_(K_full_scale * net.area)
    if not net.n_th == 0:
        K_th_full_scale = np.log(1. - net.conn_probs_th) / \
            np.log(1. - 1. / (net.n_th * net.full_scale_n_neurons))
        K_th_scaled  = np.int_(K_th_full_scale * net.area)
else:
    n_outer      = np.outer(n_neurons, n_neurons)
    K_scaled        = np.int_(np.log(1. - net.conn_probs   ) / np.log(1. - 1. / n_outer))
    if not net.n_th == 0:
        K_th_scaled = np.int_(np.log(1. - net.conn_probs_th) / \
            np.log(1. - 1. / (net.n_th * n_neurons)))


K_th_neurons = (np.round(K_th_scaled.astype(float)/n_neurons)).astype(int)
K_bg = net.K_bg_0 - K_th_neurons



# numbers of neurons from which to record spikes and membrane potentials
n_neurons_rec_spike = np.rint(n_neurons * sim.frac_rec_spike).astype(int)
n_neurons_rec_voltage = np.rint(n_neurons * sim.frac_rec_voltage).astype(int)

n_rec_per_pop = np.minimum(n_neurons, sim.n_targets).astype(np.int)

    