import numpy as np
import os
# import h5py

import network_params as net; reload(net)
import sim_params as sim; reload(sim)

def get_output_path(th_rate, contr, m, Istr, n_digits=2):

    sim_spec = 'a%.1f_t%.1f'%(net.area, sim.t_sim * 1e-3)
    # sim_spec += '_g%i'%np.abs(net.g)
    # sim_spec += '_PSP%0.3f'%net.PSP_e
    # sim_spec += '_%iHz'%np.round(net.bg_rate)
    if th_rate%1. == 0:
        sim_spec += '_%iHz'%np.int(th_rate)
    else:
        sim_spec += '_%.2fHz'%th_rate
    if contr:
        sim_spec += '_c'
    sim_spec += '_mE%.2f'%m[0]
    # if m[1] != 0: sim_spec += '_mI%.2f'%m[1]
    sim_spec += '_mI%.2f'%m[1]
    if net.conn_rule == 'fixed_indegree':
        sim_spec += '_K'
    if Istr is not None:
        sim_spec += '%.2f'%Istr

    sim_spec += sim.path_extension
    
    output_path = sim.data_dir + sim_spec
    
    return output_path


# Compute PSC amplitude from PSP amplitude
# These are used as weights (mean for normal_clipped distribution)
def PSC(PSP, tau_m, tau_syn, C_m):
    # specify PSP and tau_syn_{ex, in}
    delta_tau   = tau_syn - tau_m
    ratio_tau    = tau_m / tau_syn
    PSC_over_PSP = C_m * delta_tau / (tau_m * tau_syn * (ratio_tau**(tau_m / delta_tau) - \
                    ratio_tau**(tau_syn / delta_tau)))
    return PSP * PSC_over_PSP
    
    
    
    
def timeConversion(t):
    t = int(np.round(t))
    # seconds
    if t < 60:
        return '%is'%t
    else:
        secs = t%60
        t = (t - secs)/60
        strng = '%is'%secs
        #minutes
        if t < 60:
            return '%im:'%t + strng
        else:
            mins = t%60
            t = (t - mins)/60
            strng = '%im:'%mins + strng
            #hours
            if t < 24:
                return '%ih:'%t + strng
            else:
                hrs = t%24
                t = (t - hrs)/24
                return '%id:'%t + '%ih:'%hrs + strng
    
