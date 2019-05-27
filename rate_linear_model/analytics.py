import os,sys, h5py#, tsys, ime, socket
import numpy as np
# from scipy.special import erfcx#, zetac
# from scipy.integrate import odeint
# import scipy.linalg
import scipy.sparse#, scipy.sparse.linalg, scipy.integrate
# from multiprocessing import Queue, Process

# import matplotlib.pyplot as plt
# import style;reload(style); import style as sty
# import output; reload(output); import output as out

# import functions; reload(functions); from functions import *


import model; reload(model); from model import model_class
import predictors; reload(predictors); from predictors import predictor_class
import calc_EV; reload(calc_EV); from calc_EV import calc_EV_class
import plot; reload(plot); from plot import plot_class
import plot_EV; reload(plot_EV); from plot_EV import plot_EV_class
import bm_separation; reload(bm_separation); from bm_separation import bm_separation_class
import rescue; reload(rescue); from rescue import rescue_class



class analytics(model_class, predictor_class, calc_EV_class, plot_class, 
                        plot_EV_class, bm_separation_class, rescue_class):

    def __init__(self, model_def_path, control_path, lowMem=False):
        '''
        Mind: everything here is working in === s/Hz ===
        '''

        self.model_def_path = model_def_path
        self.control_path = control_path

        wd = os.getcwd()    
        os.chdir(model_def_path)
        sys.path.append('.')
        import network_params as net; reload(net)
        import sim_params as sim; reload(sim)
        os.chdir(wd)

        self.net = net
        self.sim = sim

        ### neuron model parameters
        self.tau_ref = net.model_params['t_ref'] * 1e-3
        self.tau_m = net.model_params['tau_m'] * 1e-3

        self.C_m = net.model_params['C_m']

        self.V_th = net.model_params['V_th'] - net.model_params['E_L']
        self.V_r = net.model_params['V_reset'] - net.model_params['E_L']

        ### connectivity parametes
        ### (copy from derived_parameters)
        self.n_neurons = np.rint(net.full_scale_n_neurons * net.area).astype(int)
        self.N = np.sum(self.n_neurons)
        self.n_neurons_small = np.rint(net.full_scale_n_neurons * .1).astype(int)

        K_th = np.int_(np.log(1. - net.conn_probs_th) / \
                            np.log(1. - 1. / (net.n_th * net.full_scale_n_neurons)))
        self.K_th_per_neuron = (np.round(K_th.astype(float)/net.full_scale_n_neurons)).astype(int)
        self.K_bg = net.K_bg_0 - self.K_th_per_neuron

        self.K_th_all = np.repeat(self.K_th_per_neuron, self.n_neurons)
        self.K_bg_all = np.repeat(self.K_bg, self.n_neurons)

        ### read connectivity
        print 'loading: '
        print '\t"' + model_def_path + '/hdf5s/data.hdf5"'
        hdf5_file = h5py.File(model_def_path + '/hdf5s/data.hdf5', 'r')

        if lowMem:
            print '\tNOT loading connectivity'
            self.connectivity = None
            self.connectivity2 = None

        else:
            print '\tloading connectivity...'
            connectivity_Is = np.array(hdf5_file['connectivity_Is'], dtype=np.int32)
            connectivity_Js = np.array(hdf5_file['connectivity_Js'], dtype=np.int32)
            connectivity_weights = np.array(hdf5_file['connectivity_weights'])#, dtype=np.float32)

            self.connectivity = scipy.sparse.coo_matrix((connectivity_weights,
                                                    (connectivity_Is,
                                                     connectivity_Js)),
                                                   shape=(self.N, self.N)).tocsr()
            self.connectivity2 = self.connectivity.multiply(self.connectivity)

            print '\tsize of connectivity: %.3f GB'%self.getCSRsize(self.connectivity)

        self.pref_angles = []
        for p, pop in enumerate(net.populations):
            if 'pref_angles_'+pop in hdf5_file:
                self.pref_angles.append(np.array(hdf5_file['pref_angles_'+pop]))
            else:
                self.pref_angles.append(np.array([]))

        self.angles = hdf5_file.attrs['angles']/180. * np.pi
        self.m = hdf5_file.attrs['m']

        hdf5_file.close()

        ### read rates/OSVs
        hdf5_file = h5py.File(model_def_path + '/hdf5s/analysis.hdf5', 'r')

        self.v_th_mean = []
        self.v_th = []
        self.POs = []
        self.OSIs = []
        for p, pop in enumerate(net.populations):
            self.v_th.append(np.array(hdf5_file[pop + '/rates']))
            self.v_th_mean.append(np.mean(self.v_th[-1], axis=1))

            self.POs.append(np.array(hdf5_file[pop + '/POs']))
            self.OSIs.append(np.array(hdf5_file[pop + '/OSIs']))

        hdf5_file.close()

        ### read spontaneous rates
        hdf5_file = h5py.File(control_path + '/hdf5s/analysis.hdf5', 'r')

        self.v_sp_mean = []
        self.v_sp = []
        for p, pop in enumerate(net.populations):
            self.v_sp.append(np.array(hdf5_file[pop + '/rates']))
            self.v_sp_mean.append(np.mean(self.v_sp[-1], axis=1))

        hdf5_file.close()

        print 'done loading'

        self.I_input_override = None # pA
        # self.I_input_override = np.array([]) * 1e3 # pA -> nA

        self.inv = None
        self.inv_apprx = None
        self.inv_one = None

        self.dFdv, self.dFdv_bg, self.dFdv_th, self.dFdI = None, None, None, None
        self.dF_point = None

        self.sieg_rates_th = None
        self.sieg_rates_sp = None
        self.sieg_rates_lin0 = None

        self.sieg_rates_th_mean = None
        self.sieg_rates_sp_mean = None

        self.dlin_rates_th = None
        self.lin_rates_th = None
        self.dlin_rates_sp = None
        self.lin_rates_sp = None

        self.q = None
        self.evs, self.evecs = None, None
        self.Qevs, self.Qevecs = None, None

    def getCSRsize(self, csr):
        # in GB
        return (csr.data.nbytes + csr.indptr.nbytes + csr.indices.nbytes)/1024.**3



