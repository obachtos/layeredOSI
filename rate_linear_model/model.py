import time, sys, os, socket, h5py
import numpy as np
from scipy.special import erfcx
# import scipy.linalg
import scipy.sparse#, scipy.sparse.linalg, scipy.integrate
# from multiprocessing import Queue, Process

# import matplotlib.pyplot as plt
# import style;reload(style); import style as sty
import output; reload(output); import output as out

# import functions; reload(functions); from functions import *



class model_class():

    def f_mu(self, v, th_rate):

        if self.I_input_override is not None:

            if self.I_input_override.size == 8:
                I = np.repeat(self.I_input_override, self.n_neurons)
            elif self.I_input_override.size == self.N:
                I = self.I_input_override

        elif hasattr(self.sim, 'I_input'):
            I = np.repeat(self.sim.I_input, self.n_neurons) * 1e3 # pA -> fA

        else:
            I = 0.

        tmp = self.connectivity.dot(v) + \
              self.K_bg_all * self.net.PSP_ext * self.net.bg_rate + \
              self.K_th_all * self.net.PSP_th * th_rate + \
              I / self.C_m

        return self.tau_m * tmp

    def f_sig2(self, v, th_rate):

        tmp = self.connectivity2.dot(v) + \
              self.K_bg_all * self.net.PSP_ext**2. * self.net.bg_rate + \
              self.K_th_all * self.net.PSP_th**2. * th_rate

        return self.tau_m * tmp
        
    def f_dmudv(self):

        return self.tau_m * self.connectivity

    def f_dsigdv(self, v, th_rate):

        sig2 = self.f_sig2(v, th_rate).reshape((-1,1))

        tmp = scipy.sparse.csr_matrix(self.tau_m/(2.*np.sqrt(sig2)))

        return self.connectivity2.multiply(tmp)

    def f_dmudv_bg(self):

        return self.tau_m * self.K_bg_all * self.net.PSP_ext

    def f_dmudv_th(self):

        return self.tau_m * self.K_th_all * self.net.PSP_th

    def f_dsigdv_bg(self, v, th_rate):

        sig2 = self.f_sig2(v, th_rate)
        return self.tau_m * self.K_bg_all * self.net.PSP_ext**2. / (2. * np.sqrt(sig2))

    def f_dsigdv_th(self, v, th_rate):

        sig2 = self.f_sig2(v, th_rate)
        return self.tau_m * self.K_th_all * self.net.PSP_th**2. / (2. * np.sqrt(sig2))

    def Phi(self, v, th_rate):
            
        mu = self.f_mu(v, th_rate)
        sig2 = self.f_sig2(v, th_rate)
        sig = np.sqrt(sig2)
        
        integral = np.vectorize(lambda lower, upper: \
                        scipy.integrate.quad(lambda x: erfcx(-x), lower, upper)[0])

        low_lim = (self.V_r-mu)/sig
        up_lim = (self.V_th-mu)/sig
        
        return self.tau_ref + self.tau_m * np.sqrt(np.pi) * integral(low_lim, up_lim)

    def dPhi(self, v, th_rate):

        mu = self.f_mu(v, th_rate).reshape((-1,1))
        sig2 = self.f_sig2(v, th_rate).reshape((-1,1))
        sig = np.sqrt(sig2)
        
        dmudv = self.f_dmudv()
        dmudv_bg = self.f_dmudv_bg().reshape((-1,1))
        dmudv_th = self.f_dmudv_th().reshape((-1,1))
        dmudI = self.tau_m/self.C_m

        dsigdv = self.f_dsigdv(v, th_rate)
        dsigdv_bg = self.f_dsigdv_bg(v, th_rate).reshape((-1,1))
        dsigdv_th = self.f_dsigdv_th(v, th_rate).reshape((-1,1))


        up_lim = (self.V_th-mu)/sig
        low_lim = (self.V_r-mu)/sig

        a_up  = scipy.sparse.csr_matrix(erfcx(-up_lim)/sig2)
        a_low = scipy.sparse.csr_matrix(erfcx(-low_lim)/sig2)

        b_0 = scipy.sparse.csr_matrix(sig)
        b_1 = scipy.sparse.csr_matrix(mu-self.V_th)
        b_2 = scipy.sparse.csr_matrix(mu-self.V_r)

        dPhidv = (dsigdv.multiply(b_1) - dmudv.multiply(b_0)).multiply(a_up) - \
                 (dsigdv.multiply(b_2) - dmudv.multiply(b_0)).multiply(a_low)
        
        dPhidv_bg = erfcx(-up_lim) * ((mu-self.V_th) * dsigdv_bg - sig * dmudv_bg) / sig2 - \
                    erfcx(-low_lim) * ((mu-self.V_r) * dsigdv_bg - sig * dmudv_bg) / sig2

        dPhidv_th = erfcx(-up_lim) * ((mu-self.V_th) * dsigdv_th - sig * dmudv_th) / sig2 - \
                    erfcx(-low_lim) * ((mu-self.V_r) * dsigdv_th - sig * dmudv_th) / sig2

        dPhidmu = 1./sig * (erfcx(-low_lim) - erfcx(-up_lim))

        dPhidv = self.tau_m * np.sqrt(np.pi) * dPhidv
        dPhidv_bg = self.tau_m * np.sqrt(np.pi) * dPhidv_bg.flatten()
        dPhidv_th = self.tau_m * np.sqrt(np.pi) * dPhidv_th.flatten()
        dPhidI = self.tau_m * np.sqrt(np.pi) * dPhidmu.flatten() * dmudI

        return dPhidv, dPhidv_bg, dPhidv_th, dPhidI

    def dF(self, v_point, save_prefix=''):
        t0 = time.time()
        out.printNB('calculating derivatives (%s)...'%v_point)

        self.dF_point = v_point

        # if isinstance(v, basestring):
        if v_point[-2:] == 'sp':
            th_rate = self.net.bg_rate
        elif v_point[-2:] == 'th':
            th_rate = self.net.th_rate
        elif v_point[-4:] == 'lin0':
            th_rate = 0.

        if v_point == 'spk_sp':
            v = np.mean(np.concatenate(self.v_sp, axis=0), axis=1)
        elif v_point == 'spk_th':
            v = np.mean(np.concatenate(self.v_th, axis=0), axis=1)
            
        elif v_point == 'sieg_sp':
            v = np.mean(np.concatenate(self.sieg_rates_sp, axis=0), axis=1)
        elif v_point == 'sieg_th':
            v = np.mean(np.concatenate(self.sieg_rates_th, axis=0), axis=1)
        elif v_point == 'sieg_lin0':
            v = np.mean(np.concatenate(self.sieg_rates_lin0, axis=0), axis=1)

        p = self.Phi(v, th_rate)
        p = 1./p**2
        p_csr = scipy.sparse.csr_matrix(p.reshape((-1,1)))

        dPhidv, dPhidv_bg, dPhidv_th, dPhidI = self.dPhi(v, th_rate)

        self.dFdv = -dPhidv.multiply(p_csr)
        self.dFdv_bg = -dPhidv_bg * p
        self.dFdv_th = -dPhidv_th * p
        self.dFdI = -dPhidI * p

        ### save derivatives to analysis.hdf5
        hdf5_output = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'a')

        sn_mf_group_t = 'sn_mf' + save_prefix

        if sn_mf_group_t + '/dFdv_data' in hdf5_output: 
            del hdf5_output[sn_mf_group_t + '/dFdv_data']
            del hdf5_output[sn_mf_group_t + '/dFdv_row']
            del hdf5_output[sn_mf_group_t + '/dFdv_col']

            del hdf5_output[sn_mf_group_t + '/dFdv_th']

        if sn_mf_group_t + '/dFdv_bg' in hdf5_output: 
            del hdf5_output[sn_mf_group_t + '/dFdv_bg']
        
        if sn_mf_group_t + '/dFdI' in hdf5_output: 
            del hdf5_output[sn_mf_group_t + '/dFdI']


        coo = self.dFdv.asformat('coo')

        hdf5_output.create_dataset(sn_mf_group_t + '/dFdv_data', data=coo.data.astype(np.float32), 
                                    compression=self.sim.compress_records)
        hdf5_output.create_dataset(sn_mf_group_t + '/dFdv_row', data=coo.row.astype(np.int32), 
                                    compression=self.sim.compress_records)
        hdf5_output.create_dataset(sn_mf_group_t + '/dFdv_col', data=coo.col.astype(np.int32), 
                                    compression=self.sim.compress_records)

        coo = None

        hdf5_output.create_dataset(sn_mf_group_t + '/dFdv_bg', data=self.dFdv_bg)
        hdf5_output.create_dataset(sn_mf_group_t + '/dFdv_th', data=self.dFdv_th)
        hdf5_output.create_dataset(sn_mf_group_t + '/dFdI', data=self.dFdI)
        hdf5_output[sn_mf_group_t].attrs['dF_point'] = self.dF_point

        hdf5_output.close()

        print 'done (%.2fs)'%(time.time()-t0)

    def dF_load(self, v_point, save_prefix='', lowMem=False):

        out.printNB('loading derivatives (%s)...'%v_point)
        
        hdf5_output = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'r')

        sn_mf_group_t = 'sn_mf' + save_prefix

        if 'dF_point' in hdf5_output[sn_mf_group_t].attrs.keys() and v_point == hdf5_output[sn_mf_group_t].attrs['dF_point']:

            if not lowMem:
                data = np.array(hdf5_output[sn_mf_group_t + '/dFdv_data'])
                row = np.array(hdf5_output[sn_mf_group_t + '/dFdv_row'])
                col = np.array(hdf5_output[sn_mf_group_t + '/dFdv_col'])

                self.dFdv = scipy.sparse.coo_matrix((data, (row, col)), 
                                                shape=(self.N, self.N)).asformat('csr')

            self.dFdv_bg = np.array(hdf5_output[sn_mf_group_t + '/dFdv_bg'])
            self.dFdv_th = np.array(hdf5_output[sn_mf_group_t + '/dFdv_th'])
            self.dFdI = np.array(hdf5_output[sn_mf_group_t + '/dFdI'])
            self.dF_point = hdf5_output[sn_mf_group_t + ''].attrs['dF_point']

        else:
            hdf5_output.close()
            raise UserWarning('dFdv not saved for %s'%v_point)
            
        hdf5_output.close()

        print 'done'

    def F(self, v, th_rate):

        Phi = self.Phi(v, th_rate)
        
        return 1./Phi
