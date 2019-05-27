import sys, os, socket, time, h5py
import numpy as np
# from scipy.special import erfcx#, zetac
# from scipy.integrate import odeint
import scipy.linalg
import scipy.sparse, scipy.sparse.linalg#, scipy.integrate
# from multiprocessing import Queue, Process
from datetime import datetime

# import matplotlib.pyplot as plt
# import style;reload(style); import style as sty
import output; reload(output); import output as out

# import functions; reload(functions); from functions import *


class calc_EV_class():

    def EVs(self, method=10, save=True):

        if method=='load':
            hdf5_file = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'r')

            if not 'dF_point' in hdf5_file.attrs:
                hdf5_file.close()
                raise UserWarning("can't load, no EVs saved")

            out.printNB('loading EVs from file (%s)...'%hdf5_file.attrs['dF_point'])
            
            self.evs = np.array(hdf5_file['lin_EVs'])
            if 'lin_EVecs' in hdf5_file:
                self.evecs = np.array(hdf5_file['lin_EVecs'])

            hdf5_file.close()

            print('done')

            return

        out.printNB('calculating EVs (%s)...'%str(method))
        out.printNB(datetime.fromtimestamp(time.time()).strftime("%A, %B %d, %Y %I:%M:%S") + '...')
        evecs = None
        t0 = time.time()

        if type(method) == int:
            evs, evecs = scipy.sparse.linalg.eigs(self.dFdv, k=method, which='LM')
        elif method == 'scipy eigvals':
            evs = scipy.linalg.eigvals(self.dFdv.toarray())
        elif method == 'scipy eig':
            evs, evecs = scipy.linalg.eig(self.dFdv.toarray())
        else:
            raise UserWarning('EV method not known: %s'%method)

        print 'done (%s)'%out.timeConversion(time.time()-t0)

        # sort EVs acording to magnitude
        idxs = np.argsort(np.abs(evs))[::-1]
        evs = evs[idxs]
        evecs = evecs[:,idxs]

        self.evs, self.evecs = evs, evecs

        # save to analysis HDF5
        if save:
            hdf5_file = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'r+')
            
            if 'lin_EVs' in hdf5_file: 
                del hdf5_file['lin_EVs']

            hdf5_file.create_dataset('lin_EVs', data=evs)

            if evecs is not None:
                if 'lin_EVecs' in hdf5_file: 
                    del hdf5_file['lin_EVecs']

                n_sv = 50
                if type(method) == int and method<n_sv:
                    n_sv = method
                hdf5_file.create_dataset('lin_EVecs', data=evecs[:,:n_sv])

            hdf5_file.attrs['dF_point'] = self.dF_point

            hdf5_file.close()

    def QEVs(self, save=True):
        
        t0 = time.time()
        out.printNB('calculating QEVs...')

        if self.q is None:
            self.q = np.zeros((8,8))
            I = 0
            for i in range(8):
                J = 0
                for j in range(8):
                    self.q[i,j] = self.dFdv[I:I+self.n_neurons[i], J:J+self.n_neurons[j]].mean() 

                    J += self.n_neurons[j]
                I += self.n_neurons[i]

        qt = self.q * self.n_neurons.reshape((1,8))

        Qevs, Qevecs = scipy.linalg.eig(qt)

        # sort EVs acording do magnitude
        idxs = np.argsort(np.abs(Qevs))[::-1]
        Qevs = Qevs[idxs]
        Qevecs = Qevecs[:,idxs]

        # scale evecs back up
        Qevecs = np.repeat(Qevecs, self.n_neurons, axis=0)

        # normalize
        Qevecs = Qevecs/np.linalg.norm(Qevecs,axis=0).reshape((1,8))

        self.Qevs, self.Qevecs = Qevs, Qevecs

        if save:
            hdf5_file = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'r+')

            if 'sn_mf/qev' in hdf5_file:
                del hdf5_file['sn_mf/qev']
                
            group = hdf5_file.create_group('sn_mf/qev')

            group.create_dataset('q', data=self.q)
            group.create_dataset('Qevs', data=self.Qevs)
            group.create_dataset('Qevecs', data=self.Qevecs)

        hdf5_file.close()

        print 'done (%s)'%out.timeConversion(time.time()-t0)


    def prep_decomp(self, lin_point, save=True):

        ### calculate decomposition
        splt_pts = np.cumsum(self.n_neurons)[:-1]

        ### inputs spikes                    
        dth_rate = self.f_dth_rate(0., self.net.th_rate)
        if lin_point[-2:] == 'sp':
            dth_rate = (self.net.th_rate - self.net.bg_rate) + dth_rate
        elif lin_point[-4:] == 'lin0':
            dth_rate = self.net.th_rate + dth_rate

        self.db = self.dFdv_th * dth_rate

        self.db_B8 = np.array(map(np.mean, np.split(self.db, splt_pts)))
        self.db_B = np.repeat(self.db_B8, self.n_neurons)
        self.db_M = self.db - self.db_B

        # calculate mean step EVecs
        self.Qevecs8 = np.split(self.Qevecs[:,:8], splt_pts, axis=0)
        self.Qevecs8 = np.vstack(map(lambda x: np.mean(x, axis=0), self.Qevecs8))

        # make basis transformation
        trans = np.linalg.inv(self.Qevecs8)
        self.xi = np.dot(trans, self.db_B8)

        # transform EVs
        self.Qevs_tmp = 1./(1-self.Qevs[:8])

        # calculate products
        self.mul = self.xi * self.Qevs_tmp
        self.rhos = self.mul.reshape((1,8)) * self.Qevecs8
        self.dlin_rates_EV = np.real(np.sum(self.rhos, axis=1))

        if save:
            hdf5_file = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'r+')

            if 'sn_mf/decomp' in hdf5_file:
                del hdf5_file['sn_mf/decomp']
                
            group = hdf5_file.create_group('sn_mf/decomp')

            group.create_dataset('db', data=self.db)
            group.create_dataset('db_B8', data=self.db_B8)
            group.create_dataset('db_M', data=self.db_M)
            group.create_dataset('Qevecs8', data=self.Qevecs8)
            group.create_dataset('xi', data=self.xi)
            group.create_dataset('mul', data=self.mul)
            group.create_dataset('rhos', data=self.rhos)
            group.create_dataset('dlin_rates_EV', data=self.dlin_rates_EV)

        hdf5_file.close()