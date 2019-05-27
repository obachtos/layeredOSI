import sys, os, time, h5py
import numpy as np
import scipy.linalg
import scipy.sparse, scipy.sparse.linalg

import matplotlib.pyplot as plt
import style;reload(style); import style as sty
# import output; reload(output); import output as out

import functions; reload(functions); from functions import Q_mul, S_mul

from model import model_class



class bm_separation_class():

    def calc_p(self):

        if self.q is None:

            W = self.dFdv

            self.q = np.zeros((8,8))
            I = 0
            for i in range(8):
                J = 0
                for j in range(8):
                    self.q[i,j] = W[I:I+self.n_neurons[i], J:J+self.n_neurons[j]].mean()

                    J += self.n_neurons[j]
                I += self.n_neurons[i]


    def prep_separation(self, save=True, save_prefix=''):

        splt_pts = np.cumsum(self.n_neurons)[:-1]
        W = self.dFdv

        self.calc_p()

        ### inputs currents
        if self.I_input_override is not None:
            self.dg = self.dFdI * np.repeat(self.I_input_override, self.n_neurons)
        elif hasattr(self.sim, 'I_input'):
            self.dg = self.dFdI * np.repeat(self.sim.I_input, self.n_neurons)
        else:
            self.dg = np.zeros(self.N)

        self.dg_B8 = np.array(map(np.mean, np.split(self.dg, splt_pts)))
        self.dg_B = np.repeat(self.dg_B8, self.n_neurons)
        self.dg_M = self.dg - self.dg_B

        if save:
            hdf5_file = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'r+')

            if 'sn_mf/bm_sep' in hdf5_file:
                del hdf5_file['sn_mf/bm_sep']
                
            group = hdf5_file.create_group('sn_mf/bm_sep')

            group.create_dataset('q', data=self.q)
            group.create_dataset('dg_B8', data=self.dg_B8)
            group.create_dataset('dg_M', data=self.dg_M)

        hdf5_file.close()


    def calculate_separation(self, save=True, save_prefix=''):

        # W channel
        self.dv = self.linSolve('W', self.db + self.dg, prnt=True)

        # Q channel
        self.dv_B8 = self.linSolve('Q', self.db_B8 + self.dg_B8, prnt=True)
        self.dv_B = np.repeat(self.dv_B8, self.n_neurons)

        # S channel
        self.dv_M = self.linSolve('S', self.db_M + self.dg_M, prnt=True)

        if save:
            hdf5_file = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'r+')
                
            group = hdf5_file['sn_mf/bm_sep']

            group.create_dataset('dv', data=self.dv)
            group.create_dataset('dv_B8', data=self.dv_B8)
            group.create_dataset('dv_M', data=self.dv_M)

        hdf5_file.close()

    def plotSepInput(self, save_folder):

        # plt.figure()
        # plt.plot(self.db_B, label=r'$\Delta\beta_B$')
        # plt.legend(loc='best')
        # plt.savefig(save_folder + 'db_B.png')

        # plt.figure()
        # plt.plot(self.db_M, label=r'$\Delta\beta_M$')
        # plt.legend(loc='best')
        # plt.savefig(save_folder + 'db_M.png')

        plt.figure()
        plt.plot(self.db, label=r'$\Delta\beta$')
        plt.plot(self.db_B, label=r'$\Delta\beta_B$')
        plt.legend(loc='best')
        plt.savefig(save_folder + 'db.png')

        plt.figure()
        plt.plot(self.dg, label=r'$\Delta\gamma$')
        plt.plot(self.dg_B, label=r'$\Delta\gamma_B$')
        plt.legend(loc='best')
        plt.savefig(save_folder + 'dg.png')

    def plotSepOutput(self, save_folder):

        # plt.figure()
        # plt.plot(self.dv, label=r'$\Delta\nu = (1\!\!1-W)^{-1} \cdot \Delta\beta$')
        # plt.legend(loc='best')
        # plt.savefig(save_folder + 'dv.png')

        # plt.figure()
        # plt.plot(self.dv_B, label=r'$\Delta\nu_B = (1\!\!1-Q)^{-1} \cdot \Delta\beta_B$')
        # plt.legend(loc='best')
        # plt.savefig(save_folder + 'dv_B.png')

        # plt.figure()
        # plt.plot(self.dv_M, label=r'$\Delta\nu_M = (1\!\!1-S)^{-1} \cdot \Delta\beta_M$')
        # plt.legend(loc='best')
        # plt.savefig(save_folder + 'dv_M.png')

        plt.figure()
        plt.plot(self.dv, lw=3, label=r'$\Delta\nu$')
        plt.plot(self.dv_B + self.dv_M, lw=1, label=r'$\Delta\nu_B + \Delta\nu_M$')
        plt.legend(loc='best')
        plt.savefig(save_folder + 'dv_VS_dv_B+dv_M.png')


    def plotSepProjections(self, save_folder):

        W = self.dFdv
        splt_pts = np.cumsum(self.n_neurons)[:-1]

        plt.figure()
        plt.plot(self.dv_B, label=r'$\Delta\nu_B$')
        plt.legend(loc='best')
        plt.savefig(save_folder + 'dv_B.png')

        plt.figure()
        plt.plot(self.dv_M, label=r'$\Delta\nu_M$')
        plt.legend(loc='best')
        plt.savefig(save_folder + 'dv_M.png')

        ### output rates projections
        tmp1 = Q_mul(self.q, self.dv_B, self.n_neurons)
        plt.figure()
        plt.plot(tmp1, label=r'$Q\cdot\Delta\nu_B$')
        plt.legend(loc='best')
        plt.savefig(save_folder + 'Q*dv_B.png')

        tmp2 = Q_mul(self.q, self.dv_M, self.n_neurons)
        plt.figure()
        plt.plot(tmp2, label=r'$Q\cdot\Delta\nu_M$')
        plt.legend(loc='best')
        plt.savefig(save_folder + 'Q*dv_M.png')

        tmp3 = S_mul(W, self.q, self.dv_B, self.n_neurons)
        plt.figure()
        plt.plot(tmp3, label=r'$S\cdot\Delta\nu_B$')
        plt.legend(loc='best')
        plt.savefig(save_folder + 'S*dv_B.png')
        
        tmp4 = S_mul(W, self.q, self.dv_M, self.n_neurons)
        plt.figure()
        plt.plot(tmp4, label=r'$S\cdot\Delta\nu_M$')
        plt.legend(loc='best')
        plt.savefig(save_folder + 'S*dv_M.png')

        ### input rates projections
        plt.figure()
        plt.plot(Q_mul(self.q, self.db_B, self.n_neurons), label=r'$Q\cdot\Delta\beta_B$')
        plt.legend(loc='best')
        # plt.title('Q*db_B')
        plt.savefig(save_folder + 'Q*db_B.png')

        plt.figure()
        plt.plot(S_mul(W, self.q, self.db_B, self.n_neurons), label=r'$S\cdot\Delta\beta_B$')
        plt.legend(loc='best')
        # plt.title('S*db_B')
        plt.savefig(save_folder + 'S*db_B.png')

        plt.figure()
        plt.plot(Q_mul(self.q, self.db_M, self.n_neurons), label=r'$Q\cdot\Delta\beta_M$')
        plt.legend(loc='best')
        # plt.title('Q*db_M')
        plt.savefig(save_folder + 'Q*db_M.png')

        plt.figure()
        plt.plot(S_mul(W, self.q, self.db_M, self.n_neurons), label=r'$S\cdot\Delta\beta_M$')
        plt.legend(loc='best')
        # plt.title('S*db_M')
        plt.savefig(save_folder + 'S*db_M.png')


