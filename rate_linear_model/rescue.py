import sys, os, time, h5py
import numpy as np
import scipy.linalg
import scipy.sparse, scipy.sparse.linalg

import matplotlib.pyplot as plt
import style;reload(style); import style as sty
# import output; reload(output); import output as out

import functions; reload(functions); from functions import Q_mul, S_mul

from model import model_class



class rescue_class():

    def delete_mode7(self, lin_point, save_folder):

        splt_pts = np.cumsum(self.n_neurons)[:-1]

        # prepare input
        dg = - self.xi[7] * self.Qevecs[:,7]
        dg8 = - self.xi[7] * self.Qevecs8[:,7]

        dFdI8 = np.array(map(np.mean, np.split(self.dFdI, splt_pts)))
        dI_B8 = dg8/dFdI8

        # dI = dg/self.dFdI
        # dI_B8 = np.array(map(np.mean, np.split(dI, splt_pts)))

        dI_B = np.repeat(dI_B8, self.n_neurons)
        print 'dI_B8.real', dI_B8.real * 1e-3, 'pA\n'
        print 'dI_B8.imag', dI_B8.imag * 1e-3, 'pA'

        # np.save(save_folder + 'dI_del', dI * 1e-3)
        # np.save(save_folder + 'dI_del_B8', dI_B8 * 1e-3)

        dgR = self.dFdI * dI_B

        # solve W and Q system
        dv_I = self.linSolve('W', self.db + dgR, prnt=True)

        dv_BI8 = self.linSolve('Q', self.db_B8 + np.real(dFdI8 * dI_B8),
                               prnt=True)

        dv_BI = np.repeat(dv_BI8, self.n_neurons)

        ### plot stuff
        # plot input
        # plt.figure()
        # plt.plot(dgR.real, label=r'$\Delta\gamma_{del, actual}$')
        # plt.plot(dg.real, label=r'$\Delta\gamma_{del}$')
        # plt.ylabel('[Hz]')
        # plt.legend(loc='best')

        plt.figure()
        plt.plot(np.repeat(dI_B8.real, self.n_neurons) * 1e-3, label=r'$\Delta I_{del}$')
        plt.ylabel('[pA]')
        plt.legend(loc='best')

        # plot dOutput
        # plt.figure()
        # plt.plot(dv_I.real, label=r'$\Delta\nu_{I_{del}} = (1\!\!1-W)^{-1} \cdot (\Delta\beta + \Delta I_{del})$')
        # plt.ylabel('[Hz]')
        # plt.legend(loc='best')
        # # plt.savefig(save_folder + 'dv_I.png')

        plt.figure()
        plt.plot(dv_I.real, label=r'$\Delta\nu_{I_{del}} = (1\!\!1-W)^{-1} \cdot (\Delta\beta + \Delta I_{B, del})$')
        plt.plot(dv_BI.real, label=r'$\Delta\nu_{B,I_{del}} = (1\!\!1-Q)^{-1} \cdot (\Delta\beta + \Delta I_{B, del})$')
        plt.plot(self.dv_B, label=r'$\Delta\nu_B$')
        plt.ylabel('[Hz]')
        plt.legend(loc='best')
        # plt.savefig(save_folder + 'dv_B_I.png')

        # plot Output
        if lin_point[-2:] == 'sp':
            v = self.sieg_rates_sp
        elif lin_point[-4:] == 'lin0':
            v = self.sieg_rates_lin0
        
        v_B8 = np.array(map(np.mean, v))
        v_B = np.repeat(v_B8, self.n_neurons)

        v = np.concatenate(map(lambda arr: np.mean(arr, axis=1), v))
        print type(v), len(v), v.shape

        plt.figure()
        plt.plot(dv_I.real + v, label=r'$\nu_{I_{del}}$')
        plt.ylabel('[Hz]')
        plt.legend(loc='best')
        # plt.savefig(save_folder + 'dv_I.png')

        plt.figure()
        plt.plot(dv_BI.real + v, label=r'$\nu_{B,I_{del}}$')
        plt.plot(self.dv_B + v_B, label=r'$\nu_B$')
        plt.ylabel('[Hz]')
        plt.legend(loc='best')
        # plt.savefig(save_folder + 'dv_B_I.png')

    def supPop5(self, lin_point):

        splt_pts = np.cumsum(self.n_neurons)[:-1]

        qt = self.q * self.n_neurons.reshape((1,8))
        self.gains_B = np.linalg.inv(np.eye(8)-qt)

        dv8 = np.zeros(8)
        dv8[4] = -np.mean(self.lin_rates_th[4])
        dg8 = np.dot(np.eye(8)-qt, dv8)

        dFdI8 = np.array(map(np.mean, np.split(self.dFdI, splt_pts)))
        dI_B8 = dg8/dFdI8

        dI_B = np.repeat(dI_B8, self.n_neurons)
        print 'dI_B8.real', dI_B8.real * 1e-3, 'pA\n'
        print 'dI_B8.imag', dI_B8.imag * 1e-3, 'pA'

        dgR = self.dFdI * dI_B

        # solve W and Q system
        dv_I = self.linSolve('W', self.db + dgR, prnt=True)

        dv_BI8 = self.linSolve('Q', self.db_B8 + np.real(dFdI8 * dI_B8),
                               prnt=True)
        dv_BI = np.repeat(dv_BI8, self.n_neurons)

        ### plot stuff
        # plot input
        # plt.figure()
        # plt.plot(dgR.real, label=r'$\Delta\gamma_{sup, actual}$')
        # # plt.plot(dg.real, label=r'$\Delta\gamma_{sup}$')
        # plt.ylabel('[Hz]')
        # plt.legend(loc='best')

        plt.figure()
        plt.plot(np.repeat(dI_B8.real, self.n_neurons) * 1e-3, label=r'$\Delta I_{sup}$')
        plt.ylabel('[pA]')
        plt.legend(loc='best')

        # plot dOutput
        # plt.figure()
        # plt.plot(dv_I.real, label=r'$\Delta\nu_{I_{sup}} = (1\!\!1-W)^{-1} \cdot (\Delta\beta + \Delta I_{sup})$')
        # plt.ylabel('[Hz]')
        # plt.legend(loc='best')
        # # plt.savefig(save_folder + 'dv_I.png')

        plt.figure()
        plt.plot(dv_I.real, label=r'$\Delta\nu_{I_{sup}} = (1\!\!1-W)^{-1} \cdot (\Delta\beta + \Delta I_{B, sup})$')
        plt.plot(dv_BI.real, label=r'$\Delta\nu_{B,I_{sup}} = (1\!\!1-Q)^{-1} \cdot (\Delta\beta + \Delta I_{B, sup})$')
        plt.plot(self.dv_B, label=r'$\Delta\nu_B$')
        plt.ylabel('[Hz]')
        plt.legend(loc='best')
        # plt.savefig(save_folder + 'dv_B_I.png')

        # plot Output
        if lin_point[-2:] == 'sp':
            v = self.sieg_rates_sp
        elif lin_point[-4:] == 'lin0':
            v = self.sieg_rates_lin0

        v_B8 = np.array(map(np.mean, v))
        v_B = np.repeat(v_B8, self.n_neurons)

        v = np.concatenate(map(lambda arr: np.mean(arr, axis=1), v))
        print type(v), len(v), v.shape

        plt.figure()
        plt.plot(dv_I.real + v, label=r'$\nu_{I_{sup}}$')
        plt.ylabel('[Hz]')
        plt.legend(loc='best')
        # plt.savefig(save_folder + 'dv_I.png')

        plt.figure()
        plt.plot(dv_BI.real + v, label=r'$\nu_{B,I_{sup}}$')
        plt.plot(self.dv_B + v_B, label=r'$\nu_B$')
        plt.ylabel('[Hz]')
        plt.legend(loc='best')
        # plt.savefig(save_folder + 'dv_B_I.png')
