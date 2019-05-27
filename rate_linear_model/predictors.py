import sys, os, socket, time, h5py
import numpy as np
from scipy.special import erfcx#, zetac
# from scipy.integrate import odeint
import scipy.linalg
import scipy.sparse, scipy.sparse.linalg, scipy.integrate
from multiprocessing import Queue, Process

# import matplotlib.pyplot as plt
import output; reload(output); import output as out

import functions; reload(functions); from functions import S_mul

from model import model_class



class predictor_class(model_class):

    def f_dth_rate(self, theta, th_rate):
        '''modulation part ("cos") of th input in Hz'''

        dth_rate = []
        for p, pop in enumerate(self.net.populations):
            if self.pref_angles[p].size == 0:
                dth_rate.append(np.zeros(self.n_neurons[p]))
            else:
                m = self.m[p%2]
                cos = np.cos(2. * (self.pref_angles[p] - theta))

                dth_rate.append(th_rate * m * cos)

        return np.concatenate(dth_rate)

    def calcInv(self, inv='exact'):

        t0 = time.time()
        out.printNB('calculating inverse (%s)...'%inv)

        if inv == 'exact':

            A = -self.dFdv.toarray().astype(np.float32)
            A[xrange(self.N),xrange(self.N)] += 1.

            self.inv = scipy.linalg.inv(A, overwrite_a=True, check_finite=False)

        elif inv == 'approx':
            self.inv_apprx = scipy.sparse.eye(self.N, format='csr') + self.dFdv

        elif inv == 'one':
            self.inv_one = scipy.sparse.eye(self.N, format='csr')

        print 'done (%.2fs)'%(time.time()-t0)

    def siegPredict(self, a, th_mode, plot=False, queue=None, **kwargs):
        '''a (int): idx of angle to use
        th_mode ('th' or 'sp'): state of thalamic population'''

        t0 = time.time()
        theta=self.angles[a]
        n_tc = 10       

        if 'der_tol' not in kwargs:
            kwargs['der_tol'] = 1e-6
        if 's_step' not in kwargs:
            kwargs['s_step'] = 10.
        if 's_max' not in kwargs:
            kwargs['s_max'] = 100

        if queue is None or a==0:
            print 'der_tol:', '%.1e'%kwargs['der_tol']
            print 's_step:', kwargs['s_step']
            print 's_max:', kwargs['s_max']

        s_step = kwargs['s_step']
        if plot:
            s_step  = s_step/10.

        if th_mode=='th':            
            dth_rate = self.f_dth_rate(theta, self.net.th_rate)
            th_rates = self.net.th_rate + dth_rate

        elif th_mode=='sp':
            th_rates = self.net.bg_rate

        elif th_mode=='lin0':
            th_rates = 0.

        dvds = lambda t, v, th_ratesp: (self.F(v, th_ratesp) - v)

        init = np.zeros(self.N)

        if queue is None or a==0:
            out.printNB('integrating...\n')

        t0 = time.time()

        if plot:
            times = [0.]
            res = [init.reshape((1,-1))]

        ode_obj = scipy.integrate.ode(dvds)
        ode_obj.set_f_params(th_rates)

        ode_obj.set_integrator('dopri5')#'vode', method='adams')#'lsoda')#'dopri5'

        ode_obj.set_initial_value(init, 0.)

        if queue is None or a==0:
            out.printNB('[')

        s = 0.
        n = 0        
        while True:
            t1 = time.time()
            n += 1

            s += s_step
            result = ode_obj.integrate(s)
            
            if plot:
                times.append(s)
                res.append(result.reshape((1,-1)))
            
            last_der = dvds(None, result, th_rates)

            if queue is None or a==0:
                # print n, s, '%.2e'%np.max(np.abs(last_der)), out.timeConversion(time.time()-t1)
                out.printNB('\r[')
                for i in range(n):
                    out.printNB(str((i+1)%10))
                    if (i+1)%10 == 0:
                        out.printNB('|')

                out.printNB(' %.4e'%np.max(np.abs(last_der)) + ' ' + out.timeConversion(time.time()-t1) + ' ')

            if not ode_obj.successful():
                raise UserWarning('Integration not successful')
            elif np.all(np.abs(last_der) < kwargs['der_tol']):
                break
            elif s == kwargs['s_max']:
                print '\nder:' + str(np.max(np.abs(last_der)))
                raise UserWarning('s_max reached without converging')
        
        if queue is None or a==0:
            print ']...done (%s)'%out.timeConversion(time.time()-t0)
            print 'waiting for other angles...'

        if plot:

            if queue is None:
                out.printNB('plotting...')

            res = np.concatenate(res, axis=0)
            res = np.split(res, np.cumsum(self.n_neurons)[:-1], axis=1)

            figsize = (16,8)
            f, axs_rec = plt.subplots(2,4,figsize=figsize,sharex=True)#,sharey=True)
            axs = axs_rec.flatten()
            for p, pop in enumerate(self.net.populations):
                ax = axs[p]

                for i in range(n_tc):
                    ax.plot(times, res[p][:,i], lw=2.)

                ylims = ax.get_ylim()
                xmax = float(ax.get_xlim()[1])
                k = np.int(np.ceil(xmax/kwargs['s_step']))
                for i in xrange(k):
                    ax.plot(np.ones(2) * i * kwargs['s_step'], [-1000, 1000], 'k--', lw=1.)
                ax.set_ylim(ylims)

                ax.set_title(pop)

            plt.suptitle('rates')
            axs_rec[-1,0].set_xlabel('s [a.u.]')
            axs_rec[-1,0].set_ylabel(r'$\nu$ [Hz]')

            save_folder=self.model_def_path+'/analysis/linearization/'
            plt.savefig(save_folder + 'integration_time_course_%i.png'%a)

            plt.show()

            if queue is None:
                print 'done'

        # print ':::::::::::', a, s, out.timeConversion(time.time()-t0)

        if queue is None:
            return result, last_der
        else:
            queue.put((a, result, last_der))

    def linSolve(self, syst, rhs, maxiter=None, prnt=False):

        W = self.dFdv

        if syst == 'W':

            if prnt: print '===== starting with solution of (1-W) x = rhs ====='
            x, info = scipy.sparse.linalg.bicg(scipy.sparse.eye(self.N) - W, 
                                               rhs)
            if prnt: print info, 'done'

            if info != 0:
                print('========== Warning: return status of bicg %i =========='%info)

        if syst == 'Q':

            if prnt: print '===== starting with solution of (1-Q) x = rhs ====='
            qt = self.q * self.n_neurons.reshape((1,8))

            x = scipy.linalg.solve(np.eye(8)-qt, rhs)
            if prnt: print 'done'

        if syst == 'S':

            if prnt: print '===== starting with solution of (1-S) x = rhs ====='
            def linfun(x):
                x = x.flatten()
                return x - S_mul(W, self.q, x, self.n_neurons)

            sSystem = scipy.sparse.linalg.LinearOperator(dtype=np.float32,
                                                         shape=(self.N, self.N),
                                                         matvec=linfun)

            x, info = scipy.sparse.linalg.cgs(sSystem, rhs)
            if prnt: print info, 'done'

            if info != 0:
                print('========== Warning: return status of bicg %i =========='%info)

        return x

    def linPredict(self, a, th_mode, lin_point, queue=None, method='bicg', maxiter=None):
        '''a (int): idx of angle to use
        th_mode (string): "th" or "sp"'''

        theta=self.angles[a]

        if queue is None:
            out.printNB('predicting...')

        if th_mode == 'th':
            dth_rate = self.f_dth_rate(theta, self.net.th_rate)

            if lin_point[-2:] == 'sp':
                dth_rate = (self.net.th_rate - self.net.bg_rate) + dth_rate

            elif lin_point[-4:] == 'lin0':
                dth_rate = self.net.th_rate + dth_rate

        elif th_mode == 'sp':

            if lin_point[-2:] == 'sp':
                dth_rate = np.zeros(self.N)

            elif lin_point[-4:] == 'lin0':
                dth_rate = self.net.bg_rate

        db = self.dFdv_th * dth_rate

        if self.I_input_override is not None:
            dg = self.dFdI * np.repeat(self.I_input_override, self.n_neurons)
        elif hasattr(self.sim, 'I_input'):
            dg = self.dFdI * np.repeat(self.sim.I_input, self.n_neurons)
        else:
            dg = 0.

        if method == 'inv':

            if self.inv is not None:
                dv_lin = self.inv.dot(db+dg)
            else:
                raise UserWarning('inverse not yet available')

        elif method == 'spsolve':
            dv_lin = scipy.sparse.linalg.spsolve(scipy.sparse.eye(self.N) - self.dFdv, 
                                                 db+dg)

        elif method == 'bicg':

            splt_pts = np.cumsum(self.n_neurons)[:-1]

            dv_lin = self.linSolve('W', db+dg, maxiter=maxiter, prnt=False)

        if queue is None:
            print 'done'

        if queue is None:
            return dv_lin, dth_rate
        else:
            queue.put((a, dv_lin, dth_rate))

    def createSiegTuning(self, th_mode, max_proc=12, save_prefix='', save_loc=None):

        if th_mode=='th':
            self.sieg_rates_th = [np.zeros((self.n_neurons[p],len(self.angles))) \
                                  for p in range(len(self.net.populations))]
        elif th_mode=='sp':
            self.sieg_rates_sp = [np.zeros((self.n_neurons[p],len(self.angles))) \
                                  for p in range(len(self.net.populations))]
        elif th_mode=='lin0':
            self.sieg_rates_lin0 = [np.zeros((self.n_neurons[p],len(self.angles))) \
                                    for p in range(len(self.net.populations))]

        print('\nstarting Siegert processes (%s)'%th_mode)
        t0 = time.time()
        queue = Queue()
        a, n = 0, 0 # started, finished, a-n: running
        while n < len(self.angles):

            if (a-n) == max_proc or a==len(self.angles):               

                a2, result, _ = queue.get()

                result = np.split(result, np.cumsum(self.n_neurons)[:-1])            
                for p, pop in enumerate(self.net.populations):
                    if th_mode=='th':
                        self.sieg_rates_th[p][:,a2] = result[p]
                    elif th_mode=='sp':
                        self.sieg_rates_sp[p][:,a2] = result[p]
                    elif th_mode=='lin0':
                        self.sieg_rates_lin0[p][:,a2] = result[p]

                n += 1

            if a < len(self.angles):

                p = Process(target=self.siegPredict, 
                            args=(a, th_mode),
                            kwargs={'der_tol': 1e-5, 's_step': 1., 's_max': 100., 
                                    'queue': queue, 'plot': False})
                p.start()

                a += 1

        print('done (%s)'%out.timeConversion(time.time()-t0))


        if save_loc is None:
            hdf5_output = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'a')
            sn_mf_group_t = 'sn_mf' + save_prefix + '/'
        else:
            hdf5_output = h5py.File(save_loc + 'dIscan_' + save_prefix + '.hdf5', 'w')
            sn_mf_group_t = '/'

        for p, pop in enumerate(self.net.populations):
            if sn_mf_group_t + 'sieg_rates_'+th_mode+'_'+pop in hdf5_output: 
                del hdf5_output[sn_mf_group_t + 'sieg_rates_'+th_mode+'_'+pop]

            if th_mode=='th':
                hdf5_output.create_dataset(sn_mf_group_t + 'sieg_rates_'+th_mode+'_'+pop, 
                                            data=self.sieg_rates_th[p])
            elif th_mode=='sp':
                hdf5_output.create_dataset(sn_mf_group_t + 'sieg_rates_'+th_mode+'_'+pop, 
                                            data=self.sieg_rates_sp[p])
            elif th_mode=='lin0':
                hdf5_output.create_dataset(sn_mf_group_t + 'sieg_rates_'+th_mode+'_'+pop, 
                                            data=self.sieg_rates_lin0[p])

        tmp = self.I_input_override
        if self.I_input_override is None:
            tmp = False

        hdf5_output[sn_mf_group_t].attrs['I_input_override'] = tmp

        hdf5_output.close()

    def createLinTuning(self, th_mode, lin_point='sieg_sp', method='bicg', max_proc=12, maxiter=None, save_prefix='', save_loc=None):

        dlin_rates = [np.zeros((self.n_neurons[p],len(self.angles))) \
                              for p in range(len(self.net.populations))]
        lin_rates = [np.zeros((self.n_neurons[p],len(self.angles))) \
                              for p in range(len(self.net.populations))]
        
        print('starting linear processes (%s), getting results...'%th_mode)
        queue = Queue()
        a, n = 0, 0 # started, finished, a-n: running
        while n < len(self.angles):

            if (a-n) == max_proc or a==len(self.angles):               

                a2, dlin_rates_tmp, foo = queue.get()
                dlin_rates_tmp = np.split(dlin_rates_tmp, np.cumsum(self.n_neurons)[:-1])
                for p, pop in enumerate(self.net.populations):
                    dlin_rates[p][:,a2] = dlin_rates_tmp[p].flatten()

                n += 1

            if a < len(self.angles):

                p = Process(target=self.linPredict, 
                            args=(a, th_mode, lin_point),
                            kwargs={'queue': queue, 
                                    'method': method,
                                    'maxiter': maxiter})
                p.start()

                a += 1

        print('done')


        if lin_point=='sieg_sp' and self.sieg_rates_sp is not None:
            for p, pop in enumerate(self.net.populations):
                lin_rates[p] = self.sieg_rates_sp[p] + dlin_rates[p]

        elif lin_point=='sieg_lin0' and self.sieg_rates_lin0 is not None:
            for p, pop in enumerate(self.net.populations):
                lin_rates[p] = self.sieg_rates_lin0[p] + dlin_rates[p]

        else:
            print '*** filling lin_rates with NAN out of lazyness ***'
            for p, pop in enumerate(self.net.populations):
                lin_rates[p] = lin_rates[p] * np.nan

        sn_mf_group_t = 'sn_mf' + save_prefix

        # save everything globally ...
        if th_mode == 'sp':
            self.dlin_rates_sp = dlin_rates
            self.lin_rates_sp = lin_rates

        if th_mode == 'th':
            self.dlin_rates_th = dlin_rates
            self.lin_rates_th = lin_rates

        # ... and to analysis file
        if save_loc is None:
            hdf5_output = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'a')
            sn_mf_group_t = 'sn_mf' + save_prefix + '/'
        else:
            hdf5_output = h5py.File(save_loc + 'dIscan_' + save_prefix + '.hdf5', 'r+')
            sn_mf_group_t = '/'

        dtitle = sn_mf_group_t + 'dlin_rates_' + th_mode + '_'
        title = sn_mf_group_t + 'lin_rates_' + th_mode + '_'

        hdf5_output[sn_mf_group_t].attrs['lin_point'] = lin_point

        for p, pop in enumerate(self.net.populations):
            if dtitle+pop in hdf5_output: 
                del hdf5_output[dtitle+pop]
            if title+pop in hdf5_output: 
                del hdf5_output[title+pop]

            hdf5_output.create_dataset(dtitle+pop, data=dlin_rates[p])
            hdf5_output.create_dataset(title+pop, data=lin_rates[p])

        hdf5_output.close()

    def loadSiegTuning(self, th_mode, save_prefix=''):

        hdf5_output = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'r')
        # hdf5_output = h5py.File('sn_mf_saves.hdf5', 'r')

        sn_mf_group_t = 'sn_mf' + save_prefix

        if th_mode=='th':
            self.sieg_rates_th = []
        elif th_mode=='sp':
            self.sieg_rates_sp = []
        elif th_mode=='lin0':
            self.sieg_rates_lin0 = []

        for p, pop in enumerate(self.net.populations):
            if sn_mf_group_t + '/sieg_rates_'+th_mode+'_'+pop in hdf5_output:

                if th_mode=='th':
                    self.sieg_rates_th.append(np.array(hdf5_output[sn_mf_group_t + '/sieg_rates_'+th_mode+'_'+pop]))
                elif th_mode=='sp':
                    self.sieg_rates_sp.append(np.array(hdf5_output[sn_mf_group_t + '/sieg_rates_'+th_mode+'_'+pop]))
                elif th_mode=='lin0':
                    self.sieg_rates_lin0.append(np.array(hdf5_output[sn_mf_group_t + '/sieg_rates_'+th_mode+'_'+pop]))
                
            else:
                raise UserWarning('no Sieg rates saved for ' + th_mode)

        hdf5_output.close()

    def loadLinTuning(self, th_mode, lin_point='sieg_sp', save_prefix=''):

        sn_mf_group_t_0 = 'sn_mf' + save_prefix

        # determine corrsponding name of variable
        dtitle = sn_mf_group_t_0 + '/dlin_rates_'  + th_mode + '_'
        title = sn_mf_group_t_0 + '/lin_rates_' + th_mode + '_'

        # get from analysis file
        hdf5_output = h5py.File(self.model_def_path + '/hdf5s/analysis.hdf5', 'r')

        if 'lin_point' in hdf5_output[sn_mf_group_t_0].attrs:
            if lin_point != hdf5_output[sn_mf_group_t_0].attrs['lin_point']:
                raise UserWarning('no lin tuning saved for lin_point ' + lin_point +\
                            ' but for '+ hdf5_output[sn_mf_group_t_0].attrs['lin_point'])
        else:
            print ' *** Warngin: no lin_point attribute in loadLin Tunig, ***'
            print ' *** unkonw for which linearization was calculated for ***'
        dlin_rates = []
        for p, pop in enumerate(self.net.populations):
            if dtitle+pop in hdf5_output:
                dlin_rates.append(np.array(hdf5_output[dtitle+pop]))
            else:
                raise UserWarning('no dlin rates saved')

        lin_rates = []
        for p, pop in enumerate(self.net.populations):
            if title+pop in hdf5_output:
                lin_rates.append(np.array(hdf5_output[title+pop]))
            else:
                raise UserWarning('no lin rates saved')

        hdf5_output.close()

        # save everything globally
        if th_mode == 'sp':
            self.dlin_rates_sp = dlin_rates
            self.lin_rates_sp = lin_rates
        if th_mode == 'th':
            self.dlin_rates_th = dlin_rates
            self.lin_rates_th = lin_rates
