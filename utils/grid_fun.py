# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 23:05:10 2021

@author: bumbl
"""

"""
The class and functions for basic grid operations
1). Construct the grid model
2). State estimation and Bad data detection
3). Random MTD
4). Update MTD matrices
5). Random attacks
"""

# %% Import
from scipy.stats.distributions import chi2
import numpy as np
from pypower.api import dcopf, ext2int, bustypes, dSbus_dV, dSbr_dV, ppoption, opf, makeYbus
from numpy.linalg import inv, norm, svd
from copy import deepcopy
import random
from scipy.stats import ncx2
from scipy.optimize import Bounds, minimize
import scipy
import time

from utils.utils import x_to_b

# %% Define the grid fundamentals
"""
Build the grid and some fundamental functions like state estimation and BDD settings
"""
class dc_grid:
    
    def __init__(self, mpc, gencost, genlimit_max, genlimit_min, flowlimit):
        
        # mpc: the case file loaded from pypower
        
        mpc = deepcopy(mpc)                  # deep copy to avoid curroption
        # we assume:
            
        if len(mpc['bus']) == 57:
            pass
        else:
            mpc['bus'][:,4:6] = 0                # shunt element
            mpc['branch'][:,4] = 0               # line charging susseptance
            mpc['branch'][:,8:10] = 0            # transformer
        
        # set the grid constraints
        mpc['gencost'][:,3] = 2                    # linear cost
        mpc['gencost'][:,4] = gencost              # linear cost
        mpc['gencost'][:,5:] = 0                   # no constant cost
        mpc['gen'][:,8] = genlimit_max             # generator power limit max
        mpc['gen'][:,9] = genlimit_min             # generator power limit min
        
        mpc['gen'][:,3] = genlimit_max             # generator reactive limit
        mpc['gen'][:,4] = -mpc['gen'][:,3]
        
        mpc['branch'][:,5] = flowlimit             # branch flow limit
        
        
        self.mpc_int = ext2int(mpc)                # interior representation by pypower
        self.mpc = mpc
        
        # find the index of reference buses
        self.ref_index, self.pv_index, self.pq_index = bustypes(self.mpc_int['bus'], self.mpc_int['gen'])
        self.non_ref_index = (list(self.pq_index) + list(self.pv_index))
        self.non_ref_index.sort()
        
        # define numbers
        self.baseMVA = self.mpc['baseMVA']
        self.no_bus = self.mpc['bus'].shape[0]                   # number of bus
        self.no_branch = self.mpc['branch'].shape[0]             # number of branch
        self.no_measure = 2*(self.no_bus + self.no_branch)       # in full ac setting            
        self.no_measure_ = self.no_branch                        # in jacobian ac setting
        self.no_gen = self.mpc['gen'].shape[0]          # number of generator
        
        # incidence matrix
        from_bus = self.mpc['branch'][:, 0]; # list of "from" buses
        to_bus = self.mpc['branch'][:, 1]; # list of "to" buses
        
        self.f = (from_bus - 1).astype('int')
        self.t = (to_bus - 1).astype('int')
        
        self.Cf = np.zeros((self.no_branch,self.no_bus))          # from bus incidence matrix
        self.Ct = np.zeros((self.no_branch,self.no_bus))          # to bus incidence matrix
        
        for i in range(self.no_branch):
            self.Cf[i,self.f.astype('int')[i]] = 1
            self.Ct[i,self.t.astype('int')[i]] = 1
        
        self.A = self.Cf - self.Ct                                # the full incidence matrix
        self.Ar = np.delete(self.A, self.ref_index, axis = 1)     # the reduced incidence matrix
        
        # generator incidence matrix
        self.Cg = np.zeros((self.no_bus,self.no_gen))
        for i in range(self.no_gen):
            self.Cg[int(self.mpc['gen'][i,0]-1),i] = 1
        
        # admittance matrix
        Ybus, Yf, Yt = makeYbus(self.mpc_int['baseMVA'], self.mpc_int['bus'], self.mpc_int['branch'])
        self.Ybus = scipy.sparse.csr_matrix.todense(Ybus).getA()
        self.Yf = scipy.sparse.csr_matrix.todense(Yf).getA()
        self.Yt = scipy.sparse.csr_matrix.todense(Yt).getA()
        
        # reactance matrix
        self.r = self.mpc['branch'][:,2]
        self.x = self.mpc['branch'][:,3]
        self.g = self.r/(self.r**2+self.x**2)
        self.b = -self.x/(self.r**2+self.x**2)
        
        # dc matrices
        self.Hr = np.diag(self.b)@self.Ar
        
        
    def prepare_BDD_variable(self, alpha, R, R_):
        # chi square test threshold for the ac-se
        self.alpha = alpha
        self.DoF = self.no_measure - 2*(self.no_bus - 1)             # degree of freedom of chi-square distribution
        self.BDD_threshold = chi2.ppf(1-alpha, df = self.DoF)      # threshold for chi-square test for normalized residual
        self.R = R
        self.R_inv = inv(R)
        self.R_inv_12 = np.sqrt(self.R_inv)
        
        # chi square test threshold for the jacobian based approximation
        self.DoF_ = self.no_measure_ - (self.no_bus - 1)             # degree of freedom of chi-square distribution
        self.BDD_threshold_ = chi2.ppf(1-alpha, df = self.DoF_)      # threshold for chi-square test for normalized residual
        self.R_ = R_
        self.R_inv_ = inv(R_)
        self.R_inv_12_ = np.sqrt(self.R_inv_)
    
    def lambda_eff(self, alpha):
        # Determine the critical lambda with different detection probability for effective MTD
        # lambda_c is the minimum value of the non centrality parameter to achieve to have 1-alpha detection probability

        # MTD effectiveness

        for i in range(100000):
            if np.abs(ncx2.cdf(self.BDD_threshold_, df = self.DoF_, nc = 0.005*i)-alpha) < 0.0001:
                break
        ncx_lambda = 0.005*i
        lambda_c_eff_ = ncx_lambda
            
        print(f"lambda_c_eff_ is {lambda_c_eff_}")
        print(f"P_eff is {1-ncx2.cdf(self.BDD_threshold_, df = self.DoF_, nc = lambda_c_eff_)}")
        print(f"BDD threshold_ is {self.BDD_threshold_}")
        
        return lambda_c_eff_

    def branch_incidence(self):
        for i in range(self.no_bus):
            print(f"bus {i+1}: {np.nonzero(self.A[:,i])[0]}")

    """
    JACOBIAN MATRIX
    """
    
    def flat_start(self):
        return np.ones((self.bus_no,)), np.zeros((self.bus_no,))
    
    def jacobian(self,v_mag, v_ang, b):
        # find the reduced jacobian matrix
        # Dishonest SE: v_mag and v_ang can be flat vectors.
        # Honest SE: v_mag and v_ang can be arbireary vectors from state estimation.
        
        B_mtd = np.diag(b)
        
        # define constants
        V_N = self.R_inv_12_@np.diag((self.Cf@v_mag)*(self.Ct@v_mag))
        Cr_N = -V_N@np.diag(self.g)@np.diag(np.sin(self.A@v_ang))@self.Ar
        AAr = np.diag(np.cos(self.A@v_ang))@self.Ar
        
        # reduced jacobian matrix
        Jr_N = Cr_N+V_N@B_mtd@AAr
        
        return Cr_N, V_N, AAr, Jr_N
    
    def jacobian_check(self,v_mag,v_ang):
        # can only check with default reactance
        v = v_mag*np.exp(1j*v_ang)   # complex voltage
        dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St = dSbr_dV(self.mpc_int['branch'], self.Yf, self.Yt, v)
        
        # get sub-matrix of H relating to line flow
        dPF_dVa = np.real(dSf_dVa) # from end
        Jr_N = self.R_inv_12_@dPF_dVa[:, self.non_ref_index]
        
        return -Jr_N
    
    def mtd_matrix(self,Jr_N):
        
        #Jr_N: normalized reduced jacobian matrix
        Pr_N = Jr_N@inv(Jr_N.T@Jr_N)@Jr_N.T
        Sr_N = np.eye(self.no_measure_) - Pr_N
        
        return Pr_N, Sr_N
    
    def find_lambda_residual(self, a, Jr_N):
        
        #a is NOT normalzied
        #a can be 1) attack vector (to find the lambda)
        #         2) attack vector with noise (to find residual)
        #         3) real measurement (to find residual))
        
        Pr_N, Sr_N = self.mtd_matrix(Jr_N)
        
        return norm(Sr_N@self.R_inv_12_@a, ord = 2)**2  # square
    
    def find_lambda_residual_ac(self, x, z, result):

        # ac se based residual
        # used in the new case file, e.g. case_new in full ac test
        # z: is the attacked measurement

        assert np.all(x == self.x)                       # test if we are in case_new

        v_mag_est, v_ang_est = self.ac_se(z, result)     # state estimation
        z_est = self.h_x(v_mag_est, v_ang_est)           # estimated measurement
        
        return (z - z_est).T@self.R_inv@(z-z_est), z_est

    def find_lambda_residual_ac_new(self, x, z, result):
        assert np.all(x == self.x)
        
        v_mag_est, v_ang_est = self.ac_se(z, result)     # state estimation
        z_est = self.h_x(v_mag_est, v_ang_est)   # estimated measurement
        self.DoF_ac = self.no_branch + self.no_bus - (self.no_bus-1)
        self.BDD_threshold_ac = chi2.ppf(1-self.alpha, df = self.DoF_ac)
        
        z_ac = z[:self.no_bus+self.no_branch]
        z_est_ac = z_est[:self.no_bus+self.no_branch]
        
        return (z_ac - z_est_ac).T@self.R_inv[:self.no_bus+self.no_branch,:self.no_bus+self.no_branch]@(z_ac-z_est_ac), z_est
        
    """
    OPTIMAL POWER FLOW
    """
    
    def ac_opf(self, active_power, reactive_power):
        
        mpc = deepcopy(self.mpc)
        mpc['bus'][:,2] = active_power
        mpc['bus'][:,3] = reactive_power
        ppopt = ppoption(verbose = 0, out_all = 0)
        result = opf(mpc, ppopt)
        PI = self.Cg@result['gen'][:,1] - result['bus'][:,2]
        PF = result['branch'][:,13]
        QI = self.Cg@result['gen'][:,2] - result['bus'][:,3]
        QF = result['branch'][:,14]
        
        z = np.concatenate([PI, PF, QI, QF], axis = -1)/self.baseMVA                                      ## in p.u.
        z_noise = z + np.random.multivariate_normal(mean = np.zeros((self.no_measure,)), cov = self.R)
        
        # update and record the state whenever the opf is called
        self.v_mag, self.v_ang = result['bus'][:,7], result['bus'][:,8]*np.pi/180
        
        return result, z, z_noise
    
    
    """
    STATE ESTIMATION
    """

    def h_x(self,v_mag, v_ang):
        
        # given state, finds the estimated measurement

        V = v_mag*np.exp(1j*v_ang)
        
        Sfe = np.diag(self.Cf@V)@np.conj(self.Yf)@np.conj(V)  # from
        #Ste = np.diag(self.Ct@V)@np.conj(self.Yt)@np.conj(V)  # to
        
        Sbus = np.diag(V)@np.conj(self.Ybus)@np.conj(V)       # injection
        
        z_est = np.concatenate([np.real(Sbus), np.real(Sfe), np.imag(Sbus), np.imag(Sfe)], axis = 0)
        
        return z_est

    def flat_state(self, result):
        v_mag = np.ones(self.no_bus)
        v_ang = np.zeros(self.no_bus)

        v_mag[self.ref_index] = result['bus'][self.ref_index,7]
        v_ang[self.ref_index] = result['bus'][self.ref_index,8]

        return v_mag, v_ang 

    def ac_se(self,z_noise,result):
        
        # the general ac state estimation

        # options
        tol     = 1e-5 # mpopt.pf.tol;
        max_it  = 20  # mpopt.pf.nr.max_it;
        
        # initialization flat start
        V0 = np.ones((self.no_bus,), np.cdouble)
        # it is important to correctly set the defualts at the reference bus
        #V0[self.ref_index] = self.mpc_int['bus'][self.ref_index,7] * np.exp( 1j*self.mpc_int['bus'][self.ref_index,8]*np.pi/180)
        V0[self.ref_index] = result['bus'][self.ref_index,7] * np.exp( 1j*result['bus'][self.ref_index,8]*np.pi/180)
        # Newton-Raphson iteration
        converged = 0
        i = 0 # iteration times = 0
        V = V0 
        Va = np.angle(V) # voltage angle (polar form) in rad
        Vm = np.abs(V)   # voltage magnitude (polar form) in p.u.
        
        while converged != 1 and i < max_it:
            
            i+=1

            Sfe = np.diag(self.Cf@V)@np.conj(self.Yf)@np.conj(V)
            Ste = np.diag(self.Ct@V)@np.conj(self.Yt)@np.conj(V)
            Sbus = np.diag(V)@np.conj(self.Ybus)@np.conj(V)
            z_est = np.concatenate([np.real(Sbus), np.real(Sfe), np.imag(Sbus), np.imag(Sfe)], axis = 0)
            
            # --- get Jacobian matrix ---
            dSbus_dVm, dSbus_dVa = dSbus_dV(self.Ybus, V)
            dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm, Sf, St = dSbr_dV(self.mpc_int['branch'], self.Yf, self.Yt, V)
            
            # get sub-matrix of H relating to line flow
            dPF_dVa = np.real(dSf_dVa) # from end
            dQF_dVa = np.imag(dSf_dVa)   
            dPF_dVm = np.real(dSf_dVm)
            dQF_dVm = np.imag(dSf_dVm)
            
            # get sub-matrix of H relating to generator output
            dPbus_dVa = np.real(dSbus_dVa)
            dQbus_dVa = np.imag(dSbus_dVa)
            dPbus_dVm = np.real(dSbus_dVm)
            dQbus_dVm = np.imag(dSbus_dVm)
            
            H = np.block([
                    [dPbus_dVa[:, self.non_ref_index], dPbus_dVm[:, self.non_ref_index]],
                    [dPF_dVa[:, self.non_ref_index],   dPF_dVm[:, self.non_ref_index]],
                    [dQbus_dVa[:, self.non_ref_index], dQbus_dVm[:, self.non_ref_index]],
                    [dQF_dVa[:, self.non_ref_index],   dQF_dVm[:, self.non_ref_index]]
                    ])
            
            # update
            J = np.transpose(H)@self.R_inv@H # Gain matrix
            F = np.transpose(H)@self.R_inv@(z_noise-z_est).reshape(self.no_measure,1) # evalute F(x)
            dx = np.array(np.linalg.inv(J)@F).flatten()
            
            # check for convergence
            normF = np.linalg.norm(F, np.inf)
            if normF < tol:
                converged = 1
            
            # update decision varialbe
            Va[self.non_ref_index] = Va[self.non_ref_index] + dx[:len(self.non_ref_index)]
            Vm[self.non_ref_index] = Vm[self.non_ref_index] + dx[len(self.non_ref_index):]
            
            V = Vm * np.exp(1j * Va)    # NOTE: angle is in radians in pf solver, but in degree in default case data
            
            Vm = np.abs(V)                          # update Vm and Va again in case
            Va = np.angle(V)                        # we wrapped around with a negative Vm
        
        #print(f'se no: {i}')
        return Vm, Va

    """
    MTD ALGORITHM
    """

    def random_mtd(self, x_max_ratio, x_min_ratio, dfacts_index):
        # the reactance should be inside the range
        # we note that the reactance should not be too small 
        # e.g. x_max_ratio = 0.2
        # e.g. x_min_ratio = 0.05
        # dfacts_index is the branches installed with D-FACTS devices : START FROM 0!!!!!
        
        pertur_ratio = x_min_ratio + (x_max_ratio - x_min_ratio)*np.random.rand(self.no_branch) # [0.05,0.2] positive
        pos_neg = np.random.choice([-1,1], self.no_branch)                                      # randomly [-0.2,-0.05] or [0.05,0.2]
        pertur_ratio_ = pertur_ratio * pos_neg
        
        x_mtd = np.array(self.x)
        x_mtd = x_mtd*(1+pertur_ratio_)
        
        # set 0 to the non D-FACTS devices
        non_dfact_branch = list(set(np.arange(self.no_branch)) - set(dfacts_index))
        x_mtd[non_dfact_branch] = self.x[non_dfact_branch]
        
        assert np.all(np.abs(x_mtd[dfacts_index]-self.x[dfacts_index])/self.x[dfacts_index] >= x_min_ratio) ## test 
        assert np.all(np.abs(x_mtd[dfacts_index]-self.x[dfacts_index])/self.x[dfacts_index] <= x_max_ratio)
        
        return x_mtd
    
    # max mtd
    def max_mtd(self, case, v_mag, v_ang, x_max, x_min, a, max_run_no):
        
        # given attack vector, find the reactance that can maximize the lambda
        # a is not normalized
        max_mtd_opt_ = max_mtd_opt(case, v_mag, v_ang, x_max, x_min, a)
        lambda_mtd_summary_cache = []
        b_mtd_summary_cache = np.zeros((max_run_no, self.b.shape[0]))
        
        start_time = time.time()
        
        for j in range(max_run_no):
            # multi-run to find the best result
            x0 = x_min + (x_max-x_min)*np.random.rand(len(self.b)) ## initial guess
        
            # optimize
            results = minimize(max_mtd_opt_.fun_loss, x0, method='SLSQP', bounds = max_mtd_opt_.fun_bound())
            x_mtd_cache = results.x
            assert np.all(x_mtd_cache<=x_max+1e-6)
            assert np.all(x_mtd_cache>=x_min-1e-6)
            
            # record
            b_mtd_cache = x_to_b(case,x_mtd_cache)
            lambda_mtd_summary_cache.append(results.fun)
            b_mtd_summary_cache[j,:] = b_mtd_cache
        
        end_time = time.time()
        
        # find the optimal x_mtd
        b_mtd = b_mtd_summary_cache[np.argmin(lambda_mtd_summary_cache),:]
        
        return b_mtd, (end_time - start_time)/max_run_no


    """
    ATTACK
    """
    def random_attack(self, Jr_N, attack_strength):
        
        # Jr_N: the jacobian matrix known by the attacker
        # for simplified setting, this is the original reactance

        # phase attack vector
        c = np.zeros((self.no_bus-1,))
        att_state_no = np.random.randint(1,self.no_bus)
        #att_state_no = np.random.randint(1,int(self.no_bus*0.3))
        att_state_posi = random.sample(list(np.arange(self.no_bus-1)), att_state_no)
        c[att_state_posi] = -1+2*np.random.rand(att_state_no)
        
        a = np.sqrt(self.R_)@Jr_N@c                                                       ## not normalized!!
        ratio = attack_strength * np.sqrt(np.sum(self.R_)) / np.linalg.norm(a,2)    ## normalize a according to rho
        a = a*ratio
        a_noise = a + np.random.multivariate_normal(np.zeros(self.no_measure_), self.R_)
        
        return a, a_noise
    
    def worst_point(self, Jr_N, Jr_mtd_N, k_min):
        # k_min is the smallest nonzero principal angle
        
        Pr_N, Sr_N = self.mtd_matrix(Jr_N)
        Pr_mtd_N, Sr_mtd_N = self.mtd_matrix(Jr_mtd_N)
        
        # singular value decomposition
        U,singular_value,V = svd(Pr_N@Pr_mtd_N)
        largest_non_one_singular_value = singular_value[k_min]
        worst_point = U[:,k_min]                      # the worst case attack
        
        return worst_point, largest_non_one_singular_value

    def worst_attack(self, attack_strength, worst_point):
        
        a = np.sqrt(self.R_)@worst_point    # unnormalized
        # normalize the attack vector so that |a|_2^2 is with respect to the attack strength rho
        ratio = attack_strength * np.sqrt(np.sum(self.R_)) / np.linalg.norm(a,2)
        a = a*ratio
        a_noise = a + np.random.multivariate_normal(np.zeros(self.no_measure_), self.R_)
        return a, a_noise
    
    def fix_no_attack(self, Jr_N, attack_strength, att_state_no):
        """
        attack attack on specific number of state
        """
        # phase attack vector
        c = np.zeros((self.no_bus-1,))                                                  ## empty state attack vector exluding the reference bus
        att_state_posi = random.sample(list(np.arange(self.no_bus-1)), att_state_no)    ## find the attack position
        c[att_state_posi] = -1+2*np.random.rand(att_state_no)                           ## randomly generate c
        
        # attack vector
        a = np.sqrt(self.R_)@Jr_N@c                                         # not normalized
        ratio = attack_strength * np.sqrt(np.sum(self.R)) / np.linalg.norm(a,2)
        a = a*ratio
        a_noise = a + np.random.multivariate_normal(np.zeros(self.no_measure_), self.R_)
        
        return a, a_noise, c

    def random_ac_attack(self, v_mag, v_ang, z_noise, state_att_ratio):
        
        # state_attack_ratio: a suitable range of attack strength on the phase angle in ac setting, 
        # self defined in initialization
        
        v_mag = deepcopy(v_mag)
        v_ang = deepcopy(v_ang)
        
        c = np.zeros((self.no_bus,))
        att_state_no = np.random.randint(1, self.no_bus)  # randomly generate attack state number
        att_state_posi = random.sample(self.non_ref_index, att_state_no) # randomly assign the attack position on the non-reference bus
        
        # attack vector on the phase angle
        v_ang_a = np.zeros((self.no_bus,))
        v_ang_a[att_state_posi] = v_ang[att_state_posi] * (-state_att_ratio + state_att_ratio *2 * np.random.rand(att_state_no))  # attack strength
        v_ang_a_ = v_ang + v_ang_a                          # attacked phase angle vector

        # construact the measurement attack vector
        z_est = self.h_x(v_mag, v_ang)                      # measurement basing on old reactance and new state 
        z_a_est = self.h_x(v_mag, v_ang_a_)                 # measurement basing on old reactance and new state
        z_noise_a = z_noise + z_a_est - z_est
        
        return z_noise_a # the attack vector constrains attack on [PI,PF,QI,QF]
    
    
        
    """
    Evaluation metric
    """
    
    def evaluation_metric(self, Jr_N, Jr_mtd_N, a, a_noise):
        
        # given attack vector a and MTD specification, return the evaluation metric
        # a : attack vector without noise
        # a : attack vector with noise
        # b_mtd : suceptance after MTD
        # lambda_c : the critical lambda
        
        # BDD detection
        residual = self.find_lambda_residual(a_noise, Jr_N)        
        if residual >= self.BDD_threshold_:  
            BDD = 1
        else: 
            BDD = 0
        
        # MTD residual
        residual = self.find_lambda_residual(a_noise, Jr_mtd_N)       
        if residual >= self.BDD_threshold_:
            MTD = 1
        else:
            MTD = 0
        
        # detection probability
        lambda_eff = self.find_lambda_residual(a, Jr_mtd_N)           
        detection_prob = 1-ncx2.cdf(self.BDD_threshold_, df = self.DoF_, nc = lambda_eff)

        # 
        output = {}
        output['BDD'] = BDD
        output['MTD'] = MTD
        output['detection_prob'] = detection_prob
        output['lambda_eff'] = lambda_eff
        
        return output

# %%
"""
MAX_MTD
"""
class max_mtd_opt():
    def __init__(self, case, v_mag, v_ang, x_max, x_min, a):

        # a is not normalized
        # case: the gird class instance
        
        case = deepcopy(case)
        self.r = case.r

        # jacobian matrix for default reactance
        self.Cr_N, self.V_N, self.AAr, self.Jr_N = case.jacobian(v_mag,v_ang,case.b)
        self.Pr_N, self.Sr_N = case.mtd_matrix(self.Jr_N)
        
        # constraint parameter
        self.x_max = x_max                                                         
        self.x_min = x_min                                          
        
        self.a_N = case.R_inv_12_@a
        
    def fun_loss(self,x):

        b = -x/(self.r**2+x**2)

        # prepare matrices
        Jr_mtd_N = self.Cr_N + self.V_N@np.diag(b)@self.AAr  
        Pr_mtd_N = Jr_mtd_N@np.linalg.inv(Jr_mtd_N.T@Jr_mtd_N)@Jr_mtd_N.T 
        
        return -np.linalg.norm((np.eye(Pr_mtd_N.shape[0]) - Pr_mtd_N)@self.a_N, 2)
    
    def fun_bound(self):
        return Bounds(lb = self.x_min, ub = self.x_max) 