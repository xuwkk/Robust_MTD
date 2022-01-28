# -*- coding: utf-8 -*-

"""
THIS FILE CONTAINS THE MTD ALGORITHM FOR COMPLETE MTD CONFIGURATION
"""

# %% Import
from scipy.stats.distributions import chi2, ncx2
import numpy as np
from pypower.api import dcopf, ext2int, bustypes, printpf
from scipy.io import loadmat
from numpy.linalg import norm, inv, matrix_rank, svd
from copy import deepcopy
from scipy.optimize import minimize, NonlinearConstraint, Bounds, linprog
import random
from utils.grid_fun import dc_grid
import time
from utils.utils import x_to_b


# %% COMPLETE MTD
class complete():
    def __init__(self, case, v_mag, v_ang, x_max, x_min):
        
        case = deepcopy(case)
        
        self.r = case.r

        # jacobian matrix for default reactance
        self.Cr_N, self.V_N, self.AAr, self.Jr_N = case.jacobian(v_mag,v_ang,case.b)
        self.Pr_N, self.Sr_N = case.mtd_matrix(self.Jr_N)
        
        # constraint parameter
        self.x_max = x_max                                                         
        self.x_min = x_min                                                         
        
    # loss function
    def fun_loss(self,x):
        
        b = -x/(self.r**2+x**2)

        Jr_mtd_N = self.Cr_N + self.V_N@np.diag(b)@self.AAr  
        Pr_mtd_N = Jr_mtd_N@np.linalg.inv(Jr_mtd_N.T@Jr_mtd_N)@Jr_mtd_N.T 
        l2_norm = norm(self.Pr_N @ Pr_mtd_N, ord = 2)    ## calculate the L2 norm
        
        return l2_norm
    
    # boundary constraint
    def fun_bound(self):
        
        # don't forget to set zero on the non D-FACTS lines
        return Bounds(lb = self.x_min, ub = self.x_max)

# multi run
def run_complete(case, v_mag, v_ang, x_max, x_min, run_no):
    
    complete_ = complete(case, v_mag,v_ang, x_max, x_min)
    
    # summary and record
    success_no = 0
    loss_summary = []
    b_mtd_summary = []
    x_mtd_summary = []
    start_time = time.time()
    
    while True:
        x0 = x_min + (x_max-x_min)*np.random.rand(len(x_max)) ## optimization initial point 
            
        results = minimize(complete_.fun_loss, x0, method = "SLSQP", bounds = complete_.fun_bound())
        
        # analysis
        x_mtd = results.x
        b_mtd = x_to_b(case, x_mtd)
        assert np.all(x_mtd<=x_max)
        assert np.all(x_mtd>=x_min)
        
        if results.fun < 1:
            # only record the cos(theta_1) that is smaller than 1
            success_no += 1
            loss_summary.append(results.fun)               ## record the function loss
            b_mtd_summary.append(b_mtd)            ## record susceptance
            x_mtd_summary.append(x_mtd)
            
        if success_no == run_no:
            break
    
    end_time = time.time()
    
    # find the maximum smallest principal angle and the corresponding reactance
    b_mtd = b_mtd_summary[np.argmin(loss_summary)]                  ## reactance corresponding to smallest loss, e.g. smallest non-one singular value
    x_mtd = x_mtd_summary[np.argmin(loss_summary)]
    assert np.all(x_mtd<=x_max)
    assert np.all(x_mtd>=x_min)
    
    return b_mtd, x_mtd, (end_time - start_time)/run_no, loss_summary