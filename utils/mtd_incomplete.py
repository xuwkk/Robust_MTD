# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:26:54 2021

@author: wx3418
"""

"""
This file contains the proposed robust MTD algorithms
1. robust MTD for the grid with complete MTD configuration (normally the hiddenness is not considered for the grid with complete MTD configuration)
2. robust MTD for the grid with incomplete MTD configuration (with/out hidden)
    a. firstly find a warm start by minimizing the Frobenius norm
    b. second, starting at the warm start, minimize the l2 norm on the largest none-one singular value
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

# %%
"""
Rpbust MTD for complete MTD configuration
maximize the smallest principal angle while keeping the hiddenness - COMPLETE MTD
Normaly the hiddenness is not required for complete MTD, but you can set it for test
We don't need column constraint for grid with complete configuration
"""

# %%
# the upper loop: minimize the upper bound : which is represented by the Frob norm
class incomplete_fro():
    def __init__(self, case, v_mag, v_ang, x_max, x_min, col_constraint):
        
        case = deepcopy(case)
        
        self.r = case.r

        # jacobian matrix for default reactance
        self.Cr_N, self.V_N, self.AAr, self.Jr_N = case.jacobian(v_mag,v_ang,case.b)
        self.Pr_N, self.Sr_N = case.mtd_matrix(self.Jr_N)
        
        # constraint parameter
        self.x_max = x_max                                                         
        self.x_min = x_min                                                         
        self.nonlinear_constraint_max = col_constraint   
        
    # loss function
    def fun_loss(self,x):
        
        b = -x/(self.r**2+x**2)

        Jr_mtd_N = self.Cr_N + self.V_N@np.diag(b)@self.AAr  
        Pr_mtd_N = Jr_mtd_N@np.linalg.inv(Jr_mtd_N.T@Jr_mtd_N)@Jr_mtd_N.T 
        fro_norm = norm(self.Pr_N @ Pr_mtd_N, ord = "fro")    ## calculate the Frobenious norm
        
        return fro_norm
    
    # nonlinear constraint
    def fun_constraint(self,x):
        
        b = -x/(self.r**2+x**2)

        Jr_mtd_N = self.Cr_N + self.V_N@np.diag(b)@self.AAr  
        Pr_mtd_N = Jr_mtd_N@np.linalg.inv(Jr_mtd_N.T@Jr_mtd_N)@Jr_mtd_N.T 
        
        con = []  
        # find the cos of each bus
        for i in range(Jr_mtd_N.shape[-1]):
            # the consine of each column of H_N
            con.append(norm(Pr_mtd_N @ self.Jr_N[:,i:i+1], ord = 2)/norm(self.Jr_N[:,i:i+1], ord = 2))
        
        return con 
        
    def nonlinear_constraint(self):
        
        # construct the NonlinearConstraint class
        return NonlinearConstraint(fun = self.fun_constraint, 
                                   lb = 0, 
                                   ub = self.nonlinear_constraint_max)
    
    # boundary constraint
    def fun_bound(self):
        
        # don't forget to set zero on the non D-FACTS lines
        return Bounds(lb = self.x_min, ub = self.x_max)


# %%
# the lower loop : maximize the smallest nonzero principal angle 
# iteratively find the solution around the wart start of Frobenius norm

class incomplete_l2():
    def __init__(self, case, v_mag, v_ang, x_max, x_min, col_constraint, U_k):
        
        case = deepcopy(case)
        self.r = case.r

        # jacobian matrix for default reactance
        self.Cr_N, self.V_N, self.AAr, self.Jr_N = case.jacobian(v_mag,v_ang,case.b)
        self.Pr_N, self.Sr_N = case.mtd_matrix(self.Jr_N)
        
        # constraint parameter
        self.x_max = x_max                                                         
        self.x_min = x_min                                                         
        self.nonlinear_constraint_max = col_constraint
        
        # projector to the intersection subspace
        self.P_k = U_k@U_k.T                                                  
        
    def fun_loss(self,x):
        
        b = -x/(self.r**2+x**2)

        # prepare matrices
        Jr_mtd_N = self.Cr_N + self.V_N@np.diag(b)@self.AAr  
        Pr_mtd_N = Jr_mtd_N@np.linalg.inv(Jr_mtd_N.T@Jr_mtd_N)@Jr_mtd_N.T 
        l2_norm = norm(self.Pr_N @ Pr_mtd_N  - self.P_k, ord = 2)               
        
        return l2_norm
    
    # nonlinear constraint
    def fun_constraint(self,x):
        
        b = -x/(self.r**2+x**2)

        Jr_mtd_N = self.Cr_N + self.V_N@np.diag(b)@self.AAr  
        Pr_mtd_N = Jr_mtd_N@np.linalg.inv(Jr_mtd_N.T@Jr_mtd_N)@Jr_mtd_N.T 
        
        con = []  
        # find the cos of each bus
        for i in range(Jr_mtd_N.shape[-1]):
            # the consine of each column of H_N
            con.append(norm(Pr_mtd_N @ self.Jr_N[:,i:i+1], ord = 2)/norm(self.Jr_N[:,i:i+1], ord = 2))
        
        return con 
        
    def nonlinear_constraint(self):
        
        # construct the NonlinearConstraint class
        return NonlinearConstraint(fun = self.fun_constraint, 
                                   lb = 0, 
                                   ub = self.nonlinear_constraint_max)
    
    def fun_bound(self):
        # don't forget to set zero on the non D-FACTS lines
        return Bounds(lb = self.x_min, ub = self.x_max)



# %% multi-run of incomplete MTD algorithm
def run_incomplete(case, v_mag, v_ang, x_max, x_min, col_constraint, k_min, run_no):
    
    # default jacobian and mtd matrices
    _,_,_,Jr_N = case.jacobian(v_mag,v_ang,case.b)
    Pr_N, Sr_N = case.mtd_matrix(Jr_N)
    
    # instance the optimization incomplete_fro
    incomplete_fro_ = incomplete_fro(case, v_mag, v_ang, x_max, x_min, col_constraint)
    
    # summary and record
    success_no = 0            ## record the successful number
    loss_summary = []         ## summary on the largest non-one singular value : the solution of incomplete_l2 after convergence
    b_mtd_summary = []        ## the MTD perturbation of each run
    x_mtd_summary = []
    
    start_time = time.time()  ## record the time
    
    # multi-run loop
    while True:
        
        x0 = x_min + (x_max-x_min)*np.random.rand(len(case.x))               
        
        # solve the incomplete_fro problem to find the wart start using SLSQP
        results = minimize(incomplete_fro_.fun_loss, x0, method = "SLSQP", 
                           constraints = incomplete_fro_.nonlinear_constraint(), 
                           bounds = incomplete_fro_.fun_bound(), 
                           options = {'ftol': 1e-5})
        
        # verify on the solution
        x_mtd = results.x
        assert np.all(x_mtd <= x_max + 1e-6)
        assert np.all(x_mtd >= x_min - 1e-6)
        
        # find jacobian matrices
        b_mtd = x_to_b(case, x_mtd)
        
        _,_,_,Jr_mtd_N = case.jacobian(v_mag,v_ang,b_mtd)
        
        # rank of the composite matrix
        composite_matrix = np.concatenate([Jr_N, Jr_mtd_N], axis = -1)   
        rank_com = matrix_rank(composite_matrix)                             
        k = 2*(case.no_bus-1) - rank_com                                     
        
        Pr_mtd_N, Sr_mtd_N = case.mtd_matrix(Jr_mtd_N)
        
        if k != k_min or results.success == False:
            continue
        else:
        
            # if the k_min is satisfied, then solve the incomplete_l2 optimization problem
            singular_value_non_one_pre = 100                                ## set as a large value to pass the initial break test
            
            for _ in range(10):
                
                # set the maximum iteration number of the lower problem as 50
                # do T-SVD
                U, singular_value, V_transpose = svd(Pr_N@Pr_mtd_N)         ## the SVD can slow the optimization problem
                U_k = U[:,:k_min]                                           ## the orhornormal basis of the intersection subspaces
                singular_value_non_one = singular_value[k_min]              ## the largest non one singular value
                V_k = V_transpose.T[:,:k_min]                               ## the orthonormal basis of the intersection subspaces 
                
                print(singular_value_non_one)
                assert np.max(np.abs(U_k-V_k)) < 1e-6                       ## test the difference between U_k and V_k is small
                
                # instance the incomplete_l2 optimization problem 
                incomplete_l2_ = incomplete_l2(case, v_mag, v_ang, x_max, x_min, col_constraint, U_k)
                
                x0 = x_mtd                                                  ## use the previous running result as the initial point
                
                results = minimize(incomplete_l2_.fun_loss, x0, method = "SLSQP", 
                                   constraints = incomplete_l2_.nonlinear_constraint(),
                                   bounds = incomplete_l2_.fun_bound(),
                                   options = {'ftol': 1e-5})
                
                # analysis
                x_mtd = results.x
                
                assert np.all(x_mtd <= x_max + 1e-6)
                assert np.all(x_mtd >= x_min - 1e-6)

                # find jacobian matrices
                b_mtd = x_to_b(case, x_mtd)
                _,_,_,Jr_mtd_N = case.jacobian(v_mag,v_ang,b_mtd)
                
                # rank of the composite matrix
                composite_matrix = np.concatenate([Jr_N, Jr_mtd_N], axis = -1)   
                rank_com = matrix_rank(composite_matrix)                             
                k = 2*(case.no_bus-1) - rank_com                                     
                
                Pr_mtd_N, Sr_mtd_N = case.mtd_matrix(Jr_mtd_N)
                
                # termination condition
                if np.abs(singular_value_non_one_pre-singular_value_non_one) < 1e-7 and results.success == True:
                    
                    U, singular_value, V_transpose = svd(Pr_N@Pr_mtd_N)         
                    U_k = U[:,:k_min]                                           
                    singular_value_non_one = singular_value[k_min]              
                    V_k = V_transpose.T[:,:k_min]                               
                    loss_summary.append(singular_value_non_one)     ## record the value of largest non one singular value
                    b_mtd_summary.append(b_mtd)                     ## record the mtd perturbation
                    x_mtd_summary.append(x_mtd)
                    
                    # print(np.arccos(singular_value_non_one))        ## see the minimal non-zero principal angle                      
                    break                                           ## break the incomplete_l2 loop
            
                singular_value_non_one_pre = singular_value_non_one ## update the running result
            
            if success_no == run_no-1:
                break                                               ## break the multi run loop
            
            success_no += 1
            
    b_mtd = b_mtd_summary[np.argmin(loss_summary)]                  ## reactance corresponding to smallest loss, e.g. smallest non-one singular value
    x_mtd = x_mtd_summary[np.argmin(loss_summary)]
    
    end_time = time.time()
    # print(loss_summary)   
    return b_mtd, x_mtd, (end_time - start_time)/run_no, loss_summary                    ## return the perturbation result and the running time for each run
    
    
    
    

    