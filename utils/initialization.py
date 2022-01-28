# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 06:34:09 2022

@author: bumbl
"""
# %%
import numpy as np
from pypower.api import case6ww, case14,case57, opf
from utils.grid_fun import dc_grid
import copy
from utils.settings import *
from utils.utils import b_range

if grid_choice == 0:
    
    """
    IEEE case-6ww : complete MTD
    """

    mpc = case6ww() # load the model
    gencost = [10,20,20]
    genlimit_max = [300,100,200]
    # genlimit_min = [20,10,10]
    genlimit_min = [0,0,0]
    flowlimit = 200*np.ones(mpc['branch'].shape[0])
    case = dc_grid(mpc, gencost, genlimit_max, genlimit_min, flowlimit)
    name = "case6"
    k_min = 0                              ## the vulnerable point
    
    
    R = np.diag(np.ones((case.no_measure,))) * sigma**2           ## measurement noise covariance matrix
    R_ = np.diag(np.ones(case.no_measure_)) * sigma**2            ## for the jacobian setting 
    
    case.prepare_BDD_variable(alpha, R, R_)
    
    dfacts_index = np.arange(mpc['branch'].shape[0])  ## perturb all the lines
    
    
    ## the estimated phase attack strength when generating ac attack 
    att_state_ratio = 0.4                  

elif grid_choice == 1:
    """
    case14
    """
    mpc = case14() # load the model
    gencost = [20,20,40,40,40]
    genlimit_max = [330,140,100,100,100]
    genlimit_min = [0,0,0,0,0]
    flowlimit = 300*np.ones(mpc['branch'].shape[0])
    name = "case14"
    
    case = dc_grid(mpc, gencost, genlimit_max, genlimit_min, flowlimit)
    
    # test ac-opf
    result = opf(mpc)
    PI = case.Cg@result['gen'][:,1] - result['bus'][:,2]
    PF = result['branch'][:,13]
    QI = case.Cg@result['gen'][:,2] - result['bus'][:,3]
    QF = result['branch'][:,14]
    
    z = np.concatenate([PI, PF, QI, QF], axis = -1)/case.baseMVA  ## in p.u.
    
    R = np.diag(np.ones((case.no_measure,))) * sigma**2           ## measurement noise covariance matrix
    R_ = np.diag(np.ones(case.no_measure_)) * sigma**2            ## for the jacobian setting 
    
    #sigma_new = (z + 0.01)*0.1
    #R = np.diag(sigma_new**2)
    #R_ = R[case.no_bus:case.no_bus+case.no_branch, case.no_bus:case.no_bus+case.no_branch]
    
    case.prepare_BDD_variable(alpha, R, R_)
    name = "case14"
    
    if dfacts_choise == 0:                                ## start from 0
        dfacts_index = np.arange(mpc['branch'].shape[0])  ## perturb all the lines
        k_min = 6                                         ## The value of minimum k that can be achieved for this D-FACTS placement
        col_constraint_value = 0.999
        
    elif dfacts_choise == 1:
        dfacts_index = np.array([2,3,4,12,15,18,20])-1    ## minimum number full rank covering all buses
        k_min = 6
        if ratio_choice == 0:
            col_constraint_value = 0.9997
        elif ratio_choice == 1:
            col_constraint_value = 0.9994
        else:
            col_constraint_value = 0.999
            
    # column constraint : close to 1, degree-1 bus set as 1
    if col_choice == 0:
        col_constraint = 10*np.ones((case.no_bus-1,)) ## no column constraint
        # col_constraint = np.inf
    else:
        degree_one = np.array([8]) - 1 ## buses that are not in any loop
        degree_one_posi = []
        for i in degree_one:
            if i < case.ref_index:
                degree_one_posi.append(i)
            elif i > case.ref_index:
                degree_one_posi.append(i-1)
        
        col_constraint = col_constraint_value*np.ones((case.no_bus-1,))
        col_constraint[degree_one_posi] = 1
        # col_constraint = 0.9995*case.no_bus
    
    # ac attack ratio
    att_state_ratio = 0.5

elif grid_choice == 2:
    """
    case57
    """

    mpc = case57()

    # decrease the load level
    #mpc['bus'][:,2] = mpc['bus'][:,2]*0.4  # active
    #mpc['bus'][:,3] = mpc['bus'][:,3]*0.4  # reactive

    # identify and remove the repeated branches
    # branch_set = []
    # repeat_branch = []
    # for i in range(mpc['branch'].shape[0]):
    #     new_branch_set = set([int(mpc['branch'][i,0]), int(mpc['branch'][i,1])])
    #     if new_branch_set in branch_set:
    #         repeat_branch.append(i)
        
    #     branch_set.append(new_branch_set)

    # mpc['branch'] = np.delete(mpc['branch'],repeat_branch,axis = 0)

    gencost = 200 * mpc["gen"].shape[0]
    genlimit_max = mpc["gen"][:,8]
    genlimit_min = mpc["gen"][:,9]
    flowlimit = 200*np.ones(mpc['branch'].shape[0])

    case = dc_grid(mpc, gencost, genlimit_max, genlimit_min, flowlimit)
    
    # test ac-opf
    result = opf(mpc)
    PI = case.Cg@result['gen'][:,1] - result['bus'][:,2]
    PF = result['branch'][:,13]
    QI = case.Cg@result['gen'][:,2] - result['bus'][:,3]
    QF = result['branch'][:,14]
    
    z = np.concatenate([PI, PF, QI, QF], axis = -1)/case.baseMVA  ## in p.u.
    
    R = np.diag(np.ones((case.no_measure,))) * sigma**2           ## measurement noise covariance matrix
    R_ = np.diag(np.ones(case.no_measure_)) * sigma**2            ## for the jacobian setting 
    
    #sigma_new = (np.abs(z) + 0.01)*0.5
    #sigma_new = np.max(np.abs(z))*0.01
    #R = np.diag(sigma_new**2)
    #R_ = R[case.no_bus:case.no_bus+case.no_branch, case.no_bus:case.no_bus+case.no_branch]
    
    case.prepare_BDD_variable(alpha, R, R_)
    name = "case57"

    col_constraint_value = 0.999995  ## can be set as different values

    if dfacts_choise == 0:                                ## start from 0
        dfacts_index = np.arange(case.no_branch)          ## perturb all the lines
        # k_min = 34                                         ## The value of minimum k that can be achieved for this D-FACTS placement
        k_min = 32
    elif dfacts_choise == 1:
        dfacts_index = np.array([2,9,12,16,18,19,20,21,23,26,27,29,31,33,36,38,40,42,45,48,50,54,58,60,62,66,68,70,73,74])-1    ## minimum number full rank covering all buses
        k_min = 34
    elif dfacts_choise == 2:
        dfacts_index = np.array([2,9,13,16,19,20,21,26,29,31,33,36,38,40,42,45,47,53,55,56,58,60,62,66,68,69,71,74])-1    ## minimum number
        k_min = 37

    # column constraint : close to 1, degree-1 bus set as 1
        
    if col_choice == 0:
        col_constraint = 10*np.ones((case.no_bus-1,)) ## no column constraint
        # col_constraint = np.inf
    else:
        degree_one = np.array([33]) - 1 ## buses that are not in any loop
        degree_one_posi = []
        for i in degree_one:
            if i < case.ref_index:
                degree_one_posi.append(i)
            elif i > case.ref_index:
                degree_one_posi.append(i-1)
        
        col_constraint = col_constraint_value*np.ones((case.no_bus-1,))
        col_constraint[degree_one_posi] = 1
        # col_constraint = 0.9995*case.no_bus
    
    # ac attack ratio
    att_state_ratio = 0.35

# %% general settings
x_min_ratio = 0.05 ## |delta_x| >= x_min_ratio 

if ratio_choice == 0:
    ratio = 0.2
    x_max_ratio = 0.2 ## |delta x| <= x_max_ratio
elif ratio_choice == 1:
    ratio = 0.3
    x_max_ratio = 0.3 ## |delta x| <= x_max_ratio
elif ratio_choice == 2:
    ratio = 0.4
    x_max_ratio = 0.4 ## |delta x| <= x_max_ratio
elif ratio_choice == 3:
    ratio = 0.5 # the maximum in liturature
    x_max_ratio = 0.5 ## |delta x| <= x_max_ratio

elif ratio_choice == 10:
    # only for a test
    ratio = 0.1
    x_max_ratio = 0.1  ## |delta x| <= x_max_ratio
    x_min_ratio = 0.01 ## |delta_x| >= x_min_ratio 

# Determine the critical lambda with different detection probability for effective MTD
# lambda_c is the minimum value of the non centrality parameter to achieve to have 1-alpha detection probability
# consider 1-alpha, 1-2*alpha, 1-3*alpha, 1-4*alpha as detection probability

probability_scale = [1,2,3,4]
lambda_c_eff = case.lambda_eff(alpha)

# the range of the proposed algorithm
x_max = copy.deepcopy(case.x)*1.0001
x_min = copy.deepcopy(case.x)*0.9999
x_max[dfacts_index] = x_max[dfacts_index]*(1+ratio)
x_min[dfacts_index] = x_min[dfacts_index]*(1-ratio)

#b_max, b_min = b_range(case.r,case.x,x_max_ratio,dfacts_index)
#b_max = -1/x_max
#b_min = -1/x_min

# assert np.all(b_max >= b_min)

# %%
'''
Name specification
'''
name_suffix = f"choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}"