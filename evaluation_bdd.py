# -*- coding: utf-8 -*-

# %%
from utils.initialization import *
from utils.mtd_incomplete import run_incomplete
from utils.mtd_complete import run_complete
from utils.utils import x_to_b, find_posi
from copy import deepcopy
from utils.grid_fun import dc_grid
from numpy.linalg import norm
import matplotlib.pyplot as plt
from utils.evaluation_fun import generate_data
import os

print(name)
print(name_suffix)
RESIDUAL_BDD = [[],[],[],[],[],[]]

for i in range(ac_opf_no_ac):

    print(f"the {i}th ac-opf")
    
    # generate data: obtain the state and jacobian matrix on the original reactance
    active_power = case.mpc['bus'][:,2] * (0.8 + 0.4*np.random.rand(case.mpc['bus'].shape[0]))
    reactive_power = case.mpc['bus'][:,3] * (0.8 + 0.4*np.random.rand(case.mpc['bus'].shape[0]))
    result, z, z_noise = case.ac_opf(active_power, reactive_power)
    v_mag_est, v_ang_est = result['bus'][:,7], result['bus'][:,8]*np.pi/180
    
    # attack and detection
    for j in range(test_no):
        # generate attack using post-mtd measurement and ORIGINAL reactance
        z_noise_a = case.random_ac_attack(v_mag_est, v_ang_est, z_noise, att_state_ratio)  

        residual, z_est_a = case.find_lambda_residual_ac(case.x, z_noise_a, result)
        
        attack_strength = norm(z_noise_a - z_noise, 2)/np.sqrt(np.sum(case.R))

        posi = find_posi(attack_strength)
        RESIDUAL_BDD[posi].append(residual)

# %%
path = f'simulation_data/{name}/ac'
np.save(f'{path}/{name_suffix}_RESIDUAL_BDD.npy', RESIDUAL_BDD)