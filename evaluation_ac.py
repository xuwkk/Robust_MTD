
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

# %%

ATTACK_POSI_ROBUST = np.zeros(len(attack_strength_under_test)+1,)
TP_RANDOM_ROBUST = np.zeros(len(attack_strength_under_test)+1,)

RESIDUAL_ATTACKER_ROBUST = [[],[],[],[],[],[]]
RESIDUAL_ROBUST = [[],[],[],[],[],[]]
MAG_CHANGE = []
ANG_CHANGE = []

COST_pre = []
COST_after = []

for i in range(ac_opf_no_ac):

    print(f"the {i}th ac-opf")
    
    # generate data: obtain the state and jacobian matrix on the original reactance
    active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result = generate_data(case)
    COST_pre.append(result['f'])
    
    #if name == 'case57':
    #    v_mag_est, v_ang_est = case.flat_state(result)
    
    # robust mtd algorithm
    if name == 'case6':
        b_mtd, x_mtd, time, loss = run_complete(case, v_mag_est, v_ang_est, x_max, x_min, robust_no)
    else:
        b_mtd, x_mtd, time, loss = run_incomplete(case, v_mag_est, v_ang_est, x_max, x_min, col_constraint, k_min, robust_no)
        
    _,_,_,Jr_mtd_N = case.jacobian(v_mag_est, v_ang_est, b_mtd)
    
    # construct the new grid
    mpc_new = deepcopy(mpc)
    mpc_new['branch'][:,3] = x_mtd
    case_new = dc_grid(mpc_new, gencost, genlimit_max, genlimit_min, flowlimit)
    case_new.prepare_BDD_variable(alpha, R, R_)
    
    # run the opf on the new reactance
    result_new, z_new, z_noise_new = case_new.ac_opf(active_power, reactive_power)
    v_mag_est_new, v_ang_est_new = result_new['bus'][:,7], result_new['bus'][:,8]*np.pi/180 # find the state
    
    COST_after.append(result_new['f'])
    MAG_CHANGE.append((v_mag_est_new - v_mag_est)/v_mag_est_new)
    ANG_CHANGE.append((v_ang_est_new - v_ang_est)/v_ang_est_new)
    
    # hiddenness 
    residual_attacker, z_est_attacker = case.find_lambda_residual_ac(case.x, z_noise_new, result_new)     # on the original reactance
    RESIDUAL_ATTACKER_ROBUST.append(residual_attacker)
    
    
    # attack and detection
    for j in range(test_no):
        # generate attack using post-mtd measurement and ORIGINAL reactance
        z_noise_a = case.random_ac_attack(v_mag_est_new, v_ang_est_new, z_noise_new, att_state_ratio)  # using post-mtd measurement and ORIGINAL reactance

        residual, z_est_a = case_new.find_lambda_residual_ac(x_mtd, z_noise_a, result_new)
        
        
        attack_strength = norm(z_noise_a - z_noise_new, 2)/np.sqrt(np.sum(case.R))

        posi = find_posi(attack_strength)
        RESIDUAL_ROBUST[posi].append(residual)
        
        ATTACK_POSI_ROBUST[posi] += 1
        if residual >= case_new.BDD_threshold:
            TP_RANDOM_ROBUST[posi] += 1

# %%
path = f'simulation_data/{name}/ac'
print(TP_RANDOM_ROBUST/ATTACK_POSI_ROBUST)
np.save(f'{path}/{name_suffix}_TP_RANDOM_ROBUST.npy', TP_RANDOM_ROBUST/ATTACK_POSI_ROBUST)

# %% random mtd

ATTACK_POSI_RANDOM = np.zeros(len(attack_strength_under_test)+1,)
TP_RANDOM_RANDOM = np.zeros(len(attack_strength_under_test)+1,)
RESIDUAL_RANDOM = [[],[],[],[],[],[],[]]

for i in range(ac_opf_no_ac):
    print(f"the {i}th ac-opf")
    # loop for ac-opf on original reactance
    # generate data: obtain the state and jacobian matrix on the original reactance
    active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result = generate_data(case)
    
    for j in range(ac_random_mtd_no):

        # loop for random mtd
        # random mtd: is moved into the lower level loop
        x_mtd = case.random_mtd(x_max_ratio,x_min_ratio,dfacts_index)

        # construct new grid
        mpc_new = deepcopy(mpc)
        mpc_new['branch'][:,3] = x_mtd
        case_new = dc_grid(mpc_new, gencost, genlimit_max, genlimit_min, flowlimit)
        case_new.prepare_BDD_variable(alpha, R, R_)

        # run ac-opf (again)
        result_new, z_new, z_noise_new = case_new.ac_opf(active_power, reactive_power)
        v_mag_est_new, v_ang_est_new = result_new['bus'][:,7], result_new['bus'][:,8]*np.pi/180
        
        # attack and detection
        for k in range(int(test_no)):
            
            # loop for attack
            # generate attack using post-mtd measurement and ORIGINAL reactance
            z_noise_a = case.random_ac_attack(v_mag_est_new, v_ang_est_new, z_noise_new, att_state_ratio)  
            
            # detection
            residual, z_est_a = case_new.find_lambda_residual_ac(x_mtd,z_noise_a,result_new)  # on the new configuration
            
            
            attack_strength = norm(z_noise_a - z_noise_new, 2)/np.sqrt(np.sum(case.R))
            posi = find_posi(attack_strength)
            
            RESIDUAL_RANDOM[posi].append(residual)
            
            ATTACK_POSI_RANDOM[posi] += 1
            if residual > case.BDD_threshold:
                TP_RANDOM_RANDOM[posi] += 1

# %%
print(TP_RANDOM_ROBUST/ATTACK_POSI_ROBUST)
print(TP_RANDOM_RANDOM/ATTACK_POSI_RANDOM)
print(COST_pre)
print(COST_after)

# %% save
os.makedirs(os.path.join(f'simulation_data/{name}/ac'), exist_ok=True)
path = f'simulation_data/{name}/ac'

np.save(f'{path}/{name_suffix}_TP_RANDOM_ROBUST.npy', TP_RANDOM_ROBUST/ATTACK_POSI_ROBUST)
np.save(f'{path}/{name_suffix}_TP_RANDOM_RANDOM.npy', TP_RANDOM_RANDOM/ATTACK_POSI_RANDOM)
np.save(f'{path}/{name_suffix}_COST_pre.npy', COST_pre)
np.save(f'{path}/{name_suffix}_COST_after.npy', COST_after)
np.save(f'{path}/{name_suffix}_RESIDUAL_ATTACKER_ROBUST.npy', RESIDUAL_ATTACKER_ROBUST)

np.save(f'{path}/{name_suffix}_RESIDUAL_ROBUST.npy', RESIDUAL_ROBUST)
np.save(f'{path}/{name_suffix}_RESIDUAL_RANDOM.npy', RESIDUAL_RANDOM)
np.save(f'{path}/{name_suffix}_MAG_CHANGE.npy', MAG_CHANGE)
np.save(f'{path}/{name_suffix}_ANG_CHANGE.npy', ANG_CHANGE)

