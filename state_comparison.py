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

MAG_CHANGE = []
ANG_CHANGE = []

COST_pre = []
COST_after = []

for i in range(ac_opf_no_ac):

    print(f"the {i}th ac-opf")
    
    # generate data: obtain the state and jacobian matrix on the original reactance
    active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result = generate_data(case)
    
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

    MAG_CHANGE.append((v_mag_est_new - v_mag_est)/v_mag_est_new)
    ANG_CHANGE.append((v_ang_est_new - v_ang_est)/v_ang_est_new)

# %% save
os.makedirs(os.path.join(f'simulation_data/{name}/ac'), exist_ok=True)
path = f'simulation_data/{name}/ac'
np.save(f'{path}/{name_suffix}_MAG_CHANGE.npy', MAG_CHANGE)
np.save(f'{path}/{name_suffix}_ANG_CHANGE.npy', ANG_CHANGE)
# %%
