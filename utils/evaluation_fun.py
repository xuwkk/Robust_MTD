# -*- coding: utf-8 -*-
import numpy as np
def generate_data(case):
    # run ac-opf and retrive states
    active_power = case.mpc['bus'][:,2] * (0.8 + 0.4*np.random.rand(case.mpc['bus'].shape[0]))
    reactive_power = case.mpc['bus'][:,3] * (0.8 + 0.4*np.random.rand(case.mpc['bus'].shape[0]))
    result, z, z_noise = case.ac_opf(active_power, reactive_power)
    v_mag_est, v_ang_est = result['bus'][:,7], result['bus'][:,8]*np.pi/180

    # test jacobian
    _,_,_,Jr_N = case.jacobian(v_mag_est, v_ang_est, case.b)  # original known by the attacker
    Jr_N_check = case.jacobian_check(v_mag_est, v_ang_est)
    
    if case.no_bus == 57:
        pass
    else:
        assert np.max(np.abs(Jr_N_check - Jr_N)) <= 1e-6
    
    return active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result