
# %%
from utils.initialization import *
from utils.mtd_incomplete import run_incomplete
from utils.mtd_complete import run_complete
from utils.utils import x_to_b
from utils.evaluation_fun import generate_data
import os

print(name)
print(name_suffix)
#print(col_constraint)
#print(case.R_)

# %%
# random attack
TP_RANDOM_ROBUST = np.zeros((len(attack_strength_under_test),))
TP_RANDOM_BDD = np.zeros((len(attack_strength_under_test),))
TP_RANDOM_RANDOM = np.zeros((len(attack_strength_under_test),))
TP_RANDOM_MAX = np.zeros((len(attack_strength_under_test),))


# worst attack
PROB_WORST_ROBUST = np.zeros((len(attack_strength_under_test),))
PROB_WORST_RANDOM = [[],[],[],[],[]]

# single attack
SINGLE_POSI = np.zeros((len(attack_strength_under_test), case.no_bus-1,))
TP_SINGLE_ROBUST = np.zeros((len(attack_strength_under_test), case.no_bus-1))

# time
TIME_ROBUST = []
TIME_MAX = []

for i in range(ac_opf_no):
    
    print(f"the {i}th ac-opf")
    
    # generate data: obtain the state and jacobian matrix on the original reactance
    active_power, reactive_power, v_mag_est, v_ang_est, Jr_N, result = generate_data(case)

    # robust mtd algorithm
    if name == 'case6':
        b_mtd, x_mtd, time, loss = run_complete(case, v_mag_est, v_ang_est, x_max, x_min, robust_no)
    else:
        b_mtd, x_mtd, time, loss = run_incomplete(case, v_mag_est, v_ang_est, x_max, x_min, col_constraint, k_min, robust_no)
    
    Cr_N,V_N,AAr,Jr_mtd_N = case.jacobian(v_mag_est, v_ang_est, b_mtd)

    TIME_ROBUST.append(time)

    """
    RANDOM ATTACK
    """
    print('random_attack')
    for j in range(len(attack_strength_under_test)):
        print(f'the {j}th attack strength')
        for k in range(test_no):
            if k%100 == 0:
                print(k)
            a, a_noise = case.random_attack(Jr_N, attack_strength=attack_strength_under_test[j])
            assert np.abs(np.linalg.norm(a,2)/np.sqrt(np.sum(case.R_)) - attack_strength_under_test[j]) <= 1e-6

            # robust mtd
            output_robust = case.evaluation_metric(Jr_N, Jr_mtd_N, a, a_noise)
            TP_RANDOM_ROBUST[j] += output_robust['MTD']
            TP_RANDOM_BDD[j] += output_robust['BDD']
            
            if max_choice == 1:
                # random mtd
                x_mtd_random = case.random_mtd(x_max_ratio=x_max_ratio, x_min_ratio=x_min_ratio, dfacts_index=dfacts_index)
                b_mtd_random = x_to_b(case, x_mtd_random)
                assert np.all(x_mtd_random <= x_max)
                assert np.all(x_mtd_random >= x_min)
                _,_,_,Jr_mtd_random_N = case.jacobian(v_mag_est, v_ang_est, b_mtd_random)
                output_random = case.evaluation_metric(Jr_N, Jr_mtd_random_N, a, a_noise)
                TP_RANDOM_RANDOM[j] += output_random['MTD']
                
                
                # max mtd
                b_mtd_max, time_max = case.max_mtd(case, v_mag_est, v_ang_est, x_max, x_min, a, max_no)
                _,_,_,Jr_mtd_max_N = case.jacobian(v_mag_est, v_ang_est, b_mtd_max)
                output = case.evaluation_metric(Jr_N, Jr_mtd_max_N, a, a_noise)
                TP_RANDOM_MAX[j] += output['MTD']
                TIME_MAX.append(time_max)
    
    
    if max_choice == 1:
        """
        WORST ATTACK
        """
        print('worst attack')
        
        for j in range(len(attack_strength_under_test)):  
            worst_point_robust, largest_non_one_singular_value_robust = case.worst_point(Jr_N, Jr_mtd_N, k_min)
            assert np.abs(np.min(loss) - largest_non_one_singular_value_robust) <= 1e-6  
            a_robust, a_robust_noise = case.worst_attack(attack_strength=attack_strength_under_test[j], worst_point=worst_point_robust)
            
            # robust mtd
            output_robust = case.evaluation_metric(Jr_N, Jr_mtd_N, a_robust, a_robust_noise)
            PROB_WORST_ROBUST[j] += output_robust['detection_prob']
            lambda_eff_ = np.linalg.norm(case.R_inv_12_@a_robust, 2)**2*np.sin(np.arccos(largest_non_one_singular_value_robust))**2
            assert np.abs(lambda_eff_ - output_robust['lambda_eff']) <= 1e-6
    
            # random mtd
            for k in range(test_no):
                x_mtd_random = case.random_mtd(x_max_ratio=x_max_ratio, x_min_ratio=x_min_ratio, dfacts_index=dfacts_index)
                assert np.all(x_mtd_random <= x_max)
                assert np.all(x_mtd_random >= x_min)
                b_mtd_random = x_to_b(case, x_mtd_random)
                _,_,_,Jr_mtd_random_N = case.jacobian(v_mag_est, v_ang_est, b_mtd_random)
                
                # find the worst point for this mtd configuration
                worst_point_random, largest_non_one_singular_value_random = case.worst_point(Jr_N, Jr_mtd_random_N, k_min)
                a_random, a_random_noise = case.worst_attack(attack_strength=attack_strength_under_test[j], worst_point=worst_point_random)
                output_random = case.evaluation_metric(Jr_N, Jr_mtd_random_N, a_random, a_random_noise)
                PROB_WORST_RANDOM[j].append(output_random['detection_prob'])
    
        """
        SINGLE-STATE ATTACK
        """
        print('single-state attack')
        for j in range(len(attack_strength_under_test)):
            print(f'the {j}th attack strength')
            for k in range(test_no):
                a, a_noise, c = case.fix_no_attack(Jr_N = Jr_N, attack_strength = attack_strength_under_test[j], att_state_no = att_state_no)
                # robust mtd
                output_robust = case.evaluation_metric(Jr_N, Jr_mtd_N, a, a_noise)
                posi = np.where(c != 0)[0]
                SINGLE_POSI[j,posi] += 1
                TP_SINGLE_ROBUST[j,posi] += output_robust['MTD']


# %%
print(f'RANDOM_ROBUST: {TP_RANDOM_ROBUST/(ac_opf_no*test_no)}')
print(f'RANDOM_BDD: {TP_RANDOM_BDD/(ac_opf_no*test_no)}')
print(f'RANDOM_RANDOM: {TP_RANDOM_RANDOM/(ac_opf_no*test_no)}')
print(f'RANDOM_MAX: {TP_RANDOM_MAX/(ac_opf_no*test_no)}')
print(f'WORST_ROBUST: {PROB_WORST_ROBUST/ac_opf_no}')
print(f'WORST_RANDOM: {np.mean(PROB_WORST_RANDOM,1)}')
print(f'SINGLE_ROBUST: {TP_SINGLE_ROBUST/SINGLE_POSI}')

print(f'TIME_ROBUST: {np.mean(TIME_ROBUST)}')
print(f'TIME_MAX: {np.mean(TIME_MAX)}')

# %% save

os.makedirs(os.path.join(f'simulation_data/{name}/dc'), exist_ok=True)
path = f'simulation_data/{name}/dc'

np.save(f'{path}/{name_suffix}_RANDOM_ROBUST.npy', TP_RANDOM_ROBUST/(ac_opf_no*test_no))
if max_choice == 1:
    np.save(f'{path}/{name_suffix}_RANDOM_BDD.npy', TP_RANDOM_BDD/(ac_opf_no*test_no))
    np.save(f'{path}/{name_suffix}_RANDOM_RANDOM.npy', TP_RANDOM_RANDOM/(ac_opf_no*test_no))


    np.save(f'{path}/{name_suffix}_RANDOM_MAX.npy', TP_RANDOM_MAX/(ac_opf_no*test_no))
    
    np.save(f'{path}/{name_suffix}_WORST_ROBUST.npy', PROB_WORST_ROBUST/ac_opf_no)
    np.save(f'{path}/{name_suffix}_WORST_RANDOM.npy', PROB_WORST_RANDOM)
    np.save(f'{path}/{name_suffix}_SINGLE_ROBUST.npy', TP_SINGLE_ROBUST/SINGLE_POSI)
    np.save(f'{path}/{name_suffix}_TIME_ROBUST.npy', TIME_ROBUST)
    np.save(f'{path}/{name_suffix}_TIME_MAX.npy', TIME_MAX)
