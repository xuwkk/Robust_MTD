

"""
settings to the simultation
"""


"""
ONLY CHANGE HERE
"""

grid_choice = 2         ## 0:case6, 1:case14, 2:case57
ratio_choice = 3        ## the level of D-FACTS perturbation ratio ---- 0 : pm 0.2 --- 1 : pm 0.3 --- 2 : pm 0.4 --- 3 : pm 0.5
dfacts_choise = 1       ## the d-facts perturbation ----- 0 : perturb all lines --- 1 : minimum full rank --- 2 : minimum 
col_choice = 1          ## the on/off of the column angle constraint ---- 0 : no column constraint 1 : with column constraint

max_choice = 1

if dfacts_choise != 0:
    # if the dfacts is not fully located, we don't run the max mtd algorithm
    max_choice = 0

"""
DONT CHANGE BELOW
"""

attack_strength_under_test = [5, 7, 10, 15, 20]   
att_state_no = 1        ## number of single state attack position

# measurement noise and detection confidence
sigma = 0.01            ## assume 1% std of 1p.u. noise
alpha = 0.05            ## FPR confidence (for the BDD)

if grid_choice == 0:
    # complete case
    dfacts_choise = 0
    col_choice = 0
    
    # mtd
    max_no = 10
    robust_no = 50
    
    # attack-dc
    ac_opf_no = 50          ## the number of ac opf
    test_no = 100
    
    # attack-ac
    ac_opf_no_ac = 50
    ac_random_mtd_no = 20


if grid_choice == 1:
    # complete case
    
    # mtd
    max_no = 10
    robust_no = 20
    
    # attack-dc
    ac_opf_no = 50          ## the number of ac opf
    test_no = 100
    
    # attack-ac
    ac_opf_no_ac = 50
    ac_random_mtd_no = 20

if grid_choice == 2:
    # complete case
    
    # mtd
    max_no = 3
    robust_no = 20
    
    # attack-dc
    ac_opf_no = 50          ## the number of ac opf
    test_no = 100
    
    # attack-ac
    ac_opf_no_ac = 50
    ac_random_mtd_no = 20
    
