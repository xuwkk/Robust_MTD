import numpy as np
from copy import deepcopy

def b_range(r,x,x_ratio,dfacts_index):
    b_min = deepcopy(x)
    b_max = deepcopy(x)
    x_max = deepcopy(x)*(1+1e-6)
    x_min = deepcopy(x)*(1-1e-6)
    x_max[dfacts_index] = (1+x_ratio)*x[dfacts_index]
    x_min[dfacts_index] = (1-x_ratio)*x[dfacts_index]
    
    for i in range(len(b_max)):
        
        if x_max[i] >= r[i] and x_min[i] < r[i]:
            b_min[i] = -r[i]/(r[i]**2+r[i]**2)
            b_max[i] = np.max([-x_max[i]/(x_max[i]**2+r[i]**2),-x_min[i]/(x_min[i]**2+r[i]**2)])
        
        else:
            b_max[i] = np.max([-x_max[i]/(x_max[i]**2+r[i]**2),-x_min[i]/(x_min[i]**2+r[i]**2)])
            b_min[i] = np.min([-x_max[i]/(x_max[i]**2+r[i]**2),-x_min[i]/(x_min[i]**2+r[i]**2)])
        
    
    return b_max, b_min

def x_to_b(case,x):
    return -x/(case.r**2+x**2)

def find_posi(rho):
    # FIND THE ATTACK STRENGHT OF AC ATTACK
    if rho >= 5 and rho<25:
        if rho < 7:
            posi = 0
        elif rho < 10:
            posi = 1
        elif rho < 15:
            posi = 2
        elif rho < 20:
            posi = 3
        else:
            posi = 4
    else:
        posi = 5
    
    return posi

