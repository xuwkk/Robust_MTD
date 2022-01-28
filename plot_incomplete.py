# -*- coding: utf-8 -*-

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import ncx2, chi2

# %% SETTINGS
dfacts_choise = 0
col_choice = 1
ratio_choice = 0
name = 'case14'

suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'

if name == 'case14':
    BDD_threshold = 58.124
elif name == 'case57':
    BDD_threshold = 192.700
    
rc = {"font.family" : "serif", 
      "mathtext.fontset" : "stix",
      }
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
font = {'size'   : 25}
plt.rc('font', **font)

attack_strength_under_test = [5,7,10,15,20]

x_axis_name = [str(i) for i in attack_strength_under_test]

# %% LOAD DATA
WORST_ROBUST = np.load(f'simulation_data/{name}/dc/{suffix}_WORST_ROBUST.npy', allow_pickle = True)
WORST_RANDOM = np.load(f'simulation_data/{name}/dc/{suffix}_WORST_RANDOM.npy', allow_pickle = True)

RANDOM_MAX = np.load(f'simulation_data/{name}/dc/{suffix}_RANDOM_MAX.npy', allow_pickle = True)
RANDOM_ROBUST = np.load(f'simulation_data/{name}/dc/{suffix}_RANDOM_ROBUST.npy', allow_pickle = True)
RANDOM_RANDOM = np.load(f'simulation_data/{name}/dc/{suffix}_RANDOM_RANDOM.npy', allow_pickle = True)

SINGLE_ROBUST = np.load(f'simulation_data/{name}/dc/{suffix}_SINGLE_ROBUST.npy', allow_pickle = True)

TIME_MAX = np.load(f'simulation_data/{name}/dc/{suffix}_TIME_MAX.npy', allow_pickle = True)
TIME_ROBUST = np.load(f'simulation_data/{name}/dc/{suffix}_TIME_ROBUST.npy', allow_pickle = True)

RANDOM_ROBUST_AC = np.load(f'simulation_data/{name}/ac/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)
RANDOM_RANDOM_AC = np.load(f'simulation_data/{name}/ac/{suffix}_TP_RANDOM_RANDOM.npy', allow_pickle = True)
RESIDUAL = np.load(f'simulation_data/{name}/ac/{suffix}_RESIDUAL_ATTACKER_ROBUST.npy', allow_pickle = True)

COST_PRE = np.load(f'simulation_data/{name}/ac/{suffix}_COST_pre.npy', allow_pickle = True)
COST_AFTER = np.load(f'simulation_data/{name}/ac/{suffix}_COST_after.npy', allow_pickle = True)

# residual
RESIDUAL_ROBUST = np.load(f'simulation_data/{name}/ac/{suffix}_RESIDUAL_ROBUST.npy', allow_pickle = True)
RESIDUAL_RANDOM = np.load(f'simulation_data/{name}/ac/{suffix}_RESIDUAL_RANDOM.npy', allow_pickle = True)
RESIDUAL_BDD = np.load(f'simulation_data/{name}/ac/{suffix}_RESIDUAL_BDD.npy', allow_pickle = True)

WORST_RANDOM_MEAN = np.mean(WORST_RANDOM,1)


# %% Visualization on the residual distribution
if name == 'case14':
    lambda_eff_robust = 21.830000000000002
    lambda_eff_random = [18.19, 5.93, 14.56, 7.14, 11.41, 16.57]
    x = np.arange(1,100)
    fig = plt.figure(figsize=(8, 6))
    cdf = chi2.cdf(x, 7)
    plt.plot(x, cdf, label = 'No MTD', color = plt.cm.tab10(1))
    cdf = ncx2.cdf(x, 7, lambda_eff_robust)
    plt.plot(x, cdf, label = 'Desired MTD', color = plt.cm.tab10(0))
    for i in range(len(lambda_eff_random)):
        cdf = ncx2.cdf(x, 7, lambda_eff_random[i])
        if i == 0:
            plt.plot(x, cdf, label = 'Max-Rank MTD', color = plt.cm.tab10(2))
        else:
            plt.plot(x, cdf, label = '_nolegend_', color = plt.cm.tab10(2))
    
    plt.vlines(x = 14.067, ymin = 0, ymax = 1, colors = 'r', linestyle = 'dashed', label='BDD Threshold')
    #plt.hlines(y = 0.05, xmin = 0, xmax = 100, colors = 'r', linestyles = 'dashdot', label = r'$\alpha$')
    plt.xlabel(r"Residual $\gamma$")
    plt.ylabel("c.d.f")
    plt.grid()
    plt.legend(loc = 7)
    plt.ylim([0,1.])
    plt.xlim([0,100])
    plt.show()
    fig.savefig(f"figure/{name}/{name}_dc_residual_illustration.pdf", bbox_inches="tight", dpi=400)

# with attack strength = 20


# %% dc
fig = plt.figure(figsize=(8, 6))
width = 0.1
plt.plot(WORST_ROBUST, color = plt.cm.tab10(0), linestyle = '-', marker = '^', lw = 2)
plt.plot(WORST_RANDOM_MEAN, color = plt.cm.tab10(1), linestyle = '-', marker = 'o', lw = 2)
for i in range(len(WORST_RANDOM)):
    plt.scatter(x = i*np.ones(len(WORST_RANDOM[0,:500])) + (np.random.rand(len(WORST_RANDOM[0,:500]))*width-width/2.), 
                y = WORST_RANDOM[i,:500], color = plt.cm.tab10(2), s = 35, marker = "x")

plt.xticks(range(len(attack_strength_under_test)), x_axis_name)
plt.xlabel(r"Attack strength $\rho$")
plt.ylabel("Attack detection probability")
plt.grid()
plt.legend([r'Robust MTD',"Max-Rank MTD (Mean)", "Max-Rank MTD"], labelspacing=0.1, loc = 2)
plt.ylim([0,1.1])
plt.show()
fig.savefig(f"figure/{name}/{name}_dc_worst.pdf", bbox_inches="tight", dpi=400)


fig = plt.figure(figsize=(8, 6))
plt.plot(RANDOM_MAX, label = 'MAX MTD', color = plt.cm.tab10(1), linestyle = '-', marker = 'o', lw = 2)
plt.plot(RANDOM_ROBUST, label = 'ROBUST MTD', color = plt.cm.tab10(0), linestyle = '-', marker = '^', lw = 2)
plt.plot(RANDOM_RANDOM, label = 'Max-Rank MTD', color = plt.cm.tab10(2), linestyle = '-', marker = '*', lw = 2)
plt.xlabel(r"Attack strength $\rho$")
plt.ylabel("Attack detection probability")
plt.xticks(range(len(attack_strength_under_test)), x_axis_name)
plt.grid()
plt.legend(['Max MTD', 'Robust MTD', "Max-Rank MTD"], labelspacing=0.1, loc = 2)
plt.ylim([0,1.1])
plt.show()
fig.savefig(f"figure/{name}/{name}_dc_random.pdf", bbox_inches="tight", dpi=400)

print(f'TIME_ROBUST: {np.mean(TIME_ROBUST)}')
print(f'TIME_MAX: {np.mean(TIME_MAX)}')

# %% AC
plt.figure(figsize=(8, 6))
plt.plot(RANDOM_ROBUST_AC[:-1], color = plt.cm.tab10(0), linestyle = '-', marker = '^', lw = 2)
plt.plot(RANDOM_RANDOM_AC[:-1], color = plt.cm.tab10(2), linestyle = '-', marker = '*', lw = 2)
plt.xlabel(r"Attack strength $\rho$")
plt.ylabel("Attack detection probability")
plt.xticks(range(len(attack_strength_under_test)), x_axis_name)
plt.grid()
plt.legend(['Robust MTD', "Max-Rank MTD"], labelspacing=0.1, loc = 2)
plt.ylim([0,1.1])
plt.plot()
fig.savefig(f"figure/{name}/{name}_ac_random.pdf", bbox_inches="tight", dpi=400)

print(f'Attacker residual: {RESIDUAL}')
print(f'cost change: {(COST_AFTER - COST_PRE)/COST_PRE}')

# %%
kwargs = {'cumulative': True, 'linewidth' : 2}

for i in range(len(RESIDUAL_ROBUST)-1):
    fig = plt.figure(figsize=(8, 6))
    seaborn.distplot(RESIDUAL_BDD[i], kde = True, hist = False, kde_kws=kwargs, color = plt.cm.tab10(1))
    seaborn.distplot(RESIDUAL_ROBUST[i], kde = True, hist = False, kde_kws=kwargs, color = plt.cm.tab10(0))
    seaborn.distplot(RESIDUAL_RANDOM[i], kde = True, hist = False, kde_kws=kwargs, color = plt.cm.tab10(2))
    plt.vlines(x = BDD_threshold, ymin = 0, ymax = 1, colors = 'r', linestyle = 'dashed')
    plt.xlabel(r"Residual $\gamma$")
    plt.ylabel("c.d.f")
    plt.grid()
    plt.legend(['No MTD','Robust MTD', "Max-Rank MTD", 'BDD Threshold'], labelspacing=0.1, loc = 4)
    plt.ylim([0,1.])
    if name == 'case14':
        plt.xlim([0,250])
    elif name == 'case57':
        plt.xlim([100,700])
    
    
    plt.show()
    fig.savefig(f"figure/{name}/{name}_ac_residual_{i}.pdf", bbox_inches="tight", dpi=400)
    

# %% single state attack
if name == 'case14':
    width = 0.4
    x = np.arange(14-1)
    dfacts_choise = 0
    col_choice = 0
    ratio_choice = 0
    suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'
    SINGLE_ROBUST_ = np.load(f'simulation_data/{name}/dc/{suffix}_SINGLE_ROBUST.npy', allow_pickle = True)

    fig = plt.figure(figsize=(8, 6))
    plt.bar(x - width/2, SINGLE_ROBUST[2], width, label = 'with Principle 2')
    plt.bar(x + width/2, SINGLE_ROBUST_[2], width, label = 'without Principle 2')
    plt.xticks(range(14-1), np.arange(2,14+1))
    plt.xlabel("Attack Position/Bus")
    plt.ylabel("Attack detection probability")
    plt.grid()
    plt.ylim([0,1.1])
    plt.legend(loc = 6, labelspacing=0.1)
    plt.show()
    fig.savefig(f"figure/{name}/{name}_single.pdf", bbox_inches="tight", dpi=400)
    
# %% different d-facts choices
if name == 'case14':
    dfacts_choise = 1
    col_choice = 1
    ratio_choice = 0
    suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'
    RANDOM_ROBUST_0 = np.load(f'simulation_data/{name}/ac/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)

    dfacts_choise = 1
    col_choice = 1
    ratio_choice = 1
    suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'
    RANDOM_ROBUST_1 = np.load(f'simulation_data/{name}/ac/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)

    dfacts_choise = 1
    col_choice = 1
    ratio_choice = 2
    suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'
    RANDOM_ROBUST_2 = np.load(f'simulation_data/{name}/ac/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)

    dfacts_choise = 1
    col_choice = 1
    ratio_choice = 3
    suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'
    RANDOM_ROBUST_3 = np.load(f'simulation_data/{name}/ac/{suffix}_TP_RANDOM_ROBUST.npy', allow_pickle = True)
    fig = plt.figure(figsize=(8, 6))
    plt.plot(RANDOM_ROBUST_AC[:-1], linestyle = '--', marker = 'o', lw = 2, color = "grey", label = r"$\tau = 0.2$ (all)")
    plt.plot(RANDOM_ROBUST_0[:-1], linestyle = '-', marker = 'o', lw = 2, color = plt.cm.tab10(0), label = r"$\tau = 0.2$ (part)")
    plt.plot(RANDOM_ROBUST_1[:-1], linestyle = '-', marker = 'o', lw = 2, color = plt.cm.tab10(1), label = r"$\tau = 0.3$ (part)")
    plt.plot(RANDOM_ROBUST_2[:-1], linestyle = '-', marker = 'o', lw = 2, color = plt.cm.tab10(2), label = r"$\tau = 0.4$ (part)")
    plt.plot(RANDOM_ROBUST_3[:-1], linestyle = '-', marker = 'o', lw = 2, color = plt.cm.tab10(3), label = r"$\tau = 0.5$ (part)")
    plt.legend(labelspacing = 0.1)
    x_axis_name_ac = [r'$[5,7)$', r'$[7,10)$',r'$[10,15)$',r'$[15,20)$',r'$[20,25)$']
    plt.xticks(range(len(attack_strength_under_test)), x_axis_name_ac)
    plt.xticks(range(5), x_axis_name_ac)
    plt.xlabel(r"Attack strength $\rho$")
    plt.ylabel("Attack detection probability")
    plt.grid()
    plt.ylim([0,1.1])
    #plt.xlim([0,4])
    plt.show()
    fig.savefig(f"figure/{name}/{name}_compare.pdf", bbox_inches="tight", dpi=400)
# %%

# %%
