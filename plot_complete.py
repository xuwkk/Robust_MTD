# -*- coding: utf-8 -*-

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn

# %% SETTINGS
dfacts_choise = 0
col_choice = 0
ratio_choice = 0
name = 'case6'

suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'

BDD_threshold = 36.415
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

# %% single state attack
width = 0.2
x = np.arange(6-1)
dfacts_choise = 0
col_choice = 0
ratio_choice = 0
suffix = f'choice_{dfacts_choise}_ratio_{ratio_choice}_column_{col_choice}'
SINGLE_ROBUST_ = np.load(f'simulation_data/{name}/dc/{suffix}_SINGLE_ROBUST.npy', allow_pickle = True)

fig = plt.figure(figsize=(8, 6))
plt.bar(x - width/2, SINGLE_ROBUST[2], width, label = 'with Principle 2')
plt.bar(x + width/2, SINGLE_ROBUST_[2], width, label = 'without Principle 2')
plt.xticks(range(6-1), np.arange(2,6+1))
plt.xlabel("Attack Position/Bus")
plt.ylabel("Attack detection probability")
plt.grid()
plt.ylim([0,1.1])
plt.legend(loc = 6, labelspacing=0.1)
plt.show()
fig.savefig(f"figure/{name}/{name}_single.pdf", bbox_inches="tight", dpi=400)

# %% AC
fig = plt.figure(figsize=(8, 6))
plt.plot(RANDOM_ROBUST_AC[:-1], color = plt.cm.tab10(0), linestyle = '-', marker = '^', lw = 2)
plt.plot(RANDOM_RANDOM_AC[:-1], color = plt.cm.tab10(2), linestyle = '-', marker = '*', lw = 2)
plt.xlabel(r"Attack strength $\rho$")
plt.ylabel("Attack detection probability")
plt.grid()
plt.legend(['Robust MTD', "Max-Rank MTD"], labelspacing=0.1, loc = 2)
plt.ylim([0,1.1])
plt.xticks(range(len(attack_strength_under_test)), x_axis_name)
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
    plt.legend(['No MTD', 'Robust MTD', "Max-Rank MTD", 'BDD Threshold'], labelspacing=0.1, loc = 4)
    plt.ylim([0,1.])
    plt.xlim([0,175])
    plt.show()
    fig.savefig(f"figure/{name}/{name}_ac_residual_{i}.pdf", bbox_inches="tight", dpi=400)




# %% state change

MAG_CHANGE = np.load(f'simulation_data/{name}/ac/{suffix}_MAG_CHANGE.npy', allow_pickle = True)
ANG_CHANGE = np.load(f'simulation_data/{name}/ac/{suffix}_ANG_CHANGE.npy', allow_pickle = True)

print(f'MAG_CHANGE_MEAN: {np.mean(np.abs(MAG_CHANGE))}')
print(f'ANG_CHANGE_MEAN: {np.mean(np.abs(np.nan_to_num(ANG_CHANGE, nan=0)))}')
# %%
