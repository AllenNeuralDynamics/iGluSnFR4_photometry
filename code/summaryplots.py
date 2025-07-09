# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 05:13:09 2024

@author: kenta.hagihara
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scikit_posthocs as sp

AnalDir = r"/results/Analysis/"
SaveDir_Fig = r'/results/Fig_publication/'
#DAT 3 vs 4
Data1 = np.load(AnalDir + "751752Psth_G_RewardC.npy")
Data2 = np.load(AnalDir + "761754Psth_G_RewardC.npy")
Data3 = np.load(AnalDir + "759856Psth_G_RewardC.npy")
Data4 = np.load(AnalDir + "759857Psth_G_RewardC.npy")

#GAD 3 vs 4
Data5 = np.load(AnalDir + "734808Psth_G_RewardC.npy")
Data6 = np.load(AnalDir + "734809Psth_G_RewardC.npy")
Data7 = np.load(AnalDir + "734811Psth_G_RewardC.npy")
Data8 = np.load(AnalDir + "734812Psth_G_RewardC.npy")
Data9 = np.load(AnalDir + "754979Psth_G_RewardC.npy")

#GAD 2 vs 4
Data10 = np.load(AnalDir + "734805Psth_G_RewardC.npy")
Data11 = np.load(AnalDir + "734806Psth_G_RewardC.npy")
Data12 = np.load(AnalDir + "734813Psth_G_RewardC.npy")
Data13 = np.load(AnalDir + "734814Psth_G_RewardC.npy")
Data14 = np.load(AnalDir + "754977Psth_G_RewardC.npy")
#DAT 2 vs 4
Data15 = np.load(AnalDir + "762321Psth_G_RewardC.npy")
Data16 = np.load(AnalDir + "762327Psth_G_RewardC.npy")
Data17 = np.load(AnalDir + "763856Psth_G_RewardC.npy")
Data18 = np.load(AnalDir + "744066Psth_G_RewardC.npy")

window = np.arange(100, 120) 

D1_1 = np.mean(Data1[window,0,:])
D1_2 = np.mean(Data1[window,1,:])
D2_1 = np.mean(Data2[window,0,:])
D2_2 = np.mean(Data2[window,1,:])
D3_1 = np.mean(Data3[window,0,:])
D3_2 = np.mean(Data3[window,1,:])
D4_1 = np.mean(Data4[window,0,:])
D4_2 = np.mean(Data4[window,1,:])

D5_1 = np.mean(Data5[window,0,:])
D5_2 = np.mean(Data5[window,1,:])
D6_1 = np.mean(Data6[window,0,:])
D6_2 = np.mean(Data6[window,1,:])
D7_1 = np.mean(Data7[window,0,:])
D7_2 = np.mean(Data7[window,1,:])
D8_1 = np.mean(Data8[window,0,:])
D8_2 = np.mean(Data8[window,1,:])
D9_1 = np.mean(Data9[window,0,:])
D9_2 = np.mean(Data9[window,1,:])

D10_1 = np.mean(Data10[window,0,:])
D10_2 = np.mean(Data10[window,1,:])
D11_1 = np.mean(Data11[window,0,:])
D11_2 = np.mean(Data11[window,1,:])
D12_1 = np.mean(Data12[window,0,:])
D12_2 = np.mean(Data12[window,1,:])
D13_1 = np.mean(Data13[window,1,:]) #surgery L/R counter
D13_2 = np.mean(Data13[window,0,:])
D14_1 = np.mean(Data14[window,1,:]) #surgery L/R counter
D14_2 = np.mean(Data14[window,0,:])

D15_1 = np.mean(Data15[window,0,:])
D15_2 = np.mean(Data15[window,1,:])
D16_1 = np.mean(Data16[window,0,:])
D16_2 = np.mean(Data16[window,1,:])
D17_1 = np.mean(Data17[window,0,:])
D17_2 = np.mean(Data17[window,1,:])
D18_1 = np.mean(Data18[window,0,:])
D18_2 = np.mean(Data18[window,1,:])

#
# Data aggregate
group1 = np.array([[D1_1, D2_1, D3_1, D4_1],  # Group 1(n=4)
                   [D1_2, D2_2, D3_2, D4_2]]) 
group2 = np.array([[D5_1, D6_1, D7_1, D8_1,  D9_1],  # Group 2(n=5)
                   [D5_2, D6_2, D7_2, D8_2,  D9_2]]) 

group3 = np.array([[D10_1, D11_1, D12_1, D13_1,  D14_1],  # Group 3(n=5)
                   [D10_2, D11_2, D12_2, D13_2,  D14_2]]) 

group4 = np.array([[D15_1, D16_1, D17_1, D18_1],  # Group 4(n=4)
                   [D15_2, D16_2, D17_2, D18_2]]) 

# Ave. SEM
mean_group1 = np.mean(group1, axis=1)
sem_group1 = np.std(group1, axis=1, ddof=1) / np.sqrt(group1.shape[1])

mean_group2 = np.mean(group2, axis=1)
sem_group2 = np.std(group2, axis=1, ddof=1) / np.sqrt(group2.shape[1])

mean_group3 = np.mean(group3, axis=1)
sem_group3 = np.std(group3, axis=1, ddof=1) / np.sqrt(group3.shape[1])

mean_group4 = np.mean(group4, axis=1)
sem_group4 = np.std(group4, axis=1, ddof=1) / np.sqrt(group4.shape[1])

x = [1, 2]

#%% iGlu3 vs 4
plt.figure(figsize=(6, 6))

plt.subplot(1,2,1)
# Group 1
for i in range(group1.shape[1]):
    plt.plot([x[0], x[1]], group1[:, i], color='gray', alpha=0.5, linestyle='-')
    #plt.scatter([x[0], x[1]], group1[:, i], color='gray', alpha=0.8, s=50)

# Group 1 Ave+SEM
plt.errorbar(x[0]-0.2, mean_group1[0], yerr=sem_group1[0], fmt='o', color='goldenrod', capsize=4, markersize=10)
plt.errorbar(x[1]+0.2, mean_group1[1], yerr=sem_group1[1], fmt='o', color='blue', capsize=4, markersize=10)

# axis
plt.xticks(x, ['iGlu3', 'iGlu4'])
plt.ylabel('dF/F (AUC)')
plt.title('DAT-Cre')
plt.xlim([0.5, 2.5])
plt.ylim([0, 7])

plt.subplot(1,2,2)

# Group 2
for i in range(group2.shape[1]):
    plt.plot([x[0], x[1]], group2[:, i], color='gray', alpha=0.5, linestyle='-')
    #plt.scatter([x[0], x[1]], group2[:, i], color='gray', alpha=0.8, s=50)

# Group 2 Ave+SEM
plt.errorbar(x[0] -0.2, mean_group2[0], yerr=sem_group2[0], fmt='o', color='goldenrod', capsize=4, markersize=10)
plt.errorbar(x[1] +0.2, mean_group2[1], yerr=sem_group2[1], fmt='o', color='blue', capsize=4, markersize=10)

# axis
plt.xticks(x, ['iGlu3', 'iGlu4'])
plt.ylabel('dF/F (AUC)')
plt.title('Gad2-Cre')
plt.xlim([0.5, 2.5])
plt.ylim([0, 7])

plt.tight_layout()
plt.show()
#%% SF-iGlu2 vs 4
plt.figure(figsize=(6, 6))

plt.subplot(1,2,2)
# Group 3
for i in range(group3.shape[1]):
    plt.plot([x[0], x[1]], group3[:, i], color='gray', alpha=0.5, linestyle='-')
    #plt.scatter([x[0], x[1]], group1[:, i], color='gray', alpha=0.8, s=50)

# Group 3 Ave+SEM
plt.errorbar(x[0]-0.2, mean_group3[0], yerr=sem_group3[0], fmt='o', color='goldenrod', capsize=4, markersize=10)
plt.errorbar(x[1]+0.2, mean_group3[1], yerr=sem_group3[1], fmt='o', color='blue', capsize=4, markersize=10)

# axis
plt.xticks(x, ['SFiGlu', 'iGlu4'])
plt.ylabel('dF/F (AUC)')
plt.title('GAD-Cre')
plt.xlim([0.5, 2.5])
plt.ylim([0, 7])

plt.subplot(1,2,1)

# Group 4
for i in range(group4.shape[1]):
    plt.plot([x[0], x[1]], group4[:, i], color='gray', alpha=0.5, linestyle='-')
    #plt.scatter([x[0], x[1]], group2[:, i], color='gray', alpha=0.8, s=50)

# Group 2 Ave+SEM
plt.errorbar(x[0] -0.2, mean_group4[0], yerr=sem_group4[0], fmt='o', color='goldenrod', capsize=4, markersize=10)
plt.errorbar(x[1] +0.2, mean_group4[1], yerr=sem_group4[1], fmt='o', color='blue', capsize=4, markersize=10)

# axis
plt.xticks(x, ['SFiGlu', 'iGlu4'])
plt.ylabel('dF/F (AUC)')
plt.title('DAT-Cre')
plt.xlim([0.5, 2.5])
plt.ylim([-2, 7])

plt.tight_layout()
plt.show()


#%% GAD summary
plt.figure(figsize=(6, 4))

plt.subplot(1,2,1)

# Group 2
for i in range(group2.shape[1]):
    plt.plot([x[0], x[1]], group2[:, i], color='gray', alpha=0.5, linestyle='-')
    #plt.scatter([x[0], x[1]], group2[:, i], color='gray', alpha=0.8, s=50)

# Group 2 Ave+SEM
plt.errorbar(x[0] -0.2, mean_group2[0], yerr=sem_group2[0], fmt='o', color='goldenrod', capsize=4, markersize=10)
plt.errorbar(x[1] +0.2, mean_group2[1], yerr=sem_group2[1], fmt='o', color='blue', capsize=4, markersize=10)

# axis
plt.xticks(x, ['iGlu3', 'iGlu4'])
plt.ylabel('dF/F (AUC)')
plt.title('Gad2-Cre')
plt.xlim([0.7, 2.3])
plt.ylim([0, 8])

plt.subplot(1,2,2)
# Group 3
for i in range(group3.shape[1]):
    plt.plot([x[0], x[1]], group3[:, i], color='gray', alpha=0.5, linestyle='-')
    #plt.scatter([x[0], x[1]], group1[:, i], color='gray', alpha=0.8, s=50)

# Group 3 Ave+SEM
plt.errorbar(x[0]-0.2, mean_group3[0], yerr=sem_group3[0], fmt='o', color='goldenrod', capsize=4, markersize=10)
plt.errorbar(x[1]+0.2, mean_group3[1], yerr=sem_group3[1], fmt='o', color='blue', capsize=4, markersize=10)

# axis
plt.xticks(x, ['SFiGlu', 'iGlu4'])
plt.ylabel('dF/F (AUC)')
plt.title('GAD-Cre')
plt.xlim([0.7, 2.3])
plt.ylim([0, 8])

plt.tight_layout()
plt.savefig(SaveDir_Fig + "Figure5d_Summary.pdf")


# %%
from scipy.stats import ttest_rel

t_statistic, p_value = ttest_rel(group2[0, :], group2[1,:])
t_statistic, p_value = ttest_rel(group3[0, :], group3[1,:])

print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")


# %%　revision analysis
x = [1, 2, 3]
plt.figure(figsize=(4, 4))


# Group 2
for i in range(group2.shape[1]):
    plt.scatter(x[1], group2[0, i], color='gray', alpha=0.5, marker='o')
    plt.scatter(x[2], group2[1, i], color='gray', alpha=0.5, marker='o')

# Group 2 Ave+SEM
plt.errorbar(x[1] -0.2, mean_group2[0], yerr=sem_group2[0], fmt='o', color='goldenrod', capsize=4, markersize=10)

for i in range(group3.shape[1]):
    plt.scatter(x[0], group3[0, i], color='gray', alpha=0.5, marker='o')
    plt.scatter(x[2], group3[1, i], color='gray', alpha=0.5, marker='o')

# Group 3 Ave+SEM
plt.errorbar(x[0]-0.2, mean_group3[0], yerr=sem_group3[0], fmt='o', color='goldenrod', capsize=4, markersize=10)
plt.errorbar(x[2]+0.2, np.mean(np.hstack((group2[1, :], group3[1, :]))), yerr=np.std(np.hstack((group2[1, :], group3[1, :])), axis=0, ddof=1) / np.sqrt(np.hstack((group2[1, :], group3[1, :])).size), fmt='o', color='blue', capsize=4, markersize=10)

# axis
plt.xticks(x, ['SFiGlu', 'iGlu3', 'iGlu4'])
plt.ylabel('dF/F (AUC)')
plt.title('GAD-Cre')
plt.xlim([0.7, 3.3])
plt.ylim([0, 8])

plt.tight_layout()
plt.savefig(SaveDir_Fig + "GadCre_234_Summary_nopaired_RevisionFig.pdf")

# %%
df = pd.DataFrame({
    "group": ["control"] * 10 + ["SFiGlu"] * 5 + ["iGlu3"] * 5,
    "value": np.hstack((group2[1, :], group3[1, :], group3[0, :],group2[0, :])).tolist()
})

dunn = sp.posthoc_dunn(df, val_col="value", group_col="group", p_adjust="holm")
print(dunn)
