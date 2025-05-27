import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd

"""
          This code will need some outputs from SlepianMultitapers.py
"""


PowerSpec = Fft_Pow    # From SlepianMultitapers decomposition
Frequencies = F_f      # From SlepianMultitapers decomposition
electrodes = 59


def FrequencyCovariation_E(PowerSpec, Frequencies, electrodes):

    CovElec = []
    for i in range(electrodes):
        CovElec.append(np.cov(PowerSpec[i,:,:].T))
        
    CovElec = np.array(CovElec)

    #PCA per electrode
    eigval = np.zeros((electrodes,60))
    eigvec = np.zeros((electrodes,60,60))

    for i in range(electrodes):
        eigval[i,:], eigvec[i,:,:] = np.linalg.eigh(CovElec[i,:,:])

    indexes = np.flip(np.argsort(eigval[0,:]))
    eigval = eigval[:,indexes]
    eigvec = eigvec[:,:,indexes]

    maximum = np.max(np.abs(eigvec))
    minimum = np.min(np.abs(eigvec))
    Maximum = np.max(np.abs(CovElec))

    eigvecE = eigvec[:,:,0].T ## PC1
    eigvecE2 = eigvec[:,:,1].T ## PC2

    MatPC1 = eigvecE * Frequencies [:,None]    #eigval, frequency 
    MatPC2 = eigvecE2 * Frequencies[:,None]    #eigval, frequency 

    CovMat = []
    for i in range(electrodes):
        for j in range(electrodes):
            CovMat.append(MatPC1[:,i]* MatPC1[:,j])
        
    CovMat = np.array(CovMat)
    CovMat = np.array([CovMat[i:i+electrodes] for i in range(0,len(CovMat),electrodes)]) 
    CovMat = np.mean(CovMat, axis=2)

    eigval_r, eigvec_r= np.linalg.eigh(CovMat)
    indexes = np.flip(np.argsort(eigval_r))
    eigval_r = eigval_r[indexes]
    eigvec_r = eigvec_r[:, indexes]
    maximum = np.max(np.abs(eigvec_r))

    return eigval_r, eigvec_r

eigval, eigvec = FrequencyCovariation_E()

### plot variance explained

tot = sum(eigval[0,:])
var_exp = [(i / tot) for i in sorted(eigval[0,:], reverse=True)]
cum_var_exp = np.cumsum(var_exp)


d = {'Electrodes': [], 'Layer' :[]}      #
dataF = pd.DataFrame(d)                  #
dataF['PC1'] = np.abs(eigvec[:,0].T)     #    # Select the number of principal component
dataF['PC2'] = np.abs(eigvec[:,1].T)     # 


####        MeanShift CLUSTERING

X = np.array((dataF['PC1'].values, dataF['PC2'].values))
bandwidth = estimate_bandwidth(X.T, quantile = 0.45, n_samples=80) 
clustering = MeanShift(bandwidth=bandwidth).fit(X.T)

labels = clustering.labels_
center = clustering.cluster_centers_


###     Epsilon metric 

import matplotlib.pyplot as plt
import random
from collections import defaultdict

#  MEA PATTERN
pattern = [
    "01111110",
    "11111111",
    "11111111",
    "11111111",
    "11111111",
    "11111111",
    "11111111",
    "01111110"
]


spacing = 1.0
hex_height = spacing
hex_width = spacing * (3 ** 0.5) / 2

coordinates = []
colors = [ 'green', 'blue', 'orange'] 

for row_idx, row in enumerate(pattern):
    for col_idx, char in enumerate(row):
        if char == "1":
            x = col_idx * hex_width + (hex_width / 2 if row_idx % 2 == 1 else 0)
            y = row_idx * hex_height * 0.87
            color = random.choice(colors)
            coordinates.append((x, y, color))


positions = np.array([[x, y] for x, y, c in coordinates])

n_clusters = len(np.unique(labels))

cluster_colors = defaultdict(list)
for i, label in enumerate(labels):
    cluster_colors[label].append(coordinates[i][2])  # Append the color


epsilons = []
for cluster_id, color_list in cluster_colors.items():
    n_points = len(color_list)
    n_colors = len(set(color_list))
    epsilon_i = n_colors / n_points if n_points > 0 else 0
    epsilons.append(epsilon_i)

epsilon_metric = sum(epsilons) / n_clusters



#        EXPLORATORY PLOTS

# 2 first MODES of the dimension reduction 
plt.style.use('fivethirtyeight')
plt.figure(figsize = (10,10))
plt.scatter(dataF['PC1'],dataF['PC2'],
            c = np.arange(0,electrodes,1), s =300,  
            cmap = 'Set2', alpha = 0.9)
plt.colorbar()
plt.box(False)

plt.style.use('fivethirtyeight')
plt.figure()
plt.plot(dataF['PC1'].values, label = 'Mode 1')
plt.plot(dataF['PC2'].values, label = 'Mode 2')
plt.legend()


# Cumulate variance

plt.figure()
plt.subplot(121)
plt.bar(range(60), var_exp, color= 'k',alpha=0.3,
        align='center', label='Individual variance')
plt.ylim(0,0.2)
plt.ylabel('Proportion of variance')
plt.xlabel('Modes')
plt.legend(loc='best')
plt.box(False)
plt.grid(False)
plt.subplot(122)
plt.step(range(60), cum_var_exp, where='mid',
         label='Cumulate variance',color= 'k',alpha = 0.3)
plt.ylabel('Proportion of variance')
plt.xlabel('Modes')
plt.legend(loc='best')
plt.box(False)
plt.grid(False)
plt.show()

#  Scatter projection of the  two principal components of the frequency covariation

plt.figure(figsize = (6,5))
plt.scatter(dataF['PC1'],dataF['PC2'], c = labels.astype(float)*0.5,  s =300,
            alpha =0.9)
plt.scatter(center[:,0], center[:,1], c= 'r', marker = 'x', alpha = 0.9)
plt.title("Principal Components projection geometic space")
plt.colorbar()
plt.box(False)

plt.figure(figsize=(7, 6))
for i, (x, y, color) in enumerate(coordinates):
    plt.scatter(x, y, c=color, s=120, edgecolor='black')
    plt.text(x, y + 0.2, str(labels[i]), ha='center', fontsize=8)

plt.title(f'Hexagonal Array with MeanShift Clusters\nϵ = {epsilon_metric:.4f}')
plt.gca().set_aspect('equal')
plt.grid(True)
plt.show()