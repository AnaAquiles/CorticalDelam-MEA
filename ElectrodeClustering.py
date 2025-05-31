import numpy as np
import matplotlib.pyplot as plt 
from sklearn.cluster import MeanShift, estimate_bandwidth
import pandas as pd
from collections import Counter


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

eigval, eigvec = FrequencyCovariation_E(PowerSpec, Frequencies, electrodes)

### plot variance explained

# tot = sum(eigval[0,:])
# var_exp = [(i / tot) for i in sorted(eigval[0,:], reverse=True)]
# cum_var_exp = np.cumsum(var_exp)


d = {'Electrodes': [], 'Layer' :[]}      #
dataF = pd.DataFrame(d)                  #
dataF['PC1'] = np.abs(eigvec[:,0].T)     #    # Select the number of principal component
dataF['PC2'] = np.abs(eigvec[:,1].T)     # 


####        MeanShift CLUSTERING

X = np.array((dataF['PC1'].values, dataF['PC2'].values))
bandwidth = estimate_bandwidth(X.T, quantile = 0.65, n_samples=80) 
clustering = MeanShift(bandwidth=bandwidth).fit(X.T)

labels = clustering.labels_
center = clustering.cluster_centers_

pattern = [
    "01111110",
    "11111111",
    "11111111",
    "11111111",
    "01111111",
    "11111111",
    "11111111",
    "01111110"
]

spacing = 1.0
hex_height = spacing * (3 ** 0.5) / 2
hex_width = spacing

positions = []
for row_idx, row in enumerate(pattern):
    for col_idx, char in enumerate(row):
        if char == "1":
            x = col_idx * hex_width * 1.5
            y = row_idx * hex_height
            positions.append((x, y))

# Assign colors per label using a categorical colormap
unique_labels = sorted(set(labels))
n_clusters = len(unique_labels)
cmap = plt.cm.get_cmap('viridis', n_clusters)
colors = [cmap(label) for label in labels]

# Epsilon metric calculation
label_history = [[label] for label in labels]  # Simulating label history

L = len(label_history)
n_points = [len(history) for history in label_history]
n_colors = [len(set(history)) for history in label_history]

epsilon = sum([n_points[i] / n_colors[i] for i in range(L)]) / n_clusters if n_clusters > 0 else 0
epsilon_normalized = epsilon / len(labels)  # Normalize epsilon to be between 0 and 1
print(f"Epsilon diversity metric (ε): {epsilon_normalized:.4f}")

# Plotting the MEA
plt.figure(figsize=(8, 8))
for i, (x, y) in enumerate(positions):
    plt.scatter(x, y, s=400, color=colors[i], edgecolor='w')

# Add legend with counts
label_counts = Counter(labels)
for label in unique_labels:
    plt.scatter([], [], color=cmap(label), label=f"Cluster {label} ({label_counts[label]})")
plt.legend(title='Clusters', loc='upper right')

plt.title(f"MEA Cluster Assignment\nEpsilon (ε) = {epsilon_normalized:.3f}")
plt.axis('equal')
plt.axis('off')
plt.show()


# #        EXPLORATORY PLOTS

# # 2 first MODES of the dimension reduction 
# plt.style.use('fivethirtyeight')
# plt.figure(figsize = (10,10))
# plt.scatter(dataF['PC1'],dataF['PC2'],
#             c = np.arange(0,electrodes,1), s =300,  
#             cmap = 'Set2', alpha = 0.9)
# plt.colorbar()
# plt.box(False)

# plt.style.use('fivethirtyeight')
# plt.figure()
# plt.plot(dataF['PC1'].values, label = 'Mode 1')
# plt.plot(dataF['PC2'].values, label = 'Mode 2')
# plt.legend()


# # Cumulate variance

# plt.figure()
# plt.subplot(121)
# plt.bar(range(60), var_exp, color= 'k',alpha=0.3,
#         align='center', label='Individual variance')
# plt.ylim(0,0.2)
# plt.ylabel('Proportion of variance')
# plt.xlabel('Modes')
# plt.legend(loc='best')
# plt.box(False)
# plt.grid(False)
# plt.subplot(122)
# plt.step(range(60), cum_var_exp, where='mid',
#          label='Cumulate variance',color= 'k',alpha = 0.3)
# plt.ylabel('Proportion of variance')
# plt.xlabel('Modes')
# plt.legend(loc='best')
# plt.box(False)
# plt.grid(False)
# plt.show()

# #  Scatter projection of the  two principal components of the frequency covariation

# plt.figure(figsize = (6,5))
# plt.scatter(dataF['PC1'],dataF['PC2'], c = labels.astype(float)*0.5,  s =300,
#             alpha =0.9)
# plt.scatter(center[:,0], center[:,1], c= 'r', marker = 'x', alpha = 0.9)
# plt.title("Principal Components projection geometic space")
# plt.colorbar()
# plt.box(False)

# plt.figure(figsize=(7, 6))
# for i, (x, y, color) in enumerate(coordinates):
#     plt.scatter(x, y, c=color, s=120, edgecolor='black')
#     plt.text(x, y + 0.2, str(labels[i]), ha='center', fontsize=8)

# plt.title(f'Hexagonal Array with MeanShift Clusters\nϵ = {epsilon_metric:.4f}')
# plt.gca().set_aspect('equal')
# plt.grid(True)
# plt.show()

