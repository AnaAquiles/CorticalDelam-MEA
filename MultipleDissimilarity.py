import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.manifold import MDS

"""
   This code will need some outputs from MI_signalModules.py

"""


Example = np.concatenate((MatFreq, MatFreq2, MatFreq3, MatFreq4, MatFreq5), axis=0) # all the matrices of the different activities

def compute_pairwise_dists(mat_a, mat_b):
    m = mat_a.shape[0]
    dists = np.zeros((m, mat_b.shape[0]))
    for i in range(m):
        dists[i] = np.linalg.norm(mat_a[i] - mat_b, axis=1)
    return dists

pairwise_dists = [
    compute_pairwise_dists(Example[i], Example[i+1])
    for i in range(len(Example) - 1)
]
DissVals = np.array([dist[0] for dist in pairwise_dists])


dist_mtx = manhattan_distances(DissVals)
mds = MDS(dissimilarity='precomputed', random_state=0)
X_mds = mds.fit_transform(dist_mtx)


def fit_ellipse_area(x_vals, y_vals):
    cov = np.cov(x_vals, y_vals)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    return math.pi * ell_radius_x * ell_radius_y


start, end = 0, 15
x_sel, y_sel = X_mds[start:end, 0], X_mds[start:end, 1]
ellipse_area = fit_ellipse_area(x_sel, y_sel)

# Plot ellipse 
plt.figure()
plt.style.use('fivethirtyeight')
plt.scatter(x_sel, y_sel,
            c=np.linspace(0.5, 9.5, end - start),
            s=500, alpha=0.4, cmap='rainbow', vmin=1, vmax=2, zorder=1)
plt.title('Mg Freq')
plt.colorbar()
plt.grid(False)


# Plot example output from MDS
plt.figure(figsize=(8, 6))
plt.style.use('fivethirtyeight')
plt.scatter(X_mds[100:, 0], X_mds[100:, 1],
            c=np.linspace(0.5, 9.5, 19),
            s=500, alpha=0.4, cmap='rainbow', vmin=1, vmax=2, zorder=1)
plt.title('Mg2')
plt.colorbar()
plt.grid(False)
plt.xlim(-130, 70)
plt.ylim(-130, 50)
