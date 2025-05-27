import numpy as np
import matplotlib.pyplot as plt

"""
        This script need of somes  outputs from SlepianMultitapers.py

"""

PowerSpec = Fft_Pow    # From SlepianMultitapers decomposition
Frequencies = F_f      # From SlepianMultitapers decomposition
Time = TimeTap         # From SlepianMultitapers decomposition
electrodes = 59


def CommonBand(PowerSpec, Frequencies):

    TimeW = int(len(PowerSpec[0,:,:]))

    CovSpec = [] 
    for i in range(0,len(PowerSpec[0,:,:])):
        CovSpec.append(np.cov(PowerSpec[:,i,:].T))
        
    CovSpec = np.array(CovSpec)
    eigval = np.zeros((TimeW,len(Frequencies)))
    eigvec = np.zeros((TimeW,len(Frequencies),len(Frequencies)))

    for i in range(TimeW):
        eigval[i,:],eigvec[i,:,:] = np.linalg.eigh(CovSpec[i,:,:])
        
    indexes = np.flip(np.argsort(eigval[0,:]))
    eigval = eigval[:,indexes]
    eigvec = eigvec[:,:,indexes]

    return eigvec,eigval

eigvec, eigval = CommonBand(PowerSpec, Frequencies)



#######     PLOT SECTION 
#######   SPECTOGRAM per PCA ELECTRODE CovTime result 

fig,axs = plt.subplots(nrows = 1, ncols = 4, figsize=(8,6))

fig.suptitle("First 4 components", fontsize = 12, y = 0.95)

axs = axs.ravel()

for d in range(4):
    axs[d].pcolormesh(Time, Frequencies[:], np.abs(eigvec[:,:,d].T),
                      vmin = 0, vmax=0.8, cmap = 'rainbow')
    axs[d].set_ylim(0,25)
    axs[d].set_title(str(d))


tot = sum(eigval[0,:])
var_exp = [(i / tot) for i in sorted(eigval[0,:], reverse=True)]
cum_var_exp = np.cumsum(var_exp)

## Plot explained variances and mean of Power Dimensionality Reduction 
plt.style.use('fivethirtyeight')
plt.figure()
plt.bar(range(60), var_exp, color ='r',alpha=0.5,
        align='center', label='Varianza individual')
plt.step(range(60), cum_var_exp, where='mid',
         label='Varianza acumulada',color = 'r',alpha = 0.5)
plt.ylabel('Proportion of variance')
plt.xlabel('Modes')
plt.legend(loc='best')
plt.box(False)
plt.show()

#####
 
eigvecMean = np.mean(eigvec, axis=0)
plt.figure()
plt.plot(np.abs(eigvecMean[:,0].T), label = 'Mode 1')
plt.plot(np.abs(eigvecMean[:,1].T), label = 'Mode 2')
plt.legend()