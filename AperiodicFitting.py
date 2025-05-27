import powerlaw as pwl
import numpy as np
from scipy import optimize
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d


'''
         Power spectrum only from the frequencies 0.5-100 Hz

          Fitting of aperiodic exponent using the same formula
        of the paper 2022 Thomas Donoghue Technical Report NATURE


    ##### This script needs some outputs from Preprocessing.py######


'''
 
def powerSpec(datos,fs, th):  

    freqs = np.fft.rfftfreq(len(datos), d=1/fs)
    powerspec = np.abs(np.fft.rfft(datos))**2
    threshold = th
    indices = freqs > threshold
    Freq = freqs[indices]
    Pow = powerspec[indices]
    totalPow = np.sum(powerspec)
    normPow = powerspec/totalPow
    
    PowN = normPow[indices]
  
    return Freq,Pow,PowN

electrodes = 59
Points = 49950     # 150001 - 5 min  # 50001 -100 s  #5001 - 10s
    
Freqs = np.zeros((electrodes,Points)) 
PowerN = np.zeros((electrodes, Points))
Power = np.zeros((electrodes, Points))

for i in range(electrodes):
    Freqs[i,:], Power[i,:],PowerN[i,:] = powerSpec(DataFiltBP_RA[i,:100000],1000,0.5) # from preprocessing.py

def combinedMod(bias, f,alpha,k):
    return bias-(np.log(k + f**(alpha))) # Lorentzian Function 
def fit (freqs,Power):
    popt, pcov = curve_fit(combinedMod, freqs, Power)
    alpha, k= popt
    return alpha

AperiodicExp = np.zeros(electrodes)

for i in range(electrodes):
    AperiodicExp[i] = np.abs(fit(Freqs[i,:], Power[i,:]))
    
    
smoothed_power = []
for i in range(electrodes):
    smoothed_power.append(gaussian_filter1d(PowerN[i,:], sigma = 5))
smoothed_power = np.array(smoothed_power)

def varianceLog(spectrogram):

    log_powerSpec = np.log10(spectrogram + 1e-10)
    logpowerMEAN = np.mean(log_powerSpec, axis = 0)  
    logvariancePower = np.var(log_powerSpec, axis = 0)

    return logvariancePower, logpowerMEAN


def bimodality_coefficient(data):
    n = len(data)
    mean = np.mean(data)
    std_dev = np.std(data, ddof=1)

    skewness = np.mean(((data - mean) / std_dev) ** 3)
    kurtosis = np.mean(((data - mean) / std_dev) ** 4) - 3  # Excess kurtosis

    bc = (skewness**2 + 1) / (kurtosis + 3 * ((n-1)**2) / ((n-2)*(n-3)))
    return bc, skewness, kurtosis

bc, skew, kurt = bimodality_coefficient(AperiodicExp)

### bc < 0.05 - evidence of bimodality
### bc > 0.05 - likely unimodality

# Print results
print(f"Bimodality Coefficient: {bc}")
print(f"Skewness: {skew}")
print(f"Kurtosis (Excess): {kurt}")


