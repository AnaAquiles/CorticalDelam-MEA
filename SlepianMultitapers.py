
from scipy.signal.windows import dpss
import numpy as np
from scipy import signal


"""
         This code will need some outputs from Preprocessing.py
"""

def slepian_multitapers(d,fs, window, NW, tapers,overlap, electrodes):

    
    timeWin =  window 
    timeWinID = int(np.round(timeWin/(window/fs)))
    tapers = dpss(timeWinID, NW, tapers, overlap) 
    
    ### window fixed 
    LenWind = window 
    OverLap = int(LenWind/overlap) #75% 
    electrodes = len(d)
    tap = len(tapers)
    

    data = []
    for n in range(0,len(d[0,:]) - LenWind + 1, OverLap):
        for i in range(0,len(d)):
            data.append(d[i,n:n+LenWind])
            
    data = np.array(data) 
    data = np.array([data[i:i+electrodes] for i in range(0,len(data),electrodes)])
    data = signal.detrend(data, axis =-1, type='constant')                   
    
    
    taperD= [] 

    for i in range(0,len(data)):
        for j in range(0, len(tapers)):
            for n in range(0,len(data[0,:,:])):
                taperD.append(data[i,n,:] * tapers[j,:])   

    taperD = np.array(taperD)
    # Reshape the result of the last computation
    taperDd = np.array([taperD[i:i+tap] for i in range(0,len(taperD),tap)])    #len tapers 
    taperDd_ele = np.array([taperDd[i:i+electrodes] for i in range(0,len(taperDd),electrodes)])  #len electrodes
       
    Fft_Tap = np.fft.fft(taperDd_ele, axis = 3)
    
    
    ###   MEAN ALONG THE TAPERS  
    Fft_PowerT = np.mean(np.absolute(Fft_Tap), axis=2)
    
    # TAKE THE FREQUENCIES OF INTEREST < 100 Hz  
    f = np.linspace(0.5,window/2,(timeWinID//2))  
    F_t = np.argmin(np.abs(f-99))
    
    f = f[0:F_t]

    #### Log frequency selection 

    N = len(f)
    F_log = np.log10(np.logspace(f[0],f[-1], 60))     ### reduce frequqency partitions into 60
    
    Fid = []
    for i in range(0,len(F_log)):
      Fid.append(np.argmin(np.abs(f-F_log[i])))
      
    Fid = np.array(Fid).astype(int)
    f = np.take(f,Fid)
    Fft_PowerN = np.take(Fft_PowerT,Fid,axis = 2)
    f_Norm = np.mean(Fft_PowerN, axis=0)
    
    Fft_PowerNorm = []
    for i in range(0,electrodes):
        Fft_PowerNorm.append(np.log10(Fft_PowerN[:,i,:]/f_Norm[i,:]))
    
    Fft_PowerNorm = np.array(Fft_PowerNorm)
 
    
    time = np.linspace(1,300,len(taperDd_ele))
    Spectro = np.concatenate(Fft_PowerNorm, axis=1)                           
    return Fft_PowerNorm,time, f

dataset = DownDataset  #from the preprocessing analysis


datos = signal.detrend(np.squeeze(dataset[:,:900000]))                         
Fft_Pow, TimeTap, F_f = slepian_multitapers(dataset, 1000,1000, 3, 6, 3, 59)