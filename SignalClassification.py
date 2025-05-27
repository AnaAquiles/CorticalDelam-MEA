import numpy as np
from scipy.signal import butter, lfilter

"""
       This code will need some outputs from Preprocessing.py and ElectrodeClustering.py
"""

DownData = DataFiltBP_RA


cluster_0_indices = np.where(labels == 0)[0]
cluster_1_indices = np.where(labels == 1)[0]
cluster_2_indices = np.where(labels == 2)[0]


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    return butter(order, [low, high], btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)


fs = 1000  # Sampling frequency
lowcut = 0.5
highcut = 30
n_channels = DownData.shape[0]


filtered_data = np.array([
    butter_bandpass_filter(DownData[ch], lowcut, highcut, fs, order=3)
    for ch in range(n_channels)
])


filtered_data = np.delete(filtered_data, 30, axis=0)


Act1 = filtered_data[cluster_0_indices]
Act2 = filtered_data[cluster_1_indices]
Act3 = filtered_data[cluster_2_indices]
