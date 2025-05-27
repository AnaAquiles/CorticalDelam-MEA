import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

"""
            This code will need some outputs from Preprocessing.py
     
"""


# Parameters 
fs = 1000  # Sampling frequency
window_length = 15000  # 15 seconds
delay = 5000           # 5-second delay
n_electrodes = 59


def slice_windows(data, window_len, stride, n_electrodes):
    
    windows = [
        data[chan, start:start + window_len]
        for start in range(0, data.shape[1] - window_len + 1, stride)
        for chan in range(n_electrodes)
    ]
    return np.array(windows)

def compute_features(signal):
    """Return normalized amplitude and phase from the analytic signal."""
    analytic = hilbert(signal)
    amplitude = np.abs(analytic)
    norm_amp = np.log10(amplitude / np.mean(amplitude))
    phase = np.angle(analytic)
    return norm_amp, phase

# Mutual Information (your original functions preserved)

def shannon_entropy(counts):
    prob = counts / np.sum(counts)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def calc_mutual_info(x, y, bins='sturges', value_range=(0, 5)):
    bin_x = np.histogram_bin_edges(x, bins=bins, range=value_range)
    bin_y = np.histogram_bin_edges(y, bins=bins, range=value_range)
    hist_xy, _, _ = np.histogram2d(x, y, bins=(bin_x, bin_y))
    hist_x = np.histogram(x, bins=bin_x)[0]
    hist_y = np.histogram(y, bins=bin_y)[0]
    return shannon_entropy(hist_x) + shannon_entropy(hist_y) - shannon_entropy(hist_xy)

def compute_mi_matrix(data, n_windows):
    """Compute MI matrix for each time window."""
    n_channels = data.shape[1]
    mi_matrix = np.zeros((n_windows, n_channels, n_channels))

    for w in range(n_windows):
        for i in range(n_channels):
            for j in range(n_channels):
                mi_matrix[w, i, j] = calc_mutual_info(data[w, i], data[w, j])
    return mi_matrix



# Data slicing 
data_segment = DataFiltBP_RA[:, :300000]
windows = slice_windows(data_segment, window_length, window_length, n_electrodes)
windows_stacked = windows.reshape(-1, n_electrodes, window_length)

delayed = slice_windows(data_segment, window_length, window_length + delay, n_electrodes)
delayed_array = delayed.reshape(-1, window_length)
n_windows = delayed_array.shape[0] // n_electrodes
delayed_stacked = delayed_array.reshape(n_windows, n_electrodes, window_length)

# Feature extraction 
amp_window, phase_window = compute_features(windows_stacked)
amp_delay, phase_delay = compute_features(delayed_array)

# Reshape features for further processing
phase_window = phase_window.reshape(n_windows, n_electrodes, window_length)
phase_delay = phase_delay.reshape(n_windows, n_electrodes, window_length)

inst_freq_window = np.diff(phase_window, axis=2) / (2 * np.pi) * fs
inst_freq_delay = np.diff(phase_delay, axis=2) / (2 * np.pi) * fs

# Variance calculations 
within_var = np.var(phase_window, axis=2).mean(axis=1)
between_var = np.var(np.mean(phase_window, axis=2), axis=1)
pooled_var = within_var + between_var
median_pooled = np.median(pooled_var)

# Mutual Information analysis 
mi_phase = compute_mi_matrix(phase_window, n_windows)
mi_amp = compute_mi_matrix(np.abs(amp_window), n_windows)
mi_freq = compute_mi_matrix(inst_freq_window, n_windows)




# Sample channel 
plt.style.use('fivethirtyeight')
plt.figure()
plt.plot(DataFiltBP_RA[45, :300000], 'k-', alpha=0.5, label='Channel 45')
plt.title('Signal Preview')
plt.legend()
plt.box(False)

# Visualize one example MI matrix
plt.figure()
plt.imshow(mi_amp[8], cmap='YlGnBu_r', alpha=0.8, vmin=0, vmax=1.0)
plt.title('Mutual Information (Amplitude) - Example Window')
plt.xlabel('Electrodes')
plt.ylabel('Electrodes')
plt.colorbar()
plt.grid(False)

# --- CSV Export Function ---
#def label_and_export(data_list, labels, filename):
#    """Combine data lists and attach experimental condition labels before saving."""
#    data = np.concatenate(data_list)
#    df = pd.DataFrame(data)
#    df['Category'] = [labels[i // 885 % len(labels)] for i in range(len(df))]
#    df.to_csv(filename, index=False)_
