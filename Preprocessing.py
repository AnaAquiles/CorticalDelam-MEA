
import sys, importlib, os
import McsPy.McsData
from McsPy import ureg, Q_

import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt,lfilter
from scipy import signal

import math
import numpy as np


# Data IMPORTATION AND EXPLORATION

test_data_folder = r'/yourPATH/'          
channel_raw_data = McsPy.McsData.RawData(os.path.join(
    test_data_folder, 'AP005_BCNU_p35_60_200_cero0002.h5'))

def voltageValues (data = channel_raw_data, Fs =  22500000):
    
    analog_stream_0 = channel_raw_data.recordings[0].analog_streams[0]
    analog_stream_0_data = analog_stream_0.channel_data
    np_analog_stream_0_data = np.transpose(analog_stream_0_data)

    channel_ids = channel_raw_data.recordings[0].analog_streams[0].channel_infos.keys()
    channel_id = list(channel_raw_data.recordings[0].analog_streams[0].channel_infos.keys())[0]

    stream = channel_raw_data.recordings[0].analog_streams[0]
    time = stream.get_channel_sample_timestamps(channel_id, 0, )
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude

    time_in_sec = time[0] * scale_factor_for_second

    signal = stream.get_channel_in_range(channel_id, 0, Fs)
    sampling_frequency = stream.channel_infos[channel_id].sampling_frequency.magnitude 

    data = channel_raw_data.recordings[0].analog_streams[0].channel_data[:, 0:]

    return data, channel_ids, time_in_sec

data, channelID, time = voltageValues()


#### Downsampling from 22.5kHz to 1 kHz 


from scipy import signal
from scipy.signal import butter,filtfilt

def Downsampling (data = data, Fs_base = 2255000, Fs_down = 1000, nElectrodes = 60):

    u_T = len(data[0,:])/Fs_base
    down = int(u_T * Fs_down)

    DownSample = []

    for i in range(0,nElectrodes):
        DownSample.append(signal.resample(data[i,:],down))

    DownData = np.array(DownSample)

    return DownData

DownDataset = Downsampling()


#### BAND PASS EXPLORATION

fs = 1000
lowcut = 0.5
highcut = 3


def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

filtered = []
for i in range(0,60):
    filtered.append(butter_bandpass_filter(DownDataset[i], lowcut, highcut, fs, order = 3))
    
DataFiltBP = np.array(filtered)


def reArrangeELECTRODE (dataset):


    newForm = [23,25,28,31,34,36,20,21,24,29,30,35,38,39,18,19,22,
                27,32,37,40,41,15,16,17,26,33,42,43,44,
                14,13,12,3,56,47,46,45,11,10,7,2,57,52,
                49,48,9,8,5,0,59,54,51,50,6,4,1,58,55,53]

    idx = np.empty_like(newForm)
    idx[newForm] = np.arange(len(newForm))
    dataset[:] = dataset[idx,:]

    return dataset

DataFiltBP_RA = reArrangeELECTRODE(DataFiltBP)


####### EXPLORE YOUR ORIGINAL, DOWNSAMPLED AND BAND FILTERED DATA 
plt.style.use('fivethirtyeight')
plt.figure(2)
plt.subplot(411)
plt.plot(time[:], data[27,:], label = 'Raw Data')
plt.legend()
plt.box(False)

plt.subplot(412)
plt.plot(time[:60000], DownDataset[27,120000:180000], label = 'Downsample Data')
plt.legend()
plt.box(False)

plt.subplot(413)
plt.plot(time[:60000], DataFiltBP[27,120000:180000], label = 'Low pass 200Hz Data')
# plt.xlim(0,4)
plt.legend()
plt.box(False)

plt.subplot(414)
powerSpectrum, frec, y = plt.magnitude_spectrum(DataFiltBP[27,120000:180000],Fs =1000,
                                                color = 'coral', alpha = 1, 
                       label = 'Magnitude Spectrum')
plt.ylim(0,300)
plt.xlim(0,100)
plt.legend()
plt.box(False)



