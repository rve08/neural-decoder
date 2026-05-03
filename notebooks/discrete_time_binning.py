import numpy as np
import matplotlib.pyplot as plt
import h5py

raw_data = h5py.File(r'C:\Users\ruben\Desktop\Neuro\Neural-Decoder\data\indy_20160627_01.mat', 'r')

spike_times = []
for i in range(96):
    ref = raw_data['spikes'][0, i]
    times = raw_data[ref][:].flatten()
    spike_times.append(times)

# binning into discrete time bins of 50ms width
binned_spikes_raw = []
for j in range(96) : 
    spikes_binned = np.histogram(spike_times[j] , np.arange(raw_data["t"][0, 0], raw_data["t"][0, -1], 0.05))
    binned_spikes_raw.append(spikes_binned[0])
binned_spikes = np.array(binned_spikes_raw).transpose()
np.save(r'C:\Users\ruben\Desktop\Neuro\Neural-Decoder\data\spikes_bins_50ms.npy' , binned_spikes)

# Resampling cursor_pos to roughly match spikes
cursor_binned_full = raw_data['cursor_pos'][:, ::12]
cursor_binned = cursor_binned_full[0:2 , 0: 67258]
np.save(r'C:\Users\ruben\Desktop\Neuro\Neural-Decoder\data\cursor_bins.npy', cursor_binned )