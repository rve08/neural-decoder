from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

data_dir = Path(__file__).parent.parent / "data"
raw_data = h5py.File(data_dir / "indy_20160627_01.mat", "r")

spike_times = []
for i in range(96):
    ref = raw_data["spikes"][0, i]
    times = raw_data[ref][:].flatten()
    spike_times.append(times)

for j in range(96):
    x = spike_times[j][spike_times[j] < (spike_times[0].min() + 60)]
    y = np.full(len(x), j)
    plt.scatter(x, y, s=0.1)
plt.show()
