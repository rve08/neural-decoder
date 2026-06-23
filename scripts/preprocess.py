import h5py
import numpy as np
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
raw_data = h5py.File(data_dir / "indy_20160627_01.mat", "r")

t = raw_data["t"][0]  # (840737,) timestamps at 250Hz
cursor_pos = raw_data["cursor_pos"][:]  # (2, 840737)

# Spike binning — 50ms bins
bin_edges = np.arange(t[0], t[-1], 0.05)

binned_spikes_raw = []
for i in range(96):
    ref = raw_data["spikes"][0, i]
    spike_times = raw_data[ref][:].flatten()
    counts, _ = np.histogram(spike_times, bin_edges)
    binned_spikes_raw.append(counts)
binned_spikes = np.array(binned_spikes_raw).transpose()  # (n_bins, 96)
np.save(data_dir / "spikes_bins_50ms.npy", binned_spikes)

# Cursor — resampled onto the SAME 50ms grid via interpolation
bin_centers = bin_edges[:-1] + 0.025  # one time point per bin
cursor_x = np.interp(bin_centers, t, cursor_pos[0])
cursor_y = np.interp(bin_centers, t, cursor_pos[1])
cursor_binned = np.array([cursor_x, cursor_y])  # (2, n_bins)
np.save(data_dir / "cursor_bins.npy", cursor_binned)

print("spikes:", binned_spikes.shape, "cursor:", cursor_binned.shape)
