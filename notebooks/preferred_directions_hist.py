import numpy as np
import scipy
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
cursor_bin = np.load(data_dir / 'cursor_bins.npy')
spikes_bin = np.load(data_dir / 'spikes_bins_50ms.npy')

# Rate of change of x and y position
x_diff = np.diff(cursor_bin[0, :])
y_diff = np.diff(cursor_bin[1, :])

# Movement angle relative to positive x-axis
angle = np.arctan2(y_diff, x_diff)


# Neuron firing rate modeled as a cosine function
def firing_rate(theta, b, g, phi):
    cos_model = b + g * np.cos(theta - phi)
    return cos_model


phi_list = []
for i in range(96):
    angle_bin = scipy.stats.binned_statistic(angle, spikes_bin[:-1, i], statistic='mean', bins=8)
    X = (angle_bin.bin_edges[:-1] + angle_bin.bin_edges[1:]) / 2
    try:
        best_params, _ = scipy.optimize.curve_fit(firing_rate, X, angle_bin.statistic)
    except RuntimeError:
        continue
    phi_list.append(best_params[2])

# Plot preferred directions on histogram
Y = phi_list
plt.hist(Y)
plt.show()

np.save(data_dir / 'phi_list.npy', phi_list)

