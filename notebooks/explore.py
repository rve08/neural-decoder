import numpy as np
import matplotlib.pyplot as plt
import scipy
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
cursor_bin = np.load(data_dir / 'cursor_bins.npy')
spikes_bin = np.load(data_dir / 'spikes_bins_50ms.npy')

# Rate of change of x and y position
x_diff = np.diff(cursor_bin[0, :])
y_diff = np.diff(cursor_bin[1, :])

# Movement angle relative to positive x-axis
angle = np.arctan2(y_diff, x_diff)


def firing_rate(x_diff, b, g, phi):
    cos_model = b + g * np.cos(x_diff - phi)
    return cos_model


for i in range(5):
    angle_bin = scipy.stats.binned_statistic(angle, spikes_bin[:-1, i], statistic='mean', bins=8)
    # Tuning curve plot of neuron 0
    X = (angle_bin.bin_edges[:-1] + angle_bin.bin_edges[1:]) / 2
    Y = angle_bin.statistic
    plt.plot(cursor_bin[1, :500])
plt.show()
