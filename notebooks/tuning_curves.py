
"""
Neural decoding mini-project:
- Plots neuron tuning graphs for well-tuned neurons
- Overlays a fitted cosine function model of the tuning graph
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy

cursor_bin = np.load('data/cursor_bins.npy')
spikes_bin = np.load(r'data/spikes_bins_50ms.npy')

# Rate of change of x and y position
x_diff = np.diff(cursor_bin[0, : ])
y_diff = np.diff(cursor_bin[1, :])

# Movement angle relative to positive x-axis
angle = np.arctan2(y_diff, x_diff)
np.save('data/actual_movement_angle.npy', angle)

# Neuron firing rate modeled as a cosine function
def firing_rate(theta, b , g, phi): 
    cos_model = b + g*np.cos(theta - phi)
    return cos_model

tuning_list = []
neuron_list = []
for i in range(96):
    total_spikes = np.sum(spikes_bin[:, i])
    if total_spikes < 1000:
        continue
    angle_bin = scipy.stats.binned_statistic(angle, spikes_bin[:-1, i], statistic='mean', bins=8)
    X = (angle_bin.bin_edges[:-1] + angle_bin.bin_edges[1:]) / 2
    try:
        best_params, _ = scipy.optimize.curve_fit(firing_rate, X, angle_bin.statistic)
    except RuntimeError:
        continue
    tuning = abs(best_params[1] / best_params[0])
    tuning_list.append(tuning)
    neuron_list.append(i)

# A "good" neuron is one that fits the best to the actual tuning graph
# The ratio of g (modulation depth) / b (baseline firing rate) tells us how much the firing rate varies
best_neurons = np.array(neuron_list)[np.argsort(tuning_list)[-6:]]
print(best_neurons)

# Graphing neuron tuning graph and cosine model for 5 "best" neurons
for idx, i in enumerate(best_neurons):
    angle_bin = scipy.stats.binned_statistic(angle, spikes_bin[:-1, i], statistic='mean', bins=8)
    X = (angle_bin.bin_edges[:-1] + angle_bin.bin_edges[1:]) / 2
    plt.subplot(2, 3, idx + 1)
    plt.plot(X, angle_bin.statistic)
    theta_smooth = np.linspace(-np.pi, np.pi, 100)
    best_params = scipy.optimize.curve_fit(firing_rate, X, angle_bin.statistic)
    b, g, phi = best_params[0]
    plt.plot(theta_smooth, firing_rate(theta_smooth, b, g, phi), color='red')
    plt.title(f"Neuron {i}")
    plt.xlabel("Angle (rad)")
    plt.ylabel("Firing Rate")

plt.tight_layout() 
plt.show()

