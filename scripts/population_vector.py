"""
Population vector decoder
-------------------------
Decodes movement direction from M1 neuron population activity
by summing each neuron's preferred direction weighted by the
current firing rate.
-------------------------
Produces pretty poor results (as expected), because of the
non-uniform preffered direction distribution in the used data.
This creates a bias towards the northeast direction.

"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path(__file__).parent.parent / "data"
phi_list = np.load(data_dir / 'phi_list.npy')
spikes_bin = np.load(data_dir / 'spikes_bins_50ms.npy')

# Population Vector computation
# For each time bin (50ms), we compute the weighted sum of preferred
# direction vectors. We sum over all 96 neurons.
movement_x = []
movement_y = []
for i in range(len(spikes_bin)):
    pop_vector_x = (np.cos(phi_list) * spikes_bin[i, :]).sum()
    pop_vector_y = (np.sin(phi_list) * spikes_bin[i, :]).sum()
    movement_x.append(pop_vector_x)
    movement_y.append(pop_vector_y)

# Decoded direction in radians (per time bin)
pop_vector = np.arctan2(movement_y, movement_x)

# Actual movement direction (computed from cursor_pos time series derivative)
actual_angle = np.load(data_dir / 'actual_movement_angle.npy')

# Plotting decoded and actual movement direction
plt.figure()
plt.plot(pop_vector[:500], label='Population Vector')
plt.plot(actual_angle[:500], label='Actual movement direction')
plt.xlabel('Time bin')
plt.ylabel('Angle (radians)')
plt.title('Population Vector Decoder: actual vs predicted direction')
plt.legend()
plt.tight_layout()
plt.show()

# Angular error histogram
# A good decoder would show a small peak at about 0
# The flat distribution here says the decoder is not tracking
# actual movement correctly.
error = actual_angle[:500] - pop_vector[:500]
plt.figure()
plt.hist(error, bins=20)
plt.xlabel('Angular Error (radians)')
plt.ylabel('Count')
plt.title('Population Vector angular error distribution')
plt.tight_layout()
plt.show()