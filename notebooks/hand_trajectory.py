import h5py 
import matplotlib.pyplot as plt
import numpy as np

raw_data = h5py.File(r'C:\Users\ruben\Desktop\Neuro\Neural-Decoder\data\indy_20160627_01.mat', 'r')
cursor_pos = raw_data['cursor_pos']
x = raw_data["cursor_pos"][0, 0 : 15000]
y = raw_data["cursor_pos"][1, 0 : 15000]

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(cursor_pos[0, :15000].min(), cursor_pos[0, :15000].max())
ax.set_ylim(cursor_pos[1, :15000].min(), cursor_pos[1, :15000].max())

def update(frame):
    line.set_data(cursor_pos[0, :frame], cursor_pos[1, :frame])
    return line,

ani = FuncAnimation(fig, update, frames=range(0, 15000, 50), interval=20)
plt.show()