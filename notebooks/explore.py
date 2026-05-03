import h5py
import numpy as np

f = h5py.File(r'C:\Users\ruben\Desktop\Neuro\Neural-Decoder\data\indy_20160627_01.mat', 'r')
print(f["t"][0, -1])