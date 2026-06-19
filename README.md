# Neural Decoder

Decoding 2D hand kinematics from primate motor cortex (M1) population activity.
Building classical and modern neural decoders from scratch and comparing them on the same data. 

This is a self-directed project to develop hands-on experience with machine learning and neural decoding methods, working
through the progression from simple population coding vectors to recursive Bayesian estimation. Each method is derived
from first principles and implemented in NumPy before any library is used to compare.

## Background
Neurons in the primary motor cortex are directionally tuned: each fires fastest for movement in its specific
preferred direction and follows a roughly cosine-shaped relationship with movement angle (Georgopoulos, 1986).
Any single neuron is a noisy predictor, but combining a population of neurons with diverse preferred direction gives us
a more accurate estimator. This is called population coding.

## Dataset
Sabes lab non-human primate reaching dataset (O'Doherty et al.), Zenodo record 3854034.
Session indy_20160627_01.mat: spike times for 96 sorted units and continuous cursor kinematics during a center-out reaching task. 
Stored as MATLAB v7.3 / HDF5, loaded with h5py.

## Methods & Status
Tuning curves + Population vector : Complete
Wiener filter (Ridge Regression) : Complete
Kalman filter (State-space) : In progress
GRU decoder (PyTorch) : Planned

## Phase 1 : Population Vector
Per-neuron cosine tuning curve fit by least squares, population vector decoder summing preferred-direction unit vectors weighted
by firing rate. Documents the failure mode where decoding is biased when preferred directions cluster.

## Phase 2: Wiener Filter
Ridge regression derived from scratch and implemented using NumPy (np.linalg.solve, not explicit inversion here), and verified against sklearn.linear_model.Ridge.
Lagged design matrix with K=10 lags across 96 neurons predicting x/y cursor velocity, chronological 
80/20 train/test split to avoid leakage.

## Results so far
Wiener filter on test data: R² ≈ 0.44-0.61 for x/y-velocity, but i'll try to nudge this up a bit.
