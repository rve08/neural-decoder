import numpy as np


def build_design(spikes, target, K):
    X = []
    y = []
    for t in range(K, spikes.shape[0] - 1):
        X.append(spikes[t - K : t].flatten())
        y.append(target[t])
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    return X, y


""" This builds the lagged design matrix for ridge regression.
    Parameters :
            - spikes : (T, N) array of spike counts per time bin
            - target : (T, 2) array of cursor velocity per bin
            - K : number of history bins to stack in a feature row
    Returns :
            - X : (T - K - 1, N * K) design matrix
            - y: (T - K - 1, 2) target"""
