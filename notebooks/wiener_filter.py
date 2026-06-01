import os
import time

import numpy as np
from r2 import r2
from ridge_scratch import ridge
from sklearn.linear_model import Ridge

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spikes = np.load(os.path.join(BASE, "data", "spikes_bins_50ms.npy"))
cursor = np.load(os.path.join(BASE, "data", "cursor_bins.npy"))

K = 10

# Target matrix for ridge regression function
target_mat = np.transpose(np.diff(cursor, axis=1))

# Design matrix for ridge regression
print("Building the design matrix...")
t0 = time.time()
design_mat = []
for t in range(K, spikes.shape[0] - 1):
    feature_row = list(spikes[t - K : t,].flatten())
    design_mat.append(feature_row)
design_mat = np.array(design_mat, dtype=np.float64)
print(f"design matrix done in {time.time() - t0:.1f}s, shape {design_mat.shape}")

# Dividing data into training and testing data: 80% training, 20% test
X = design_mat
y = target_mat[K:]
split = int(0.8 * X.shape[0])
X_train = X[0:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# Training parameters + predicting movement speeds
lam = 1.0
beta_train = ridge(X_train, y_train, lam)
y_predicted = X_test @ beta_train
print(f"Own model R²: {r2(y_test, y_predicted)}")

# SKLearn model comparison
model = Ridge(alpha=lam, fit_intercept=False)
model.fit(X_train, y_train)
y_pred_sklearn = model.predict(X_test)
print(f"SKLearn R²: {r2(y_test, y_pred_sklearn)}")
