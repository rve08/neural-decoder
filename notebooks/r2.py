import numpy as np


# R² model evaluation function
def r2(y_test, y_predicted):
    diff = y_test - y_predicted
    mean_diff = y_test - np.mean(y_test, axis=0)
    numerator = np.sum(diff * diff, axis=0)
    denominator = np.sum(mean_diff * mean_diff, axis=0)
    return 1 - numerator / denominator
