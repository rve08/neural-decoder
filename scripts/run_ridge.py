from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge

from src.features import build_design
from src.metrics import r2
from src.ridge import ridge


def main():
    data_dir = Path(__file__).parent.parent / "data"
    spikes = np.load(data_dir / "spikes_bins_50ms.npy")
    cursor = np.load(data_dir / "cursor_bins.npy")
    velocity = np.diff(cursor, axis=1).T
    X, y = build_design(spikes, velocity, K=10)

    # Dividing data into training and testing data: 80% training, 20% test
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

    # K-fold Cross Validation
    # Splitting data into training and validation sets
    split = int(0.8 * X.shape[0])
    X_cv, y_cv = X[:split], y[:split]  # cross-validation pool
    X_test, y_test = X[split:], y[split:]  # final test set
    # Creating arrays representing K number of folds (5 in this case)
    k = 5
    X_folds = np.array_split(X_cv, k)
    y_folds = np.array_split(y_cv, k)
    candidate_lambdas = np.logspace(-3, 4, 13)  # Set of log-spaced lambda values
    r2_avg = []  # List of average R-squared values for x and y speed
    r2_single = []

    for i in candidate_lambdas:
        r2_folds = []
        for j in range(k):
            validation_X = X_folds[j]
            validation_y = y_folds[j]
            training_X = np.concatenate(
                [X_folds[m] for m in range(k) if m != j], axis=0
            )
            training_y = np.concatenate(
                [y_folds[m] for m in range(k) if m != j], axis=0
            )
            beta = ridge(training_X, training_y, i)
            predicted_y = validation_X @ beta
            r2_val = r2(validation_y, predicted_y)
            r2_folds.append(r2_val)
        mean_r2 = np.mean(r2_folds, axis=0)
        r2_avg.append(mean_r2)
    for p in r2_avg:
        r2_singlescore = 0.5 * (p[0] + p[1])
        r2_single.append(r2_singlescore)
    best_lam = candidate_lambdas[np.argmax(r2_single)]
    # Refitting model with best lambda (regularization) value
    beta_best = ridge(X_cv, y_cv, best_lam)
    y_pred = X_test @ beta_best
    final_r2 = r2(y_test, y_pred)
    print(final_r2)

    # Plotting predicted x and y velocity compared to actual velocity
    start = 2000
    n = 400
    vy_prediction = y_pred[start : start + n, 1]
    vx_prediction = y_pred[start : start + n, 0]
    t = np.arange(n) * 0.05
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    axes[0].plot(t, y_test[start : start + n, 0], label="actual")
    axes[0].plot(t, vx_prediction, label="predicted")
    axes[1].plot(t, y_test[start : start + n, 1], label="actual")
    axes[1].plot(t, vy_prediction, label="predicted")
    axes[0].set_ylabel("x velocity")
    axes[1].set_ylabel("y velocity")
    axes[1].set_xlabel("time (s)")
    axes[0].legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
