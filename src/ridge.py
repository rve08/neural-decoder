import numpy as np


def ridge(X, y, lam):
    A = (np.transpose(X)) @ X + lam * np.eye(X.shape[1])
    b = np.transpose(X) @ y
    beta = np.linalg.solve(A, b)
    return beta


if __name__ == "__main__":
    np.random.seed(42)

    n = 200  # observations
    p = 10  # features

    X = np.random.randn(n, p)
    beta_true = np.random.randn(p)
    noise = 0.1 * np.random.randn(n)
    y = X @ beta_true + noise
    lam = 0.01
    print(np.linalg.norm(ridge(X, y, lam) - beta_true))
