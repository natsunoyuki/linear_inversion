import numpy as np


def r2(d: np.ndarray, d_pred: np.ndarray) -> float:
    return 1 - np.sum((d - d_pred)**2) / np.sum((d - np.mean(d))**2)
