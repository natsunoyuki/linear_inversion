import numpy as np


def least_squares(G: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Analytical linear least squares linear inversion.
    Inputs
        G: array
        d: array
    Outputs
        m: array
    """
    m = np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)
    m = np.dot(m, d)
    return m


def svd_inversion(G: np.ndarray, d: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """
    Analytical linear inversion using singular value decomposition.
    Inputs
        G: array
        d: array
    Outputs
        m: array
    """
    U, S, Vh = np.linalg.svd(G, full_matrices=True)
    
    Sp = S[S >= np.max(S) * tol]
    p = len(Sp)
    Up = U[:, :p]
    Vhp = Vh[:p, :]

    Lp = np.diag(Sp)
    G_inv = np.dot(Vhp.T, np.dot(np.linalg.inv(Lp), Up.T))
    m = np.dot(G_inv, d)
    return m


def least_squares_sgd(
    G: np.ndarray, 
    d: np.ndarray, 
    eta: float = 0.01, 
    n_iter: int = 100, 
    return_loss: bool = False,
) -> np.ndarray:
    """
    Least squares inversion numerical solver using stochastic gradient descent.
    """
    m = np.random.normal(size = G.shape[1])
    losses = []

    for i in range(n_iter):
        d_pred = np.dot(G, m)
        loss = d - d_pred
        m = m + eta * 2.0 * np.dot(G.T, loss) / G.shape[0]
        losses.append(np.mean(loss ** 2))

    if return_loss is True:
        return m, losses
    else:
        return m
