import numpy as np
from scipy.optimize import linprog

# Local imports.
from linear_inversion.least_squares import least_squares


def l1_norm_inversion(G: np.ndarray, d: np.ndarray, sd = None) -> np.ndarray:
    """
    Linear inversion using L1 norm error instead of mean squared error for
    over determined problems.
    The inversion problem is transformed into a linear programming problem
    and solved using the linprog() function from scipy.optimize.
    See Geophysical Data Analysis: Discrete Inverse Theory MATLAB Edition
    Third Edition by William Menke pages 153-157 for more details.
    
    Inputs
        G: np.array
        d: np.array
        sd: np.array
        
    Returns
        mest_l1: np.array
    """
    # If the std of the measurement d was not provided,
    # set it to 1.
    if sd is None:
        sd = np.ones(len(d))
    
    N, M = np.shape(G)
    L = 2 * M + 3 * N

    # 1. Create f containing the inverse data std:
    f = np.zeros(L)
    f[2*M:2*M+N] = 1.0 / sd

    # Make Aeq and beq for the equality constraints:
    Aeq = np.zeros([2*N, L])
    beq = np.zeros(2*N)
    
    Aeq[:N, :M] = G
    Aeq[:N, M:2*M] = -G
    Aeq[:N, 2*M:2*M+N] = -np.eye(N)
    Aeq[:N, 2*M+N:2*M+2*N] = np.eye(N)
    beq[:N] = d
    
    Aeq[N:2*N, :M] = G
    Aeq[N:2*N, M:2*M] = -G
    Aeq[N:2*N, 2*M:2*M+N] = np.eye(N)
    Aeq[N:2*N, 2*M+2*N:2*M+3*N] = -np.eye(N)
    beq[N:2*N] = d
    
    # Make A and b for the >=0 constraints:
    A = np.zeros([L+2*M, L])
    b = np.zeros(L+2*M)
    A[:L, :] = -np.eye(L)
    b[:L] = np.zeros(L)
    
    A[L:L+2*M] = np.eye(2*M, L)
    # For this example, we use the least squares solution
    # as the upper bound for the model parameters.
    mls = least_squares(G, d)
    mupperbound = 10 * np.max(np.abs(mls))
    b[L:L+2*M] = mupperbound
    
    res = linprog(f, A, b, Aeq, beq)
    
    # The output res = [m1, m2, alpha, x1, x2]. Extract m1 and m2
    # and calculate the model parameters using m = m1 - m2.
    mest_l1 = res['x'][:M] - res['x'][M:2*M]
    return mest_l1


def l1_norm_inversion_sgd(
    G: np.ndarray, 
    d: np.ndarray, 
    eta: float = 0.01, 
    n_iter: int = 100, 
    return_loss: bool = False,
) -> np.ndarray:
    """
    L1 norm inversion numerical solver using stochastic gradient descent.
    """
    m = np.random.normal(size = G.shape[1])
    losses = []

    for i in range(n_iter):
        d_pred = np.dot(G, m)
        loss = d - d_pred
        # m = m + eta * 2.0 * np.dot(G.T, loss) / G.shape[0]
        m = m + eta * np.dot(G.T, np.abs(d - d_pred) / (d - d_pred)) / G.shape[0]
        losses.append(np.mean(np.abs(loss)))

    if return_loss is True:
        return m, losses
    else:
        return m
