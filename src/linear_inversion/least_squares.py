# Copyright 2025 Y Natsume.
#
# linear_inversion is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.
#
# linear_inversion is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# linear_inversion. If not, see <https://www.gnu.org/licenses/>.

import numpy as np


def least_squares(G: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Analytical linear least squares linear inversion.

    Inputs
        G: ndarray
            Input data/data kernel/Green function. Must be a vander matrix.
        d: ndarray
            Measured variable/target variable.
    Outputs
        m: ndarray
            Linear inversion model parameters.
    """
    m = np.dot(np.linalg.inv(np.dot(G.T, G)), G.T)
    m = np.dot(m, d)
    return m


def svd_inversion(G: np.ndarray, d: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """
    Analytical linear inversion using singular value decomposition.

    Inputs
        G: ndarray
            Input data/data kernel/Green function. Must be a vander matrix.
        d: ndarray
            Measured variable/target variable.
        tol: float
            Minimum threshold value required for the smallest SVD eigenvalue
            with respect to the largest SVD eigenvalue. Set to 0.01 by default.
    Outputs
        m: ndarray
            Linear inversion model parameters.
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

    Inputs
        G: array
            Input data/data kernel/Green function. Must be a vander matrix.
        d: array
            Measured variable/target variable.
        eta: float
            SGD learning rate. Set to 0.01 by default.
        n_iter: int
            SGD steps. Set to 100 by default.
        return_loss: bool
            Flag to return the loss values together with the predictions. 
            False by default.
    Outputs
        m: array
            Linear inversion model parameters.
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
