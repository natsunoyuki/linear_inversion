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

from linear_inversion.least_squares import least_squares, least_squares_sgd
from linear_inversion.l1_norm_inversion import l1_norm_inversion, l1_norm_inversion_sgd


class LinearInversion:
    def __init__(
        self, 
        error_type = "l2", 
        polynomial_order = 1,
        use_sgd = False,
        sgd_lr = 0.01, 
        sgd_iter = 100,
    ):
        """
        Class for linear inversion models.

        Inputs
            error_type: str
                Error type. Choose from "l1" or "l2". "l2" by default.
            polynomial_order: int
                Polynomial order for the data kernel. y = mx has order 1.
                y = mx**2 + nx has order 2. 1 by default.
            use_sgd: bool  
                Flag to use SGD solver or not. Set to False by default. If False
                analytical solvers will be used instead.
            sgd_lr: float
                SGD learning rate. Only works when use_sgd=True.
            sgd_iter: int
                SGD iterations. Only works when sgd_iter=True.
        """
        self.model = None
        
        self.m = None
        assert int(polynomial_order) == polynomial_order, "polynomial_order must be an integer."
        assert polynomial_order > 0, "polynomial_order must be positive."
        self.vander_order = int(polynomial_order + 1)

        assert use_sgd in [True, False], "use_sgd must be boolean."
        self.use_sgd = use_sgd

        assert sgd_lr > 0, "sgd_lr must be positive."
        self.sgd_lr = sgd_lr

        assert round(sgd_iter) == sgd_iter, "sgd_iter must be an integer."
        assert sgd_iter > 0, "sgd_iter must be positive."
        self.sgd_iter = sgd_iter

        assert error_type in ["l1", "l2"], "error_type must be from the set ['l1', 'l2']."
        self.error_type = error_type
        if self.error_type == "l2":
            if self.use_sgd is True:
                self.model = least_squares_sgd
            else:
                self.model = least_squares
        elif self.error_type == "l1":
            if self.use_sgd is True:
                self.model = l1_norm_inversion_sgd
            else:
                self.model = l1_norm_inversion


    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sd: np.ndarray = None, 
        polynomial_order: int = None, 
        sgd_lr: float = None, 
        sgd_iter: int = None
    ) -> np.ndarray:
        """
        Fits the linear inversion model on the data (X, y).

        Inputs
            X: ndarray
                Data kernel/measurements.
            y: ndarray
                Target variables.
            sd: ndarray
                Standard deviation of the elements of y.
            polynomial_order: int
                Polynomial order of the linear inversion model.
            sgd_lr: float
                SGD learning rate. Only works when use_sgd=True.
            sgd_iter: int
                SGD iterations. Only works when sgd_iter=True.
        Outputs
            m: ndarray
                Model parameters.
        """
        assert self.model is not None

        G = self.make_data_kernel(X, polynomial_order)
        self.m = None

        if self.use_sgd is True:
            if sgd_lr is None:
                sgd_lr = self.sgd_lr
            if sgd_iter is None:
                sgd_iter = self.sgd_iter
            self.m = self.model(G, y, sgd_lr, sgd_iter)
        else:
            if self.error_type == "l2":
                self.m = self.model(G, y)
            elif self.error_type == "l1":
                self.m = self.model(G, y, sd)
        
        return self.m


    def predict(
        self, 
        X: np.ndarray, 
        polynomial_order: int=None
    ) -> np.ndarray:
        """
        Makes predictions given some measurements X.

        Inputs
            X: ndarray
                Data kernel/measurements.
            polynomial_order: int
                Polynomial order of the linear inversion model.
        Outputs
            m: ndarray
                Model parameters.
        """
        assert self.m is not None
        G = self.make_data_kernel(X, polynomial_order)
        return np.dot(G, self.m.reshape(-1, 1)).squeeze()


    def make_data_kernel(
        self, 
        X: np.ndarray, 
        polynomial_order: int=None
    ) -> np.ndarray:
        """
        Creates the data kernel from the input independent variables X.

        Inputs
            X: ndarray
                Data kernel/measurements.
            polynomial_order: int
                Polynomial order of the linear inversion model.
        Outputs
            G: ndarray
                Vander matrix of the data kernel representing the correct
                polynomial order of the linear inversion model.
        """
        if polynomial_order is None:
            vander_order = self.vander_order
        else:
            vander_order = polynomial_order + 1

        if len(X.shape) == 1:
            if vander_order > 1:
                return np.vander(X, vander_order)
            else:
                return X.reshape(-1, 1)
        else:
            return X


    def set_model_parameters(
        self,
        m: np.ndarray
    ):
        """
        Sets the model parameters with the given values.

        Inputs
            m: ndarray
                Model parameters
        """
        self.m = m
