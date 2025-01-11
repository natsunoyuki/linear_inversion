import numpy as np

# Local imports.
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
        self.model = None
        
        self.m = None
        self.vander_order = polynomial_order + 1

        self.sgd_lr = sgd_lr
        self.sgd_iter = sgd_iter

        self.error_type = error_type.lower()
        self.use_sgd = use_sgd
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


    def fit(self, X, y, sd = None, polynomial_order = None, sgd_lr = None, sgd_iter = None):
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


    def predict(self, X, polynomial_order = None):
        assert self.m is not None
        G = self.make_data_kernel(X, polynomial_order)
        return np.dot(G, self.m.reshape(-1, 1))


    def make_data_kernel(self, X, polynomial_order = None):
        if polynomial_order is None:
            vander_order = self.polynomial_order
        else:
            vander_order = polynomial_order + 1

        return np.vander(X, vander_order)
