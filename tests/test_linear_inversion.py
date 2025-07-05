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

import pytest
import numpy.testing as npt
import numpy as np
from linear_inversion import LinearInversion


@pytest.fixture
def analytical_l2_config(model_config):
    model_config["polynomial_order"] = 2
    return model_config


@pytest.fixture
def linprog_l1_config(analytical_l2_config):
    analytical_l2_config["error_type"] = "l1"
    return analytical_l2_config


@pytest.fixture
def sgd_l2_config(analytical_l2_config):
    analytical_l2_config["use_sgd"] = True
    return analytical_l2_config


@pytest.fixture
def sgd_l1_config(sgd_l2_config):
    sgd_l2_config["error_type"] = "l1"
    return sgd_l2_config


def load_regression_data(file_path):
    X = np.genfromtxt(file_path, delimiter=",")
    y = X[:, 1]
    X = X[:, 0]
    return X, y


def load_regression_model_parameters(file_path):
    return np.genfromtxt(file_path, delimiter=",")


@pytest.mark.mlmodel
class TestL2LinearInversion:
    def test_l2_inversion_analytical_model(
        self, analytical_l2_config, noisy_regression_data, l2_model_parameters,
    ):
        X, y = load_regression_data(noisy_regression_data)
        m = load_regression_model_parameters(l2_model_parameters)

        model = LinearInversion(**analytical_l2_config)
        np.random.seed(0)
        model.fit(X, y)
        
        npt.assert_allclose(model.m, m, atol=1e-3)
        

    def test_l2_inversion_sgd(
        self, sgd_l2_config, noisy_regression_data, l2_model_parameters,
    ):
        X, y = load_regression_data(noisy_regression_data)
        m = load_regression_model_parameters(l2_model_parameters)

        model = LinearInversion(**sgd_l2_config)
        np.random.seed(0)
        model.fit(X, y)
        
        npt.assert_allclose(model.m, m, atol=1e-3)


    def test_l1_inversion_linprog_model(
        self, linprog_l1_config, noisy_regression_data, l1_model_parameters,
    ):
        X, y = load_regression_data(noisy_regression_data)
        m = load_regression_model_parameters(l1_model_parameters)

        model = LinearInversion(**linprog_l1_config)
        np.random.seed(0)
        model.fit(X, y)
        
        npt.assert_allclose(model.m, m, atol=1e-3)
        

    def test_l1_inversion_sgd(
        self, sgd_l1_config, noisy_regression_data, l1_model_parameters,
    ):
        X, y = load_regression_data(noisy_regression_data)
        m = load_regression_model_parameters(l1_model_parameters)

        model = LinearInversion(**sgd_l1_config)
        np.random.seed(0)
        model.fit(X, y)
        
        npt.assert_allclose(model.m, m, atol=1e-3)
