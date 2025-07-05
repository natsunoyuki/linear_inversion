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

from linear_inversion import LinearInversion


@pytest.fixture
def invalid_error_type_config(model_config):
    model_config["error_type"] = "l0"
    return model_config


@pytest.fixture
def polynomial_order_not_int_config(model_config):
    model_config["polynomial_order"] = 1.1
    return model_config


@pytest.fixture
def polynomial_order_not_positive_config(model_config):
    model_config["polynomial_order"] = -1
    return model_config


@pytest.fixture
def use_sgd_not_bool_config(model_config):
    model_config["use_sgd"] = 12345
    return model_config


@pytest.fixture
def sgd_lr_not_positive_config(model_config):
    model_config["sgd_lr"] = -0.01
    return model_config


@pytest.fixture
def sgd_iter_not_int_config(model_config):
    model_config["sgd_iter"] = 100.1
    return model_config


@pytest.fixture
def sgd_iter_not_positive_config(model_config):
    model_config["sgd_iter"] = -100
    return model_config


@pytest.mark.mlmodel
class TestLinearInversionConfig:
    def test_invalid_error_type_value(self, invalid_error_type_config):
        with pytest.raises(AssertionError) as excinfo:
            _ = LinearInversion(**invalid_error_type_config)
        assert "error_type must be from the set ['l1', 'l2']." in str(excinfo.value)


    def test_polynomial_order_not_int(self, polynomial_order_not_int_config):
        with pytest.raises(AssertionError) as excinfo:
            _ = LinearInversion(**polynomial_order_not_int_config)
        assert "polynomial_order must be an integer." in str(excinfo.value)


    def test_polynomial_order_not_positive(self, polynomial_order_not_positive_config):
        with pytest.raises(AssertionError) as excinfo:
            _ = LinearInversion(**polynomial_order_not_positive_config)
        assert "polynomial_order must be positive." in str(excinfo.value)


    def test_use_sgd_not_boolean(self, use_sgd_not_bool_config):
        with pytest.raises(AssertionError) as excinfo:
            _ = LinearInversion(**use_sgd_not_bool_config)
        assert "use_sgd must be boolean." in str(excinfo.value)


    def test_sgd_lr_not_positive(self, sgd_lr_not_positive_config):
        with pytest.raises(AssertionError) as excinfo:
            _ = LinearInversion(**sgd_lr_not_positive_config)
        assert "sgd_lr must be positive." in str(excinfo.value)


    def test_sgd_iter_not_int(self, sgd_iter_not_int_config):
        with pytest.raises(AssertionError) as excinfo:
            _ = LinearInversion(**sgd_iter_not_int_config)
        assert "sgd_iter must be an integer." in str(excinfo.value)


    def test_sgd_iter_not_positive(self, sgd_iter_not_positive_config):
        with pytest.raises(AssertionError) as excinfo:
            _ = LinearInversion(**sgd_iter_not_positive_config)
        assert "sgd_iter must be positive." in str(excinfo.value)
