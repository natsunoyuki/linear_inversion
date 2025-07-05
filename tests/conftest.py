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
from pathlib import Path
import pytest
import gc


BASE_DIR = Path(__file__).resolve().parents[1] / "linear_inversion/"
TEST_DATA_DIR = BASE_DIR.parent / "tests" / "test_data"

NOISY_REGRESSION_DATA = ["data_001.csv"]
L1_MODEL_PARAMETERS = ["l1_m_001.csv"]
L2_MODEL_PARAMETERS = ["l2_m_001.csv"]


@pytest.fixture
def model_config():
    return {
        "error_type": "l2", 
        "polynomial_order": 1,
        "use_sgd": False,
        "sgd_lr": 0.01, 
        "sgd_iter": 10000,
    }


@pytest.fixture(params=NOISY_REGRESSION_DATA)
def noisy_regression_data(request):
    yield TEST_DATA_DIR / request.param
    gc.collect()


@pytest.fixture(params=L2_MODEL_PARAMETERS)
def l2_model_parameters(request):
    yield TEST_DATA_DIR / request.param
    gc.collect()


@pytest.fixture(params=L1_MODEL_PARAMETERS)
def l1_model_parameters(request):
    yield TEST_DATA_DIR / request.param
    gc.collect()
