# Linear Inversion
Creates a linear model with coefficients w = (w_1, …, w_M) to minimize the loss between the data and the predictions.

# Installation
Install this repository directly from GitHub:
```bash
!pip install git+https://github.com/natsunoyuki/linear_inversion
```

Alternatively, clone this repository, and install locally (in developer mode) with a virtual environment.
```bash
git clone https://github.com/natsunoyuki/linear_inversion
cd linear_inversion

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools
pip install -e .
```

# Basic Usage
```python
import numpy as np
from linear_inversion import LinearInversion

# Create a simple linear dataset with an outlier.
x = np.arange(0, 12, 1)
y = np.pi * x + np.exp(1)
# Add an outlier to the data.
y[-1] = y[-1] + 20

# Analytical linear inversion with the L2 error (analytical least squares).
linear_inv = LinearInversion(error_type = "l2", use_sgd = False)
m = linear_inv.fit(x, y)
print(m) 
# array([3.91082342, 0.15417926])

# Numerical (stochastic gradient descent) linear inversion with the L2 error.
linear_inv = LinearInversion(error_type = "l2", use_sgd = False)
m = linear_inv.fit(x, y, sgd_lr = 0.01, sgd_iter = 10000)
print(m) 
# array([3.91082342, 0.15417926])

# Linear inversion with the L1 error using linear programming.
linear_inv = LinearInversion(error_type = "l1", use_sgd = False)
m = linear_inv.fit(x, y)
print(m)
# array([3.14159265, 2.71828183])

# Stochastic gradient descent linear inversion with the L1 error.
linear_inv = LinearInversion(error_type = "l1", use_sgd = True)
m = linear_inv.fit(x, y, sgd_lr = 0.01, sgd_iter = 10000)
print(m)
# array([3.16194501, 2.70066348])
```


# Tests
To ensure that `linear_inversion` works consistently across different environments, various tests have been implemented. To run them, execute:
```bash
pytest tests/
```


# API Reference
## LinearInversion
```
LinearInversion(
    error_type="l2", 
    polynomial_order=1, 
    use_sgd=False, 
    sgd_lr=0.01, 
    sgd_iter=100
)

Creates a linear inversion model object.

Arguments
---------
error_type: str, optional
    The error type. "l2" for the L2 norm error (least squares),
    or "l1" for the L1 norm error.
    The "l1" norm error assumes an exponential distribution while 
    the "l2" error assumes a Gaussian distribution.
    As a result the "l1" norm error is more robust to outliers 
    than the "l2" norm error.
polynomial_order: int, optional
    Polynomial order of the inversion kernel, used to create a 
    vander matrix from the explanatory features `X`.
    The order of the vander matrix == polynomial_order + 1.
    Only used if the explanatory features `X` is 1-dimensional, 
    i.e. len(X.shape) == 1.
use_sgd: bool, optional
    Flag for stochastic gradient descent solver. 
    If False, an analytical solver is used for the "l2" norm error,
    and a linear programming solver is used for the "l1" norm error.
sgd_lr: float, optional
    Stochastic gradient descent learning rate.
sgd_iter: int, optional
    Stochastic gradient descent iterations.


LinearInversion.fit(
    X, 
    y, 
    sd = None, 
    polynomial_order = None, 
    sgd_lr = None, 
    sgd_iter = None
)

Fits a linear model with coefficients w = (w_1, …, w_M) to minimize 
the loss between the data and the predictions.

Arguments
---------
X: ndarray
    Array of explanatory features.
    If the explanatory features `X` is 1-dimensional, 
    i.e. len(X.shape) == 1, `polynomial_order` will be used to 
    create a vander matrix internally for model fitting.
y: ndarray
    1-dimensional array of dependent features.
polynomial_order: int, optional
    Setting this value will overwrite the original set value 
    only for this function call.
    Polynomial order of the inversion kernel, used to create a 
    vander matrix from the explanatory features `X`.
    The vander order of the resulting matrix == polynomial_order + 1.
    Only used if the explanatory features `X` is 1-dimensional,
    i.e. len(X.shape) == 1.
sgd_lr: float, optional
    Setting this value will overwrite the original set value 
    only for this function call.
    Stochastic gradient descent learning rate.
sgd_iter: int, optional
    Setting this value will overwrite the original set value 
    only for this function call.
    Stochastic gradient descent iterations.


LinearInversion.predict(
    X, 
    polynomial_order = None
)

Uses the trained model with coefficients w = (w_1, …, w_M) 
to make new predictions on new explanatory features `X`.

Arguments
---------
X: ndarray
    Array of explanatory features.
    If the explanatory features `X` is 1-dimensional, 
    i.e. len(X.shape) == 1, `polynomial_order` will be used to 
    create a vander matrix internally for model fitting.
polynomial_order: int, optional
    Setting this value will overwrite the original set value 
    only for this function call.
    Polynomial order of the inversion kernel, used to create a 
    vander matrix from the explanatory features `X`.
    The order of the vander matrix == polynomial_order + 1.
    Only used if the explanatory features `X` is 1-dimensional, 
    i.e. len(X.shape) == 1.


LinearInversion.make_data_kernel(
    X, 
    polynomial_order = None
)

Makes the inversion kernel from the explanatory features `X`. 
If `X` is 1-dimensional, a vander matrix with order polynomial_order + 1 
will be created. If `X` is 2-dimensional, `X` will be used as the 
inversion kernel directly. 

Arguments
---------
X: ndarray
    Array of explanatory features.
    If the explanatory features `X` is 1-dimensional, 
    i.e. len(X.shape) == 1, `polynomial_order` will be used to 
    create a vander matrix internally for model fitting.
polynomial_order: int, optional
    Setting this value will overwrite the original set value 
    only for this function call.
    Polynomial order of the inversion kernel, used to create a 
    vander matrix from the explanatory features `X`.
    The order of the vander matrix == polynomial_order + 1.
    Only used if the explanatory features `X` is 1-dimensional, 
    i.e. len(X.shape) == 1.
```