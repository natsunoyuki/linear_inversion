# Linear Inversion
Creates a linear model with coefficients w = (w_1, …, w_M) to minimize the loss between the data and the predictions.

# Installation
Install this repository directly from GitHub:
```bash
!pip install git+https://github.com/natsunoyuki/linear_inversion
```

Alternatively, clone this repository, and install locally with a virtual environment.
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
