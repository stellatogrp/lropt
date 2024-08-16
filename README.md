
Learning for Optimization under Uncertainty
=====================

LROPT is a package for decision making under uncertainty. It is based on Python, and built on top of [CVXPY](https://www.cvxpy.org). It allows model optimization problems affected by uncertainty using data. 

## Example

The following code solves a linear programming problem where the objective is to minimize $c^Tx$, while their constraints belong to an ellipsoidal uncertainty set. 

```python3
import cvxpy as cp
import numpy as np
import lropt

# Data
A = np.array([1, 2]) 
b = np.array([5])     

# Variables
x = cp.Variable(2)

# Objective
c = np.array([1, 1])
objective = cp.Minimize(c.T @ x)

# Uncertainty parameter
u = lropt.UncertainParameter(1, uncertainty_set=lropt.Ellipsoidal())

# Constraints
constraints = [
    A @ x <= b + u,
    x >= 0
]

# Problem
prob = lropt.RobustProblem(objective, constraints)
prob.solve()
```

LROPT is not a solver. It relies upon the open source solvers listed [here](https://www.cvxpy.org/tutorial/solvers/index.html#solve-method-options).


## Contributing

If you'd like to add a new example to our library, or implement a new feature,
please get in touch with us first to make sure that your priorities align with
ours. 

## Team

LROPT is  built from the contributions of many
researchers and engineers. A list of people who contributed to the development of LROPT include Irina Wang, Amit Solomon, Bartolomeo Stellato, Cole Becker, Bart Van Parys and Manav Jairam. 

## Citing

If you use LROPT for published work, we encourage you to cite this [paper](https://arxiv.org/abs/2305.19225).