import unittest

import cvxpy as cp
import numpy as np

from lropt.robust_problem import RobustProblem
from lropt.uncertain import UncertainParameter
from lropt.uncertainty_sets.box import Box


def generate_data(n, N):
    """Generate synthetic portfolio data."""
    data = np.zeros((N, n))  # Preallocate date
    beta = [i / n for i in range(n)]  # Linking factors

    for sample_idx in range(N):
        # Common market factor, mean 3%, sd 5%, truncate at +- 3 sd
        mu, std = 0.03, 0.05
        z = np.random.normal(mu, std)
        z = np.minimum(np.maximum(z, mu - 3 * std), mu + 3 * std)

        for asset_idx in range(n):
            # Idiosyncratic contribution, mean 0%, sd 5%, truncated at +- 3 sd
            mu_id, std_id = 0.00, 0.05
            asset = np.random.normal(mu_id, std_id)
            asset = np.minimum(np.maximum(asset, mu_id - 3 * std_id),
                               mu_id + 3 * std_id)

            data[sample_idx, asset_idx] = beta[asset_idx] * z + asset

    return 100 * data


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.returns = generate_data(5, 100)

    def test_portfolio_box(self):
        n = self.returns.shape[1]

        # Uncertainty
        r_bar = self.returns.mean(axis=0)
        sigma = self.returns.std(axis=0)

        A = np.diag(sigma)
        b = r_bar

        r = UncertainParameter(n, uncertainty_set=Box(rho=1.2))

        x = cp.Variable(n)
        constraints = [cp.sum(x) == 1,
                       x >= 0]

        t = cp.Variable()  # Hack to add the objective
        constraints += [(A @ r + b) @ x <= -t]

        objective = cp.Maximize(t)
        problem = RobustProblem(objective, constraints)
        problem.solve()

        print(x.value)
