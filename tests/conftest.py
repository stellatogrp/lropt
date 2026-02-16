import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (learning/training tests)")
    config.addinivalue_line("markers", "core: marks tests as core robust optimization tests")


@pytest.fixture
def rng():
    """Seeded random number generator for reproducible tests."""
    return np.random.default_rng(seed=1234)
