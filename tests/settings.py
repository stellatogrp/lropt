import cvxpy as cp

TESTS_RTOL = 1e-04
TESTS_ATOL = 1e-04
SOLVER = cp.CLARABEL
SOLVER_SETTINGS = { "equilibrate_enable": False, "verbose": False }
