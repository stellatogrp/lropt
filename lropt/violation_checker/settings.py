import numpy as np

VIOLATION_TOL = 1e-7 # Tolerance for constraint violation.
NO_BATCH = np.nan
VIOLATION_CHECK_TIMEOUT = int(1e1) # Amount of times to check feasibility before timing out