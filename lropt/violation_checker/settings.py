import numpy as np

VIOLATION_TOL = 1e-5 # Tolerance for constraint violation.
NO_BATCH = np.nan
MAX_ITER_LINE_SEARCH = int(1e1) # Amount of times to check feasibility before timing outs
