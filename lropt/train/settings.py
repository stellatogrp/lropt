from enum import Enum

import numpy as np
import torch

# General constants
RHO_LST_DEFAULT = np.logspace(-3, 1, 20)
LAYER_SOLVER = {'solve_method': "Clarabel", 'tol_feas':1e-10}
DTYPE = torch.double

"""
Different types of MRO:
#NO_MRO:
    No MRO
#DIFF_A_UNINIT:
    Different A for each k and uninitialized
#DIFF_A_INIT:
    Different A for each k, initialized with a different matrix for each k
#SAME_A:
    same A for each k
"""
MRO_CASE = Enum("MRO_CASE", "NO_MRO DIFF_A_UNINIT DIFF_A_INIT SAME_A")

# Optimizer constants
ADADELTA = "Adadelta"
ADAGRAD = "Adagrad"
ADAM = "Adam"
ADAMW = "AdamW"
SPARSEADAM = "SparseAdam"
ADAMAX = "Adamax"
ASGD = "ASGD"
LBFGS = "LBFGS"
NADAM = "NAdam"
RADAM = "RAdam"
RMSPROP = "RMSprop"
RPROP = "Rprop"
SGD = "SGD"
ECOS = "ECOS"

OPTIMIZERS = {
    ADADELTA:   torch.optim.Adadelta,
    ADAGRAD:    torch.optim.Adagrad,
    ADAM:       torch.optim.Adam,
    ADAMW:      torch.optim.AdamW,
    SPARSEADAM: torch.optim.SparseAdam,
    ADAMAX:     torch.optim.Adamax,
    ASGD:       torch.optim.ASGD,
    LBFGS:      torch.optim.LBFGS,
    NADAM:      torch.optim.NAdam,
    RADAM:      torch.optim.RAdam,
    RMSPROP:    torch.optim.RMSprop,
    RPROP:      torch.optim.Rprop,
    SGD:        torch.optim.SGD
}

# Lagrangian constants
ETA_LAGRANGIAN_DEFAULT = 0.05
KAPPA_LAGRANGIAN_DEFAULT = -0.05


# Train and grid constants
TRAIN_SIZE_DEFAULT = False
TRAIN_SHAPE_DEFAULT = True
FIXB_DEFAULT = False
NUM_ITER_DEFAULT = 45
LR_DEFAULT = 0.0001
SCHEDULER_STEPLR_DEFAULT = True
MOMENTUM_DEFAULT = 0.8
OPT_DEFAULT = SGD
INIT_RHO_DEFAULT = None
INIT_A_DEFAULT = None
INIT_B_DEFAULT = None
SAVE_HISTORY_DEFAULT = False
SEED_DEFAULT = 1
INIT_LAM_DEFAULT = 0.0
INIT_MU_DEFAULT = 0.5
MU_MULTIPLIER_DEFAULT = 1.01
INIT_ALPHA_DEFAULT = -0.01
KAPPA_DEFAULT = -0.015
TEST_PERCENTAGE_DEFAULT = 0.2
STEP_LAM_DEFAULT = 0.1
RANDOM_INIT_DEFAULT = False
NUM_RANDOM_INIT_DEFAULT = 10
TEST_FREQUENCY_DEFAULT = 10
BATCH_PERCENTAGE_DEFAULT = 0.1
LAMBDA_UPDATE_MAX = 1000
MAX_BATCH_SIZE = 30
POSITION = False
PARALLEL = True
NUM_ITER_SIZE_DEFAULT = None
LR_SIZE_DEFAULT = None
N_JOBS = 5
LR_STEP_SIZE = 500
LR_GAMMA = 0.1
QUANTILES = (0.25, 0.75)
UPDATE_INTERVAL = 20
LAMBDA_UPDATE_THRESHOLD = 0.99
NUM_YS_DEFAULT = None
TOLERANCE_DEFAULT = 1e-5
NEWDATA_DEFAULT = None
CONTEXTUAL_DEFAULT = False
CONTEXTUAL_LINEAR_DEFAULT = None
CONTEXTUAL_WEIGHT_DEFAULT = None
CONTEXTUAL_BIAS_DEFAULT = None
INIT_RHO_DEFAULT_GRID = 1
X_ENDIND_DEFAULT = None
TIME_HORIZON_DEFAULT = 1
POLICY_DEFAULT = None
BATCH_SIZE_DEFAULT = 1
TEST_BATCH_SIZE_DEFAULT = 1
SIMULATOR_DEFAULT = None
KWARGS_SIM_DEFAULT = {}
MULTISTAGE_DEFAULT = False
