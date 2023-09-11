from enum import Enum

import numpy as np
import torch

#General constants
EPS_LST_DEFAULT = np.logspace(-3, 1, 20)
LAYER_SOLVER = {'solve_method': "ECOS"}
DTYPE = torch.double
PATIENCE = 5 #TODO (Amit): This parameter is not clear to me

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

#Optimizer constants
ADADELTA    = "Adadelta"
ADAGRAD     = "Adagrad"
ADAM        = "Adam"
ADAMW       = "AdamW"
SPARSEADAM  = "SparseAdam"
ADAMAX      = "Adamax"
ASGD        = "ASGD"
LBFGS       = "LBFGS"
NADAM       = "NAdam"
RADAM       = "RAdam"
RMSPROP     = "RMSprop"
RPROP       = "Rprop"
SGD         = "SGD"
ECOS        = "ECOS"

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

#Lagrangian constants
ETA_LAGRANGIAN_DEFAULT      = 0.05
KAPPA_LAGRANGIAN_DEFAULT    = -0.05


#Train and grid constants
EPS_DEFAULT                 = False
FIXB_DEFAULT                = False
NUM_ITER_DEFAULT            = 45
LR_DEFAULT                  = 0.0001
SCHEDULER_DEFAULT           = True
MOMENTUM_DEFAULT            = 0.8
OPT_DEFAULT                 = SGD
INIT_EPS_DEFAULT            = None
INIT_A_DEFAULT              = None
INIT_B_DEFAULT              = None
SAVE_HISTORY_DEFAULT        = False
SEED_DEFAULT                = 1
INIT_LAM_DEFAULT            = 0.0
INIT_ALPHA_DEFAULT          = -0.01
KAPPA_DEFAULT               = -0.015
TEST_PERCENTAGE_DEFAULT     = 0.2
STEP_LAM_DEFAULT            = 0.1
U_BATCH_PERCENTAGE_DEFAULT    = 0.8
Y_BATCH_PERCENTAGE_DEFAULT    = 1

NUM_YS_DEFAULT              = None