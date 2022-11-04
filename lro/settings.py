import numpy as np
import torch

Adadelta = "Adadelta"
Adagrad = "Adagrad"
Adam = "Adam"
AdamW = "AdamW"
SparseAdam = "SparseAdam"
Adamax = "Adamax"
ASGD = "ASGD"
LBFGS = "LBFGS"
NAdam = "NAdam"
RAdam = "RAdam"
RMSprop = "RMSprop"
Rprop = "Rprop"
SGD = "SGD"
ECOS = "ECOS"

OPTIMIZERS = {
    "Adadelta": torch.optim.Adadelta,
    "Adagrad": torch.optim.Adagrad,
    "Adam": torch.optim.Adam,
    "AdamW": torch.optim.AdamW,
    "SparseAdam": torch.optim.SparseAdam,
    "Adamax": torch.optim.Adamax,
    "ASGD": torch.optim.ASGD,
    "LBFGS": torch.optim.LBFGS,
    "NAdam": torch.optim.NAdam,
    "RAdam": torch.optim.RAdam,
    "RMSprop": torch.optim.RMSprop,
    "Rprop": torch.optim.Rprop,
    "SGD": torch.optim.SGD,
    Adadelta: torch.optim.Adadelta,
    Adagrad: torch.optim.Adagrad,
    Adam: torch.optim.Adam,
    AdamW: torch.optim.AdamW,
    SparseAdam: torch.optim.SparseAdam,
    Adamax: torch.optim.Adamax,
    ASGD: torch.optim.ASGD,
    LBFGS: torch.optim.LBFGS,
    NAdam: torch.optim.NAdam,
    RAdam: torch.optim.RAdam,
    RMSprop: torch.optim.RMSprop,
    Rprop: torch.optim.Rprop,
    SGD: torch.optim.SGD
}

EPS_LST_DEFAULT = np.logspace(-3, 1, 20)
