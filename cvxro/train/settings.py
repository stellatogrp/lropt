from enum import Enum

import numpy as np
import torch

# General constants
RHO_LST_DEFAULT = np.logspace(-3, 1, 20)
LAYER_SOLVER = {"solve_method": "Clarabel", "tol_feas": 1e-7}
TOL = 1e-5
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
    ADADELTA: torch.optim.Adadelta,
    ADAGRAD: torch.optim.Adagrad,
    ADAM: torch.optim.Adam,
    ADAMW: torch.optim.AdamW,
    SPARSEADAM: torch.optim.SparseAdam,
    ADAMAX: torch.optim.Adamax,
    ASGD: torch.optim.ASGD,
    LBFGS: torch.optim.LBFGS,
    NADAM: torch.optim.NAdam,
    RADAM: torch.optim.RAdam,
    RMSPROP: torch.optim.RMSprop,
    RPROP: torch.optim.Rprop,
    SGD: torch.optim.SGD,
}


class TrainerSettings:
    """
    A trainer settings class.
    Contains default settings unless changed.
    Args:
    -------------
        train_size : bool, optional
           If True, train only rho. Default False.
        train_shape: bool, optional
            If True, train both the shape A, b, and size rhos. Default True.
        fixb : bool, optional
            If True, do not train b. Default False.
        num_iter : int, optional
            The total number of gradient steps performed. Default 45.
        num_iter_size : int, optional
            The total number of gradient steps performed for training
            only rho. Default None.
        lr : float, optional
            The learning rate of gradient descent. Default 0.0001.
        lr_size : float, optional
            The learning rate of gradient descent for training only rho. Default 0.0001.
        momentum: float between 0 and 1, optional
            The momentum for gradient descent. Default 0.8.
        optimizer: str or letters, optional
            The optimizer to use for the descent algorithm. Default SGD.
        init_rho : float, optional
            The rho (radius) to initialize :math:`A` and :math:`b`, if passed.
        init_A : numpy array, optional
            Initialization for the reshaping matrix, if passed.
            If not passed, :math:`A` will be initialized as the
            inverse square root of the
            covariance of the data.
        init_b : numpy array, optional
            Initialization for the relocation vector, if passed.
            If not passed, b will be initialized as :math:`\bar{d}`.
        save_history: bool, optional
            Whether or not to save the A and b over the training iterations. Default False.
        init_alpha : float, optional
            The initial alpha value for the CVaR constraint in the outer
            level problem. Default 0.
        eta: float, optional
            The eta value for the CVaR constraint. Default 0.05.
        init_lam : float, optional
            The initial lambda value for the outer level lagrangian function. Default 0.
        init_mu : float, optional
            The initial mu value for the outer level lagrangian function. Default 0.5.
        mu_multiplier : float, optional
            The initial mu multiplier for the outer level lagrangian function. Default 1.01.
        kappa : float, optional
            The target threshold for the outer level CVaR constraint. Default -0.01.
        random_int : bool, optional
            Whether or not to initialize the set with random values. Default False.
        num_random_int : int, optional
            The number of random initializations performed if random_int is True. Default 10.
        test_frequency : int, optional
            The number of iterations before testing results are recorded
        test_percentage : float, optional. Default 10.
            The percentage of data to use in the testing set. Default 0.2.
        seed : int, optional
            The seed to control the random state of the train-test data split. Default 1.
        batch_percentage : float, optional
            The percentage of data to use in each training step. Default 0.1
        solver_args:
            The optional arguments passed to the solver. Default Clarabel.
        parallel : bool, optional
            Whether or not to parallelize the training loops. Default True.
        position: bool, optional
            The position of the tqdm statements for the training loops. Default False.
        scheduler: bool, optional
            Whether or not the learning rate is decreased over steps. Default True.
        lr_step_size: int, optional
            The number of iterations before the learning rate is decreased,
            if scheduler is enabled. Default 500.
        lr_gamma: float, optional
            The multiplier of the lr if the scheduler is enabled. Default 0.1.
        quantiles: tuple, optional
            The lower and upper quantiles of the test values desired. Default (0.25, 0.75)
        aug_lag_update_interval: int, optional
            The number of iterations before the augmented lagrangian parameters
            (lambda, mu) are updated. Default 20.
        lambda_update_threshold: float, optional
            The threshold of CVaR improvement, between 0 and 1, where an update
            to lambda is accepted. Otherwise, mu is updated. Default 0.99.
        lambda_update_max: float, optional
            The maximum allowed lambda value, default 1000.
        max_batch_size: int, optional
            The maximum data batch size allowed for each iteration. Default 30.
        contextual: bool, optional
            Whether or not the learned set is contextual. Default False.
        init_weight: np.array, optional
            The initial weight of the NN model. Default None.
        init_bias: np.array, optional
            The initial bias of the NN model. Default None.
        n_jobs:
            The number of parallel processes. Default 5.
        max_iter_line_search
            The maximum number of times we halve the step size when an
            infeasibility occurs. Default 10.
        x_endind
            If given, the ending index for the context parameters we want to
            use for training the shape parameters. All parameters after this
            index will not be used for training. Default None.
        policy
            The cvxpylayers object for the robust problem. Default is to use
            all variables and parameters (when set to None).
        time_horizon
            The time horizon for the multistage problem. Single-stage problems
            must have a horizon of 1. Default 1.
        batch_size
            The training batch size for multi-stage problems. Default 1.
        test_batch_size
            The testing batch size for multi-stage problems. Default 1.
        simulator
            The self-defined functions for propagating states and constructing
            cost functions for the multi-stage problem. Initialized to a
            default version for the single-stage problem if not given.
            Default None.
        kwargs_simulator
            The extra keyword arguments required for the self-defined simulator
            class. The simulator functions must be defined with these keyword
            arguments as inputs. Default {}.
        multistage
            Flag for whether or not the problem is multistage. Default False.
        predictor
            NN model, Default Linear
        line_search
            Whether or not to perform backtracking line search to choose step sizes
        line_search_mult
            The amount to reduce the step size by if the line search condition is not met
        line_search_threshold
            The threshold (between 0 and 1) for the line search condition
    """

    def __init__(self):
        self.train_size = False
        self.train_shape = True
        self.fixb = False
        self.num_iter = 45  # Used to be "step"
        self.num_iter_size = None
        self.lr = 0.0001
        self.lr_size = 0.001
        self.scheduler = True
        self.momentum = 0.8
        self.optimizer = SGD
        self.init_rho = 1
        self.init_A = None
        self.init_b = None
        self.save_history = False
        self.seed = 1
        self.init_lam = 0.0
        self.init_mu = 0.5
        self.mu_multiplier = 1.01
        self.init_alpha = 0.0
        self.eta = 0.05
        self.kappa = -0.015  # (originally target_cvar)
        self.random_init = False
        self.num_random_init = 10
        self.test_frequency = 10
        self.test_percentage = 0.2
        self.batch_percentage = 0.1
        self.solver_args = LAYER_SOLVER
        self.n_jobs = 5
        self.quantiles = (0.25, 0.75)
        self.lr_step_size = 500
        self.lr_gamma = 0.1
        self.position = False
        self.parallel = True
        self.aug_lag_update_interval = 20
        self.lambda_update_threshold = 0.99
        self.lambda_update_max = 1000
        self.max_batch_size = 30
        self.contextual = False
        self.linear = None
        self.init_weight = None
        self.init_bias = None
        self.x_endind = None
        self.max_iter_line_search = 100  # Times to check feasibility before timeout
        self.policy = None
        self.time_horizon = 1
        self.batch_size = 1
        self.test_batch_size = 1
        self.simulator = None
        self.kwargs_simulator = {}
        self.multistage = False
        self.init_context = None
        self.init_uncertain_param = None
        self.trained_shape = False
        self.predictor = None
        self.obj_scale = 1
        self.line_search_mult = 0.8
        self.line_search_threshold = 1
        self.line_search = True
        self.initialize_predictor = True
        self.validate_percentage = 0.2
        self.validate_frequency = 10
        self.data = None
        self.indices_dict = None
        self.coverage_gamma = 0.5
        self.constraint_cvar = True
        self.delage_coverage = False
        self.target_eta = 0.1
        self.avg_scale = 0
        self._generate_slots()

    def _attr_exists(self, name) -> None:
        """
        This function raises an AttributeError if this object does not have a name property.
        """
        if hasattr(self, "__slots__") and name not in self.__slots__:
            raise AttributeError(f"TrainerSettings has no property called {name}.")

    def _generate_slots(self) -> None:
        """
        This function generates __slots__.
        __slots__ is used to make sure only existing fields can be updated, so the user cannot
        accidently add new fields.
        """
        # Set slots at the class level
        cls = self.__class__
        if not hasattr(cls, "__slots__"):
            cls.__slots__ = list(self.__dict__.keys())

    def __setattr__(self, name, value) -> None:
        """
        This function prevents adding new fields to the object.
        """
        self._attr_exists(name)
        super().__setattr__(name, value)

    def set(self, **kwargs) -> None:
        """
        This function sets all the values of kwargs to the keys.
        It sets only if all the keys are valid.
        Raises AttributeError if a key in kwargs does not exist.
        """
        # Check that all the keys are valid
        for key in kwargs.keys():
            self._attr_exists(key)

        # Assign
        for key, item in kwargs.items():
            super().__setattr__(key, item)


# Create a global default settings instance for use throughout the codebase
DEFAULT_SETTINGS = TrainerSettings()
