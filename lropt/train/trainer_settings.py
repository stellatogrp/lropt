import lropt.train.settings as settings


class TrainerSettings():
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
            The optimizer to use tor the descent algorithm. Default SGD.
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
            The multiplier of the lr if the scheduler is enabled. DEfault 0.1.
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
        linear: NN model, optional
            The linear NN model to use, Default None.
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

    """
    def __init__(self):
        self.train_size=settings.TRAIN_SIZE_DEFAULT
        self.train_shape=settings.TRAIN_SHAPE_DEFAULT
        self.fixb=settings.FIXB_DEFAULT
        self.num_iter=settings.NUM_ITER_DEFAULT  # Used to be "step"
        self.num_iter_size = settings.NUM_ITER_SIZE_DEFAULT
        self.lr=settings.LR_DEFAULT
        self.lr_size = settings.LR_SIZE_DEFAULT
        self.scheduler=settings.SCHEDULER_STEPLR_DEFAULT
        self.momentum=settings.MOMENTUM_DEFAULT
        self.optimizer=settings.OPT_DEFAULT
        self.init_rho=settings.INIT_RHO_DEFAULT
        self.init_A=settings.INIT_A_DEFAULT
        self.init_b=settings.INIT_B_DEFAULT
        self.save_history=settings.SAVE_HISTORY_DEFAULT
        self.seed=settings.SEED_DEFAULT
        self.init_lam=settings.INIT_LAM_DEFAULT
        self.init_mu=settings.INIT_MU_DEFAULT
        self.mu_multiplier=settings.MU_MULTIPLIER_DEFAULT
        self.init_alpha=settings.INIT_ALPHA_DEFAULT
        self.eta=settings.ETA_LAGRANGIAN_DEFAULT
        self.kappa=settings.KAPPA_DEFAULT  # (originall target_cvar)
        self.random_init=settings.RANDOM_INIT_DEFAULT
        self.num_random_init=settings.NUM_RANDOM_INIT_DEFAULT
        self.test_frequency=settings.TEST_FREQUENCY_DEFAULT
        self.test_percentage=settings.TEST_PERCENTAGE_DEFAULT
        self.batch_percentage=settings.BATCH_PERCENTAGE_DEFAULT
        self.solver_args=settings.LAYER_SOLVER
        self.n_jobs=settings.N_JOBS
        self.quantiles=settings.QUANTILES
        self.lr_step_size=settings.LR_STEP_SIZE
        self.lr_gamma=settings.LR_GAMMA
        self.position=settings.POSITION
        self.parallel=settings.PARALLEL
        self.aug_lag_update_interval = settings.UPDATE_INTERVAL
        self.lambda_update_threshold = settings.LAMBDA_UPDATE_THRESHOLD
        self.lambda_update_max = settings.LAMBDA_UPDATE_MAX
        self.max_batch_size = settings.MAX_BATCH_SIZE
        self.contextual = settings.CONTEXTUAL_DEFAULT
        self.linear = settings.CONTEXTUAL_LINEAR_DEFAULT
        self.init_weight = settings.CONTEXTUAL_WEIGHT_DEFAULT
        self.init_bias = settings.CONTEXTUAL_BIAS_DEFAULT
        self.x_endind = settings.X_ENDIND_DEFAULT
        self.max_iter_line_search = settings.DEFAULT_MAX_ITER_LINE_SEARCH
        self.policy = settings.POLICY_DEFAULT
        self.time_horizon = settings.TIME_HORIZON_DEFAULT
        self.batch_size = settings.BATCH_SIZE_DEFAULT
        self.test_batch_size = settings.TEST_BATCH_SIZE_DEFAULT
        self.simulator = settings.SIMULATOR_DEFAULT
        self.kwargs_simulator = settings.KWARGS_SIM_DEFAULT
        self.multistage = settings.MULTISTAGE_DEFAULT
        self.init_context = None
        self.init_uncertain_param = None
        self.covpred = False
        self._generate_slots()

    def _attr_exists(self, name) -> None:
        """
        This function raises an AttributeError if this object does not have a name property.
        """
        if hasattr(self, '__slots__') and name not in self.__slots__:
            raise AttributeError(f"TrainerSettings objects do not have a property called {name}.")

    def _generate_slots(self) -> None:
        """
        This function generates __slots__.
        __slots__ is used to make sure only existing fields can be updated, so the user cannot
        accidently add new fields.
        """
        #Set slots at the class level
        cls = self.__class__
        if not hasattr(cls, '__slots__'):
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
        #Check that all the keys are valid
        for key in kwargs.keys():
            self._attr_exists(key)

        #Assign
        for key, item in kwargs.items():
            super().__setattr__(key, item)
