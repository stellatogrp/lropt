import numpy as np
from cvxpy import Variable, norm
from sklearn.cluster import KMeans

from cvxro.parameter import ShapeParameter, SizeParameter
from cvxro.uncertainty_sets.uncertainty_set import UncertaintySet


class MRO(UncertaintySet):
    r"""
    Uncertainty set where the parameter is constrained to lie in a
    Wasserstein ball of the form
    .. math::
        \{ \sum( w_k||u_k - d_k ||^\text{power}_p)\leq \rho\\}

    Parameters
    ----------
    K : int, optional
        Number of clusters for the MRO representation (default 1).
    rho : float, optional
        Radius/scaling parameter for the Wasserstein ball (default 1).
    data : numpy.ndarray
        Full dataset of uncertainty realizations (each row is one sample).
    train_data : numpy.ndarray, optional
        Subset of `data` used for training (defaults to `data`). If
        `indices_dict` contains a 'train' key, the constructor will subset
        `train_data` from `data` using those validated indices.
    power : int, optional
        Power parameter used in the MRO distance; must be non-negative.
    p : int, optional
        Norm order used in distance computations (default 2).
    a, b : arrays or ShapeParameter, optional
        Parameters for affine transform or training parameters when `train=True`.
    train : bool, optional
        If True, indicates the set should create trainable parameters.
    c, d : arrays, optional
        Polyhedral support parameters (lhs/right-hand side) if applicable.
    ub, lb : array or float, optional
        Upper and lower bounds for the uncertain vector.
    sum_eq : array or float, optional
        Equality constraint on the uncertain vector.
    DRO : bool, optional
        If True, sets K equal to the number of training samples (WDRO variant).
    eval_data : numpy.ndarray, optional
        Data to use for evaluation purposes.
    indices_dict : dict, optional
        Optional mapping with keys 'train', 'test', and 'validate' that specify
        integer index arrays or boolean masks to split the provided `data` into
        training, testing, and validation subsets. When provided the indices are
        validated against `data.shape[0]` and normalized to integer index arrays.

    Returns
    -------
    MRO
        Constructed MRO uncertainty set.

    """

    def __init__(self, K=1, rho=1, data=None, train_data = None, power=1, p=2,
                 a=None, b=None, train=False, c=None, d=None,
                 ub=None, lb=None, sum_eq=None, DRO=False, eval_data=None, indices_dict=None):
        if data is None:
            raise ValueError("You must provide data")
        if train_data is None:
            train_data = data

        # If indices_dict provided, validate and use provided 'train' indices
        if indices_dict is not None:
            from cvxro.uncertainty_sets.utils import check_indices_dict
            # validate against full data length
            validated = check_indices_dict(indices_dict, data.shape[0])
            # if 'train' indices provided, subset train_data accordingly
            if validated is not None and 'train' in validated:
                train_idx = validated['train']
                if train_idx.size == 0:
                    raise ValueError("Provided 'train' indices are empty")
                train_data = data[train_idx]
            # replace indices_dict with validated (normalized) version
            indices_dict = validated
        if DRO:
            K = train_data.shape[0]
        self._dimension = train_data.shape[1]
        if train:
            a = ShapeParameter((self._dimension, self._dimension))
            a.value = np.eye(self._dimension)
            b = ShapeParameter(self._dimension)
            b.value = np.zeros(self._dimension)

        if (not train) and (a is not None):
            if a.shape[1] != self._dimension:
                raise ValueError("Mismatching dimension for A.")
        if rho <= 0:
            raise ValueError("Rho value must be positive.")
        if p < 0.:
            raise ValueError("Order must be a nonnegative number.")
        if power < 0:
            raise ValueError("Power must be a nonnegative integer.")

        kmeans = KMeans(n_clusters=K, n_init='auto').fit(train_data)
        self.affine_transform_temp = None
        self.affine_transform = None
        self._data = data
        self._train_data = train_data
        self.eval_data = eval_data
        self._N = train_data.shape[0]
        self._K = K
        self._power = power
        self._p = p
        self._Dbar = kmeans.cluster_centers_
        self._w = np.bincount(kmeans.labels_) / train_data.shape[0]
        self._rho = rho
        self._s = None
        self._train = train
        self._m = data.shape[1]
        self._c = c
        self._d = d
        self._define_support = False
        self._ub = ub
        self._lb = lb
        self._sum_eq = sum_eq
        self._b = b
        self._a = a
        self._rho_mult = SizeParameter(value=1.)
        self.indices_dict = indices_dict

        self._lam = None

    @property
    def rho(self):
        return self._rho

    @property
    def p(self):
        return self._p

    @property
    def rho_mult(self):
        return self._rho_mult

    @property
    def power_val(self):
        return self._power

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def dimension(self):
        return self._dimension

    @property
    def N(self):
        return self._N

    @property
    def data(self):
        return self._data

    @property
    def K(self):
        return self._K

    @property
    def w(self):
        return self._w

    @property
    def Dbar(self):
        return self._Dbar

    @property
    def ub(self):
        return self._ub

    @property
    def lb(self):
        return self._lb

    @property
    def sum_eq(self):
        return self._sum_eq

    @property
    def c(self):
        return self._c

    @property
    def d(self):
        return self._d

    def q(self):
        return 1. + 1. / (self._power - 1.)

    def dual_norm(self):
        if self.p == 1:
            return np.inf
        return 1. + 1. / (self._p - 1.)

    def s(self):
        return self._s

    def phi(self):
        if self._power == float("inf"):
            return 1
        else:
            return (self.q() - 1.)**(self.q() - 1.)/(self.q()**self.q())

    def conjugate(self, var, supp_var,k_ind):
        if not self._define_support:
            if self._c is None:
                if not isinstance(var, Variable):
                    self._c = np.zeros((var, var))
                else:
                    self._c = np.zeros((var.shape[0], var.shape[0]))
            if self._d is None:
                if not isinstance(var, Variable):
                    self._d = np.zeros(var)
                else:
                    self._d = np.zeros(var.shape[0])
            self._define_support = True
        if k_ind == 0:
            lmbda = Variable()
            self._lam = lmbda
            s = Variable(self._K)
            self._s = s
        lmbda = self._lam
        sval = self._s
        if not isinstance(var, Variable):
            constr = []
            return -sval[k_ind], constr, lmbda, sval
        else:
            supp_newvar = Variable(len(self._d))
            constr = [norm(var, p=self.dual_norm()) <= lmbda]
            constr += [lmbda >= 0]
            constr += [self._c.T@supp_newvar == supp_var]
            constr += [supp_newvar >= 0]
            return var@self.Dbar[k_ind] + \
                self._d@supp_newvar-sval[k_ind], constr, lmbda, sval


class WDRO(MRO):
    r"""
    Uncertainty set where the parameter is constrained to lie in a
    Wasserstein ball of the form
    .. math::
        \{ \sum( (1/N)||u_k - d_k ||^\text{power}_p)\leq \rho\\}
    """

    def __init__(self, rho=1, data=None, train_data = None, power=1, p=2,
                 a=None, b=None, train=False, c=None, d=None,
                 ub=None, lb=None, sum_eq=None, eval_data=None, indices_dict=None):
        super(WDRO, self).__init__(
            rho=rho,
            data=data,
            train_data=train_data,
            power=power,
            p=p,
            a=a,
            b=b,
            train=train,
            c=c,
            d=d,
            ub=ub,
            lb=lb,
            sum_eq=sum_eq,
            DRO=True,
            eval_data=eval_data,
            indices_dict=indices_dict,
        )
