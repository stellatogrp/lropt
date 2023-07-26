Welcome to LROPT's documentation!
==================================

Learning for robust optimization (LROPT) is a package to model and solve optimization Robust Optimization (RO) problems of the form

.. math::
  \begin{array}{ll}
    \mbox{minimize} & f(x)\\
    \mbox{subject to} & g(x, u) \le 0,\quad \forall u \in \mathcal{U(\theta)}
  \end{array}

where :math:`u` denotes the uncertain parameter and :math:`\mathcal{U(\theta)}` denotes the uncertainty set for the problem. Rather than forcing you to perform the often intensive math required to reduce this problem to a convex problem, LROPT lets you express RO problems naturally. There are two main ways
to use LROPT:

#. By explicitly defining an uncertainty set :math:`\mathcal{U}` and its parameterization :math:`\theta`
#. By passing a dataset :math:`U^N` of past realizations of :math:`u` and letting LROPT learn :math:`\theta`

Simple Examples
^^^^^^^^^^^^^^^

Predetermined Uncertainty Set
""""""""""""""""""""""""""""""

Lets see an example of the first case. Suppose we want to solve the following optimization problem, using an ellipsoidal uncertainty set:

.. math::
  \begin{array}{ll}
    \mbox{minimize} & c^T x \\
    \mbox{subject to} & (Pu+a)^Tx \le d,\quad \forall u \in \mathcal{U_\text{ellip}(A,b)}
  \end{array}

where :math:`\mathcal{U_\text{ellip}(A,b)} = \{u \mid \| Au + b \|_2 \leq 1 \}`. We would use LROPT to write the following:

.. code:: python

  import cvxpy as cp
  import numpy as np
  import lropt

  n = 4
  np.random.seed(0)
  A_unc = np.eye(n)
  b_unc = np.random.rand(n)
  P = np.random.rand(n,n)
  a = np.random.rand(n)
  c = np.random.rand(n)
  d = 10
  # Formulate robust constraints with lropt
  unc_set = Ellipsoidal(A = A_unc, b = b_unc)
  a = UncertainParameter(n,
                          uncertainty_set=unc_set)
  constraints = [(P @ u + a).T @ x <= d]
  objective = cp.Minimize(c @ x)
  prob_robust = RobustProblem(objective, constraints)
  prob_robust.solve()


Learned Uncertainty Set
"""""""""""""""""""""""

One of the most difficult modeling problems in RO is determining what :math:`\mathcal{U}(\theta)` should be.
Let's now use LROPT to solve the same problem, but this time passing in a dataset :math:`U^N` and letting LROPT learn :math:`\theta`.
Under the hood, training is done using pytorch, so users must additionally pass in representations of the objective and constraints with
uncertainty (we are working to remove the need for this feature). Because Ellipsoidal uncertainty sets have a continuous boundary, they are best for learning. For more details on how this learning is done, see our associated paper.

.. code:: python

  import torch
  import cvxpy as cp
  import numpy as np
  import lropt

  n = 4
  N = 100
  norms = npr.multivariate_normal(np.zeros(n), np.eye(n), N)
  u_data = np.exp(norms)

  np.random.seed(0)
  P = np.random.rand(n,n)
  a = np.random.rand(n)
  c = np.random.rand(n)
  # Formulate robust constraints with lropt
  a = UncertainParameter(n,
                          uncertainty_set=Ellipsoidal(data = u_data))
  constraints = [(P @ u + a).T @ x <= d]
  objective = cp.Minimize(c @ x)

  c_tch = torch.tensor(c, dtype=float)
  a_tch = torch.tensor(a, dtype=float)
  P_tch = torch.tensor(P, dtype=float)

  def f_tch(x,u):
      return c_tch @ x
  def g_tch(x,u):
      return (P_Tch @ u + a_tch).T @ x - d

  prob_robust = RobustProblem(objective, constraints, f_tch, [g_tch])
  prob_robust.train(step = 10)
  prob_robust.solve()

Families of problems
^^^^^^^^^^^^^^^^^
LROPT can also build uncertainty sets which generalize to a family of optimization problems, parameterized by some :math:`y`, where optimal solutions
are now functions both of :math:`\theta` and :math:`y`

.. math::

  \begin{equation}
    x(\theta, y) \in
    \begin{array}[t]{ll}
      \mbox{minimize}&f(x,y)\\
      \mbox{subject to} & g(x,u,y)  \le 0  \quad \forall u \in \mathcal{U}(\theta),
    \end{array}
  \end{equation}

In these instances, the user passes a dataset of :math:`Y^J` of :math:`y`'s we can use LROPT to learn a :math:`\theta` which generalizes well for the entire family of optimization problems. To write it up, we can consider the following example.

.. math::
  \begin{array}{ll}
    \mbox{minimize} & a^T x \\
    \mbox{subject to} & x^T (u+y) \leq c, \quad \forall u \in \mathcal{U_\text{ellip}(A,b)} \\
    & ||x|| \leq 2c
  \end{array}

This would be written as

.. code:: python

  import torch
  import cvxpy as cp
  import numpy as np
  import lropt

  n = 4
  N = 100
  norms = npr.multivariate_normal(np.zeros(n), np.eye(n), N)
  u_data = np.exp(norms)

  num_instances = 10
  y_data = npr.multivariate_normal(np.zeros(n), np.eye(n), num_instances)

  y = Parameter(n, data=y_data)
  u = UncertainParameter(n, uncertainty_set=Ellipsoidal(data=self.data))

  a = npr.randint(3, 5, n)
  c = 5

  x = cp.Variable(n)

  objective = cp.Maximize(a @ x)

  a_tch = torch.tensor(a, dtype=float)
  c_tch = torch.tensor(c, dtype=float)

  constraints = [x @ (u + y) <= c, cp.norm(x) <= 2*c]

  def f_tch(x,y,u):
      return a_tch @ x
  def g_tch(x,y,u):
      return x @ u + x @ y - c_tch

  prob = RobustProblem(objective, constraints,
                        objective_torch=f_tch, constraints_torch=[g_tch])
  prob.train(step=10)

.. image:: gif_images/newsvendor.gif
  :alt: StreamPlayer
  :align: center


.. toctree::
  :maxdepth: 2
  :hidden:

  getting_started/index
  api/index



Indices and tables
```````````````````

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
