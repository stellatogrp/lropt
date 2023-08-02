LROPT learning syntax
=====================

Recall our RO problem with Family Parameters

Families of problems
^^^^^^^^^^^^^^^^^^^^
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
