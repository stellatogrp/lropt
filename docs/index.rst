Welcome to LROPT's documentation!
==================================

Learning for robust optimization (LROPT) is a package to model and solve optimization problems under uncertainty of the form

.. math::
  \begin{array}{ll}
    \mbox{minimize} & f(x)\\
    \mbox{subject to} & g(x, u) \le 0,\quad \forall u \in \mathcal{U}
  \end{array}

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started/index
   examples/index
   api/index



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
