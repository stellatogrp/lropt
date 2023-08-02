LROPT Ruleset
=============
Recall a general Robust Optimization problem (RO)

.. math::
  \begin{array}{ll}
    \mbox{minimize} & f(x)\\
    \mbox{subject to} & g(x, u) \le 0,\quad \forall u \in \mathcal{U(\theta)}
  \end{array}

Here we discuss the acceptable uncertain functions :math:`g(x)` LROPT is able to process. For consistency purposes, we assume that
we have a problem written as the above


We say that :math:`g(x,u,y)` is LROPT-compliant if it can be written as a sum of smaller subexpressions

.. math::
    \begin{equation}
        g(x,u,y) = \sum_{i=1}^n g_i(x,u,y),
    \end{equation}

where each subexpression :math:`g_i(x,u,y)` is DPP in :math:`y` and either


	#. Disciplined Parametric Programming (DPP) in the parameter :math:`y`, and DCP in :math:`x`
	#. A non-negative scaling of LROPT atoms described in Section \ref{sec:atoms}
	#. A maximum over any number of the previous expressions

Broadly speaking, a RO problem has a convex reformulation if :math:`g(x,u,y)` is concave in :math:`u` and convex in :math:`x`, or is represented by the maximum of expressions which are concave in :math:`u` and convex in :math:`x`.


LROPT atoms
___________


The LROPT package introduces the following new atoms which are convex in :math:`x` and concave in :math:`u`.

#. Matrix-vector product.
    The matrix-vector multiplication syntax :math:`{\tt @}` is supported for writing affine expressions in :math:`u` such as :math:`\tt x@P@u`.
#. Quadratic Form
    The atom :math:`\tt{quad\_form(u,A*x)}` represents the function :math:`g_i(x,u) = (u^TAu)x` where :math:`A \in \mathbf{R}^{m\times m}`, :math:`A\preceq 0`, and :math:`x\in\mathbf{R}` is a scalar variable multiplying the quadratic form.
#. Weighted log-sum-exp.
    The atom :math:`\tt{log\_sum\_exp}(u,x)` represents the function :math:`\log\left(\sum_{i=1}^n u_i e^{x_i }\right)`. Here :math:`x\in \mathbf{R}^m`, and :math:`u\in \mathbf{R}^m` must be of the same dimension.
#. Weighted :math:`l_2` norm.
    The atom :math:`\tt{weighted\_norm2}` represents the function :math:`\left( \sum_{i=1}^n u_ix_i^2\right)^{\frac{1}{2}}`. Again :math:`x\in \mathbf{R}^m`, and :math:`u\in \mathbf{R}^m` must be of the same dimension.

#. Matrix quadratic form.
    The atom :math:`\tt{mquad\_form(U,x)}` reprsents the function :math:`g(x,u) = x^TUx` where :math:`U\succeq 0`, is constrained by LROPT to be PSD.
