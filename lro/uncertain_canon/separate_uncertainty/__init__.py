from cvxpy import Variable, problems
# from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.constraints.nonpos import Inequality

# from cvxpy.atoms.quad_form import QuadForm
from lro.uncertain_atoms.quad_form import QuadForm
from lro.uncertain_canon.separate_uncertainty.addexpression_sep import \
    addexpression_sep
from lro.uncertain_canon.separate_uncertainty.inequality_sep import \
    inequality_sep
from lro.uncertain_canon.separate_uncertainty.mulexpression_sep import \
    mulexpression_sep
from lro.uncertain_canon.separate_uncertainty.multiply_sep import multiply_sep
from lro.uncertain_canon.separate_uncertainty.negexpression_sep import \
    negexpression_sep
from lro.uncertain_canon.separate_uncertainty.quadform_sep import quadform_sep

SEPARATION_METHODS = {
    Inequality: inequality_sep,
    multiply: multiply_sep,
    MulExpression: mulexpression_sep,
    AddExpression: addexpression_sep,
    NegExpression: negexpression_sep,
    QuadForm: quadform_sep
    }
