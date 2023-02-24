from cvxpy.atoms.affine.binary_operators import MulExpression, multiply

from lropt.uncertain_canon.remove_constant.mulexpression_rm import \
    mulexpression_rm
from lropt.uncertain_canon.remove_constant.multiply_rm import multiply_rm

REMOVE_CONSTANT_METHODS = {
    multiply: multiply_rm,
    MulExpression: mulexpression_rm
    }
