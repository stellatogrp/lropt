from cvxpy.atoms.affine.binary_operators import MulExpression, multiply

#  from cvxpy.atoms.affine.unary_operators import NegExpression
from lro.remove_uncertain.atom_canonicalizers.matmul_canon import matmul_canon

#  from lro.remove_uncertain.atom_canonicalizers.negexpression_canon \
    #  import negexpression_canon


CANON_METHODS = {
    MulExpression: matmul_canon,
    multiply: matmul_canon,  # TODO: Create separate for scalars?
    #  NegExpression: negexpression_canon
}








#  from cvxpy.atoms.affine.index import special_index
#  from cvxpy.transforms.indicator import indicator
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.cumsum_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.entr_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.geo_mean_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.huber_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.indicator_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.kl_div_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.lambda_max_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.lambda_sum_largest_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_det_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.log_sum_exp_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.log1p_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.logistic_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.matrix_frac_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.normNuc_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.power_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.pnorm_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.sigma_max_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.quad_form_canon import *
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers.quad_over_lin_canon import *
#
#  from cvxpy.reductions.utilities import special_index_canon
#
#  from cvxpy.reductions.eliminate_pwl.atom_canonicalizers import (abs_canon,
#      maximum_canon, max_canon, minimum_canon, min_canon, norm1_canon,
#      norm_inf_canon, sum_largest_canon)


