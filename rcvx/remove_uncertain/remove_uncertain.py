from cvxpy.reductions.canonicalization import Canonicalization
from rcvx.remove_uncertain.atom_canonicalizers import (
    CANON_METHODS as remove_uncertain_methods)
from rcvx.uncertain import UncertainParameter
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.zero import Equality


class RemoveUncertainParameters(Canonicalization):
    """Remove uncertain parameter by reformulating robust problem."""

    def __init__(self, problem=None):
        super(RemoveUncertainParameters, self).__init__(
          problem=problem, canon_methods=remove_uncertain_methods)

    def accepts(self, problem):
        return True
        # TODO: Add proper accepts check
        #  atom_types = [type(atom) for atom in problem.atoms()]
        #  pwl_types = [abs, maximum, sum_largest, max, norm1, norm_inf]
        #  return any(atom in pwl_types for atom in atom_types)

    def apply(self, problem):

        # Check signs and apply
        # TODO: Can break easily! Need to check better method
        for c in problem.constraints:
            if type(c) in [Inequality, Equality]:
                rhs = c.args[1]
                unc_params_rhs = [x for x in rhs.parameters()
                                  if isinstance(x, UncertainParameter)]
                for param in unc_params_rhs:
                    param.flip_sign = True

        if not self.accepts(problem):
            raise ValueError("Cannot canonicalize uncertain problem atoms.")
        return super(RemoveUncertainParameters, self).apply(problem)
