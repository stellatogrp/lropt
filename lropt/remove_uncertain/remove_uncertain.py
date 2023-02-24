from lropt.remove_uncertain.atom_canonicalizers import \
    CANON_METHODS as remove_uncertain_methods
from lropt.uncertain_canon.uncertain_canonicalization import \
    Uncertain_Canonicalization


class RemoveUncertainParameters(Uncertain_Canonicalization):
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
        if not self.accepts(problem):
            raise ValueError("Cannot canonicalize uncertain problem atoms.")
        return super(RemoveUncertainParameters, self).apply(problem)
