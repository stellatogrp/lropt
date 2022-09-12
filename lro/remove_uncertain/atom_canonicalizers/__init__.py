from cvxpy.atoms.affine.binary_operators import MulExpression, multiply

from lro.remove_uncertain.atom_canonicalizers.mulexpression_canon import \
    mulexpression_canon

CANON_METHODS = {
    MulExpression: mulexpression_canon,
    multiply: mulexpression_canon,  # TODO: Create separate for scalars?
}
