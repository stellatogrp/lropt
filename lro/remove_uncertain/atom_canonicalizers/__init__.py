from cvxpy.atoms.affine.binary_operators import (AddExpression, MulExpression,
                                                 multiply)

from lro.remove_uncertain.atom_canonicalizers.add_canon import add_canon
from lro.remove_uncertain.atom_canonicalizers.mul_canon import mul_canon
from lro.remove_uncertain.atom_canonicalizers.mulexpression_canon import \
    mulexpression_canon
from lro.remove_uncertain.atom_canonicalizers.quad_canon import quad_canon
# from cvxpy.atoms.quad_form import QuadForm
from lro.uncertain_atoms.quad_form import UncertainQuadForm

CANON_METHODS = {
    AddExpression : add_canon,
    MulExpression: mulexpression_canon,
    multiply: mul_canon,  # TODO: Create separate for scalars?
    UncertainQuadForm: quad_canon
}
