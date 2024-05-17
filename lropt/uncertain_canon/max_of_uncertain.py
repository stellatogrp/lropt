import itertools

import numpy as np
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.atom import Atom
from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.constraints.constraint import Constraint
from cvxpy.utilities import scopes


class sum_of_max_of_uncertain(Constraint):

    def __init__(self, lst, term_not_in_max=0, constr_id=None) -> None:
        if not isinstance(lst,list):
            raise ValueError("Input must contain a list of lists")
        max_num_elements = 0
        for ind_lst in lst:
            if (not isinstance(ind_lst,list)):
                raise ValueError("Input must contain a list of lists")
            num_elements = len(ind_lst)
            if num_elements ==0:
                raise ValueError("Must not contain empty lists")
            max_num_elements = np.maximum(max_num_elements,num_elements)
        if max_num_elements <= 1:
                raise ValueError("At least one list must have two or more terms")
        combined_list = [[term_not_in_max],*lst]
        permutations= list(itertools.product(*combined_list))
        combined_permutation_list = [AddExpression([Atom.cast_to_const(arg) \
                                for arg in lst]) for lst in permutations]
        super(sum_of_max_of_uncertain,\
               self).__init__(combined_permutation_list, constr_id)

    def name(self) -> str:
        return "max %s <= 0" % self.args

    def is_dcp(self, dpp: bool = False) -> bool:
        """An max constraint is DCP if its argument is affine."""
        if dpp:
            with scopes.dpp_scope():
                for arg in self.args:
                    return_bool = arg.is_affine()
                    if not return_bool:
                        return False
                return return_bool
        for arg in self.args:
            return_bool = arg.is_affine()
            if not return_bool:
                return False
        return return_bool

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.args[0].is_quasiconvex()

    @property
    def residual(self):
        """The residual of the constraint.

        Returns
        ---------
        NumPy.ndarray
        """
        for arg in self.args:
            if arg.value is None:
                return None
        return np.maximum(*self.args, 0)

    def violation(self):
        res = self.residual
        if res is None:
            raise ValueError("Cannot compute the violation of an constraint "
                             "whose expression is None-valued.")
        viol = np.linalg.norm(res, ord=2)
        return viol

    def gen_torch_exp(self):
        # initarg = self.args[0].gen_torch_exp()
        # vals = []
        # for arg in self.args[1:]:
        #     vals.append(arg.gen_torch_exp()+initarg)
        # return torch.maximum(torch.stack(vals))
        # return super().gen_torch_exp()
        # vals = []
        # for arg in self.args:
        #     vals.append(arg + self.args[0])
        expr = maximum(*self.args)
        return expr.gen_torch_exp()


class max_of_uncertain(sum_of_max_of_uncertain):
    def __init__(self, lst, term_not_in_max=0, constr_id=None) -> None:
        super(max_of_uncertain, self).__init__([lst],term_not_in_max,constr_id)
        if not isinstance(lst,list):
            raise ValueError("Input must contain a list")
        if len(lst) <= 1:
            raise ValueError("There must be at least two terms in the list")
