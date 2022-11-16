
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.unary_operators import NegExpression


def promote_sep_sub(expr, unc_lst, std_lst):
    new_unc_lst = [expr.value[0] * g_u for g_u in unc_lst]
    new_std_lst = [expr.value[0] * h_x for h_x in std_lst]
    return new_unc_lst, new_std_lst


def negexpression_sep_sub(expr, unc_lst, std_lst):
    new_unc_lst = [-1 * expr.args[0] @ g_u for g_u in unc_lst]
    new_std_lst = [-1 * expr.args[0] @ h_x for h_x in std_lst]
    return new_unc_lst, new_std_lst


def default_sep_sub(expr, unc_lst, std_lst):
    new_unc_lst = [expr @ g_u for g_u in unc_lst]
    new_std_lst = [expr @ h_x for h_x in std_lst]
    return new_unc_lst, new_std_lst


SEPARATION_SUB_METHODS = {
    Promote: promote_sep_sub,
    NegExpression: negexpression_sep_sub
}


def mulexpression_sep(unc_canon, expr):
    if unc_canon.has_unc_param(expr.args[0]) and \
            unc_canon.has_unc_param(expr.args[1]):
        raise ValueError("DRP error: Cannot have uncertainty multiplied by each other")

    unc_param, non_unc_param = expr.args
    if unc_canon.has_unc_param(non_unc_param):
        non_unc_param, unc_param = expr.args

    unc_lst, std_lst, is_max = unc_canon.separate_uncertainty(unc_param)
    if type(non_unc_param) not in SEPARATION_SUB_METHODS:
        if is_max == 0:
            unc_lst, std_lst = default_sep_sub(non_unc_param, unc_lst, std_lst)
            return unc_lst, std_lst, 0
        else:
            for idx in range(len(unc_lst)):
                unc_lst[idx], std_lst[idx] = default_sep_sub(non_unc_param, unc_lst[idx], std_lst[idx])
                return unc_lst, std_lst, 1

    func = SEPARATION_SUB_METHODS[type(non_unc_param)]
    if is_max == 0:
        unc_lst, std_lst = func(non_unc_param, unc_lst, std_lst)
        return unc_lst, std_lst, 0
    else:
        for idx in range(len(unc_lst)):
            unc_lst[idx], std_lst[idx] = func(non_unc_param, unc_lst[idx], std_lst[idx])
            return unc_lst, std_lst, 1
