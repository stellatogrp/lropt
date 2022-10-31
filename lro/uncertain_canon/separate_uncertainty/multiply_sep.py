from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.unary_operators import NegExpression


def promote_sep_sub(expr, unc_lst, std_lst):
    # import ipdb
    # ipdb.set_trace()
    new_unc_lst = [expr.value[0] * g_u for g_u in unc_lst]
    new_std_lst = [expr.value[0] * h_x for h_x in std_lst]
    return (new_unc_lst, new_std_lst)


def negexpression_sep_sub(expr, unc_lst, std_lst):
    new_unc_lst = [-1 * expr.args[0] * g_u for g_u in unc_lst]
    new_std_lst = [-1 * expr.args[0] * h_x for h_x in std_lst]
    return (new_unc_lst, new_std_lst)


def default_sep_sub(expr, unc_lst, std_lst):
    new_unc_lst = [expr * g_u for g_u in unc_lst]
    new_std_lst = [expr * h_x for h_x in std_lst]
    return (new_unc_lst, new_std_lst)


SEPARATION_SUB_METHODS = {
    Promote: promote_sep_sub,
    NegExpression: negexpression_sep_sub
}


def multiply_sep(unc_canon, expr):

    if unc_canon.has_unc_param(expr.args[0]) and \
            unc_canon.has_unc_param(expr.args[1]):
        raise ValueError("DRP error: Cannot have uncertainty multiplied by each other")
    if unc_canon.has_unc_param(expr.args[0]):
        unc_lst, std_lst = unc_canon.separate_uncertainty(expr.args[0])
        if type(expr.args[1]) not in SEPARATION_SUB_METHODS:
            return default_sep_sub(expr.args[1], unc_lst, std_lst)

        func = SEPARATION_SUB_METHODS[type(expr.args[1])]
        return func(expr.args[1], unc_lst, std_lst)
    else:
        unc_lst, std_lst = unc_canon.separate_uncertainty(expr.args[1])
        if type(expr.args[0]) not in SEPARATION_SUB_METHODS:
            return default_sep_sub(expr.args[0], unc_lst, std_lst)

        func = SEPARATION_SUB_METHODS[type(expr.args[0])]
        return func(expr.args[0], unc_lst, std_lst)
        # if isinstance(expr.args[0], NegExpression):
        #     new_unc_lst = [-1*expr.args[0].args[0] * g_u for g_u in unc_lst]
        #     new_std_lst = [-1*expr.args[0].args[0] * h_x for h_x in std_lst]
        #     return (new_unc_lst, new_std_lst)
        # elif isinstance(expr.args[0], Promote):
        #     new_unc_lst = [expr.args[0].value[0] * g_u for g_u in unc_lst]
        #     new_std_lst = [expr.args[0].value[0] * h_x for h_x in std_lst]
        #     return (new_unc_lst, new_std_lst)
