def promoteexpression_sep(unc_canon, expr):
    unc_lst, std_lst, is_max = unc_canon.separate_uncertainty(expr.args[0])
    return unc_lst, std_lst, is_max
