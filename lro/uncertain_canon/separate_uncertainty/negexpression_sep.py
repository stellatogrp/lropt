def negexpression_sep(unc_canon, expr):
    unc_lst, std_lst = unc_canon.separate_uncertainty(expr.args[0])
    new_unc_lst = [-1 * g_u for g_u in unc_lst]
    new_std_lst = [-1 * h_x for h_x in std_lst]
    return (new_unc_lst, new_std_lst)
