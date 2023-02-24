def negexpression_sep(unc_canon, expr):
    unc_lst, std_lst, is_max = unc_canon.separate_uncertainty(expr.args[0])
    if is_max == 1:
        for idx in range(len(unc_lst)):
            unc_lst[idx] = [-1 * g_u for g_u in unc_lst[idx]]
            std_lst[idx] = [-1 * h_x for h_x in std_lst[idx]]
        return unc_lst, std_lst, 1
    else:
        new_unc_lst = [-1 * g_u for g_u in unc_lst]
        new_std_lst = [-1 * h_x for h_x in std_lst]
        return new_unc_lst, new_std_lst, 0
