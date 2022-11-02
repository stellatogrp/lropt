def addexpression_sep(unc_canon, expr):
    if unc_canon.has_unc_param(expr.args[0]) and \
            unc_canon.has_unc_param(expr.args[1]):
        unc_lst_0, std_lst_0 = unc_canon.separate_uncertainty(expr.args[0])
        unc_lst_1, std_lst_1 = unc_canon.separate_uncertainty(expr.args[1])
        return (unc_lst_0 + unc_lst_1, std_lst_0 + std_lst_1)

    unc_param, non_unc_param = expr.args
    if unc_canon.has_unc_param(non_unc_param):
        non_unc_param, unc_param = expr.args

    unc_lst, std_lst = unc_canon.separate_uncertainty(unc_param)
    std_lst.append(non_unc_param)
    return (unc_lst, std_lst)
