def addexpression_sep(unc_canon, expr):
    if unc_canon.has_unc_param(expr.args[0]) and \
            unc_canon.has_unc_param(expr.args[1]):
        unc_lst_0, std_lst_0 = unc_canon.separate_uncertainty(expr.args[0])
        unc_lst_1, std_lst_1 = unc_canon.separate_uncertainty(expr.args[1])
        return (unc_lst_0 + unc_lst_1, std_lst_0 + std_lst_1)
    elif unc_canon.has_unc_param(expr.args[0]):
        unc_lst, std_lst = unc_canon.separate_uncertainty(expr.args[0])
        std_lst.append(expr.args[1])
        return (unc_lst, std_lst)
    else:
        unc_lst, std_lst = unc_canon.separate_uncertainty(expr.args[1])
        std_lst.append(expr.args[0])
        return (unc_lst, std_lst)
