
def addexpression_sep(unc_canon, expr):
    unc_lst = []
    std_lst = []
    for arg in expr.args:
        if unc_canon.has_unc_param(arg):
            unc_lst_0, std_lst_0 = unc_canon.separate_uncertainty(arg)
            unc_lst = unc_lst + unc_lst_0
            std_lst = std_lst + std_lst_0
        else:
            std_lst.append(arg)
    return (unc_lst, std_lst)
