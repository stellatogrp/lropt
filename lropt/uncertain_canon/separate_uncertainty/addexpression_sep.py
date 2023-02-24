def addexpression_sep(unc_canon, expr):
    unc_lst = []
    std_lst = []
    maxlst_u = None
    maxlst_s = None
    for arg in expr.args:
        if unc_canon.has_unc_param(arg):
            unc_lst_0, std_lst_0, is_max = unc_canon.separate_uncertainty(arg)
            if is_max == 0:
                unc_lst = unc_lst + unc_lst_0
                std_lst = std_lst + std_lst_0
            else:
                if maxlst_u is None:
                    maxlst_u = unc_lst_0
                    maxlst_s = std_lst_0
                else:
                    raise ValueError("Can only have one max_affine atom in a constraint")
        else:
            std_lst.append(arg)
    if maxlst_u is None:
        return unc_lst, std_lst, 0
    else:
        for idx in range(len(maxlst_u)):
            maxlst_u[idx] = maxlst_u[idx] + unc_lst
            maxlst_s[idx] = maxlst_s[idx] + std_lst
        return maxlst_u, maxlst_s, 1
