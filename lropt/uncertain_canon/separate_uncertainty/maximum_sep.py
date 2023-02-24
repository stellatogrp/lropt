def maximum_sep(unc_canon, expr):
    unc_lst = []
    std_lst = []
    for arg in expr.args:
        assert (arg.is_affine()), "Arguments of cp.maximum must be affine"
        if unc_canon.has_unc_param(arg):
            new_unc_lst, new_std_lst, is_max = unc_canon.separate_uncertainty(arg)
            assert is_max == 0, "cannot have maximum of maximum"
            unc_lst.append(new_unc_lst)
            std_lst.append(new_std_lst)
        else:
            unc_lst.append([])
            std_lst.append([arg])
    return unc_lst, std_lst, 1
