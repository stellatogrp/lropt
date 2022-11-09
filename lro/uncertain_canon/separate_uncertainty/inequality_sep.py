def inequality_sep(unc_canon, expr):
    return unc_canon.separate_uncertainty(expr.args[0] + -1*expr.args[1])
    # return expr._expr
