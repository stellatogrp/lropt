def mulexpression_rm(unc_canon, expr, constant):
    expr1, constant = unc_canon.remove_constant(expr.args[0], constant)
    expr2, constant = unc_canon.remove_constant(expr.args[1], constant)
    return expr1*expr2, constant
