def mulexpression_rm(unc_canon, expr, constant):
    expr1, constant = unc_canon.remove_constant(expr.args[0], constant)
    expr2, constant = unc_canon.remove_constant(expr.args[1], constant)
    if len(expr1.shape)==0 or len(expr2.shape)==0 :
        return expr1*expr2, constant
    return expr1@expr2, constant
