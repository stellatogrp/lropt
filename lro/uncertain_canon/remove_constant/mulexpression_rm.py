def mulexpression_rm(unc_canon, expr, cons):
    expr1, cons = unc_canon.remove_const(expr.args[0], cons)
    expr2, cons = unc_canon.remove_const(expr.args[1], cons)
    return expr1*expr2, cons
