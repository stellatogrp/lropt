from lropt.uncertain import UncertainParameter


def has_uncertain_param_extra(self, expr):
    # does this expression have an uncertain parameter?
    has_uncertain = False
    if len(expr.args) == 0:
        return isinstance(expr, UncertainParameter)

    else:
        for args in expr.args:
            if self.has_uncertain_param_1(args):
                return True

        return has_uncertain
