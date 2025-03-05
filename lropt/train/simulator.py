import abc
from abc import ABC

import torch

import lropt.train.settings as settings


class Simulator(ABC):
    """Simulator class for the multi-stage problem. All parameters should be tensors."""

    @abc.abstractmethod
    def simulate(self,x,u,**kwargs):
        """Simulate next set of parameters using current parameters x
        and variables u, with added uncertainty
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stage_cost(self,x,u, **kwargs):
        """ Create the current stage cost using the current state x
        and decision u
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def stage_cost_eval(self,x,u, **kwargs):
        """ Create the current stage evaluation cost using the current state x
        and decision u. This may differ from the stage cost, which is used
        for training.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def constraint_cost(self,x,u,alpha,**kwargs):
        """ Create the current constraint penalty cost
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def init_state(self,batch_size, seed, **kwargs):
        """ initialize the parameter value
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def prob_constr_violation(self,x,u,**kwargs):
        """ calculate current probability of constraint violation
        """
        return torch.tensor(0,dtype = settings.DTYPE)



class Default_Simulator(ABC):
    def __init__(self,trainer):
        self.trainer = trainer

    def simulate(self,x,u,**kwargs):
        """Simulate next set of parameters using current parameters x
        and variables u, with added uncertainty
        """
        return x

    def stage_cost(self,x,u,**kwargs):
        """ Create the current stage cost using the current state x
        and decision u
        """
        return kwargs['trainer'].train_objective(kwargs['batch_int'], kwargs['eval_args'])


    def stage_cost_eval(self,x,u,**kwargs):
        """ Create the current stage evaluation cost using the current state x
        and decision u
        """
        return torch.tensor(kwargs['trainer'].evaluation_metric(
            kwargs['batch_int'], kwargs['eval_args'],
            kwargs['quantiles']),dtype=settings.DTYPE)


    def constraint_cost(self,x,u,alpha, **kwargs):
        """ Create the current constraint penalty cost
        """
        return kwargs['trainer'].train_constraint(kwargs['batch_int'],
                                                  kwargs['eval_args'],
                                                    alpha,
                                                    kwargs['slack'],
                                                    kwargs['eta'],
                                                    kwargs['kappa'])

    def init_state(self,batch_size, seed,**kwargs):
        """ initialize the parameter value
        """
        if kwargs['trainer']._eval_flag:
            return kwargs['trainer']._gen_batch(kwargs['trainer'].test_size,
                                                kwargs['trainer'].x_test_tch,
                                                kwargs['trainer'].u_test_set,
                                                1, kwargs["max_batch_size"])

        else:
            return kwargs['trainer']._gen_batch(kwargs['trainer'].train_size,
                                                kwargs['trainer'].x_train_tch,
                                                kwargs['trainer'].u_train_set,
                                                kwargs['batch_percentage'],
                                                kwargs["max_batch_size"])

    def prob_constr_violation(self,x,u,**kwargs):
        """ calculate current probability of constraint violation
        """
        return kwargs['trainer'].prob_constr_violation(kwargs['batch_int'],
                                                kwargs['eval_args'])
