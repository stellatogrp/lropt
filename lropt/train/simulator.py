import abc
from abc import ABC

import numpy as np
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
    def constraint_cost(self,x,u,**kwargs):
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



class DefaultSimulator(ABC):
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
        if self.trainer.settings.cost_func:
            if self.trainer.settings.use_eval:
            #     return self.trainer.settings.obj_scale*self.trainer.evaluation_metric(
            # kwargs['batch_int'], kwargs['eval_args'],
            # self.trainer.settings.quantiles)[1]
                return self.trainer.settings.obj_scale*\
                self.trainer.train_objective(kwargs['batch_int'], kwargs['eval_args'])
            else:
                weights = u[1]
                prod = weights.mul(kwargs["u_train"])
                port_values = torch.sum(prod, dim=1)
                loss = -port_values.sum()
                return self.trainer.settings.obj_scale*loss
            # return self.trainer.settings.obj_scale*\
            # self.trainer.train_objective(kwargs['batch_int'], kwargs['eval_args'])
        else:
            weights = u[1]
            prod = weights.mul(kwargs["u_train"])
            port_values = torch.sum(prod, dim=1)
            quant = int((1-kwargs["eta"])*len(port_values)) + 1
            port_sorted = torch.sort(port_values, descending=True)[0]
            quant = port_sorted[quant]

            port_le_quant = port_values.le(quant).float()
            port_le_quant.requires_grad = True
            cvar_loss =  port_values.mul(port_le_quant).sum() / port_le_quant.sum()
            loss = -cvar_loss
            return self.trainer.settings.obj_scale*loss

    def stage_cost_eval(self,x,u,**kwargs):
        """ Create the current stage evaluation cost using the current state x
        and decision u
        """
        return torch.tensor(self.trainer.evaluation_metric(
            kwargs['batch_int'], kwargs['eval_args'],
            self.trainer.settings.quantiles),dtype=settings.DTYPE)


    def constraint_cost(self,x,u,**kwargs):
        """ Create the current constraint penalty cost
        """
        return self.trainer.train_constraint(kwargs['batch_int'],
                                                  kwargs['eval_args'],
                                                    kwargs['alpha'],
                                                    self.trainer.settings.eta,
                                                    self.trainer.settings.kappa)

    def init_state(self,batch_size, seed,**kwargs):
        """ initialize the parameter value
        """
        if self.trainer._validate_flag:
            return self.trainer._gen_batch(self.trainer.validate_size,
                                                self.trainer.x_validate_tch,
                                                self.trainer.u_validate_set,
                                                1, self.trainer.settings.max_batch_size,
                                                seed=seed)
        elif self.trainer._test_flag:
            return self.trainer._gen_batch(self.trainer.test_size,
                                                self.trainer.x_test_tch,
                                                self.trainer.u_test_set,
                                                1, np.inf,
                                                seed=seed)
        else:
            return self.trainer._gen_batch(self.trainer.train_size,
                                                self.trainer.x_train_tch,
                                                self.trainer.u_train_set,
                                                self.trainer.settings.batch_percentage,
                                                self.trainer.settings.max_batch_size,seed=seed)

    def prob_constr_violation(self,x,u,**kwargs):
        """ calculate current probability of constraint violation
        """
        return self.trainer.prob_constr_violation(kwargs['batch_int'],
                                                kwargs['eval_args'])
