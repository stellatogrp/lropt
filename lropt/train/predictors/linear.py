import numpy as np
import torch

import lropt.train.settings as settings

torch.set_default_dtype(settings.DTYPE)

class LinearPredictor(torch.nn.Module):

    def __init__(self,predict_mean = False):
        super(LinearPredictor, self).__init__()
        self.predict = predict_mean

    def initialize(self,a_tch,b_tch,trainer):
        """Initialize the parameters"""
        num_in,num_out, a_totsize = trainer.initialize_predictor_dims()
        self.linear = torch.nn.Linear(num_in,num_out)
        self.customize(a_totsize,a_tch,b_tch,trainer.settings.init_bias,trainer.settings.init_weight,trainer.settings.random_init)
        if self.predict:
            input_tensors = trainer.create_input_tensors(trainer.x_train_tch)
            self.gen_weights(input_tensors,trainer.u_train_tch,a_tch)


    def customize(self,a_totsize,a_tch,b_tch,init_bias,init_weight,random_init):
        """Set the weights of the predictor using mean-variance or given info
        Args:
        a_totsize:
            index where the bias begins
        a_tch, b_tch
            initialized a and b
        init_weight, init_bias
            initial weight and bias to use for the linear predictor
        random_init
            whether or not the predictor weights are initialized randomly
        """
        self.linear.bias.data[a_totsize:] = b_tch
        if not random_init:
            with torch.no_grad():
                torch_b = b_tch
                torch_a = a_tch.flatten()
                torch_concat = torch.hstack([torch_a, torch_b])
            self.linear.weight.data.fill_(0.000)
            self.linear.bias.data = torch_concat
            if init_weight is not None:
                self.linear.weight.data = torch.tensor(
                    init_weight, dtype=torch.double, requires_grad=True
                )
            if init_bias is not None:
                self.linear.bias.data = torch.tensor(
                    init_bias, dtype=torch.double, requires_grad=True
                )

    def gen_weights(self,input,output,a_tch):
        """Set the weights of the predictor using lstsq
        Args:
        a_tch
            initialized a, usually the sqrt of the covariance
        input
            context data to train on
        output
            uncertain parameter data to train on
        """
        input = input.detach().numpy()
        output = output.detach().numpy()
        num_in = input.shape[1]
        m = output.shape[1]
        N = input.shape[0]
        stacked_context = np.hstack([input,np.ones((N,1))])
        mults = [np.linalg.lstsq(stacked_context,output[:,0])[0]]
        for i in range(1,m):
            new_mults = np.linalg.lstsq(stacked_context,output[:,i])[0]
            mults.append(new_mults)
        mults_mean = np.vstack(mults)
        mults_mean_weight = mults_mean[:,:-1]
        mults_mean_bias = mults_mean[:,-1]
        self.init_bias = np.hstack([a_tch.detach().numpy().flatten(),mults_mean_bias])
        self.init_weight = np.vstack([np.zeros((m*m,num_in)),mults_mean_weight])
        self.linear.weight.data = torch.tensor(
            self.init_weight, dtype=torch.double, requires_grad=True
        )
        self.linear.bias.data = torch.tensor(
            self.init_bias, dtype=torch.double, requires_grad=True
        )


    def forward(self, x,a_shape,b_shape,train_flag):
        """create a_tch and b_tch using the predictor"""
        out = self.linear(x)
        raw_a = out[:, : a_shape[0] * a_shape[1]]
        raw_b = out[:, a_shape[0] * a_shape[1] :]
        a_tch = raw_a.view(out.shape[0], a_shape[0], a_shape[1])
        b_tch = raw_b.view(out.shape[0], b_shape[0])
        if not train_flag:
            a_tch = torch.tensor(a_tch, requires_grad=False)
            b_tch = torch.tensor(b_tch, requires_grad=False)
        return a_tch, b_tch
