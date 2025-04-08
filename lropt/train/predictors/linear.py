import torch

import lropt.train.settings as settings

torch.set_default_dtype(settings.DTYPE)

class LinearPredictor(torch.nn.Module):

    def __init__(self):
        super(LinearPredictor, self).__init__()

    def initialize(self,num_in,num_out):
        """Initialize the parameters"""
        self.linear = torch.nn.Linear(num_in,num_out)

    def customize(self,a_totsize,a_tch,b_tch,init_bias,init_weight,random_init):
        """Set the weights of the predictor using mean-variance or given info"""
        # self.linear.bias.data[a_totsize:] = b_tch
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
