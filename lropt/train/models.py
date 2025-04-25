import torch

torch.set_default_dtype(torch.float64)

class TrainerModels():
    def __init__(self,*kwargs):
        self.layer_dims = list(*kwargs)
        self.num_dims = len(self.layer_dims)

    def initialize(self,input_dim,output_dim):
        if self.num_dims == 0:
            # only a single layer
            self.linear = torch.nn.Linear(input_dim, output_dim, bias = True)
            self.model = torch.nn.Sequential(self.linear)
        elif self.num_dims == 1:
            # two layers
            self.linear = torch.nn.Linear(input_dim, self.layer_dims[0], bias = True)
            self.out = torch.nn.Linear(self.layer_dims[0], output_dim, bias = True)
            self.model = torch.nn.Sequential(self.linear,self.out)
        else:
            # multiple layers
            self.linear = torch.nn.Linear(input_dim, self.layer_dims[0], bias = True)
            layers = [self.linear]
            for i in range(self.num_dims-1):
                new_layer = torch.nn.Linear(self.layer_dims[i], self.layer_dims[i+1], bias = True)
                layers.append(new_layer)
            self.out = torch.nn.Linear(self.layer_dims[i+1], output_dim, bias = True)
            layers.append(self.out)
            self.model = torch.nn.Sequential(*layers)
        return self.model

    def create_tensors(self,input):
        return self.model(input)

    def forward(self,input):
        return

    # Linear with single layer
    # feed forward with relu and arbitrary layers
    # covariance predictor
