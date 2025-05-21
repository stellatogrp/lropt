import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.transforms import CorrCholeskyTransform
import numpy as np
import scipy as sc

import lropt.train.settings as settings
from sklearn.neighbors import NearestNeighbors


torch.set_default_dtype(settings.DTYPE)

class DeepNormalModel(torch.nn.Module):
    def __init__(
        self,knn_cov = False,
                 n_neighbors = 10, knn_scale = 1
    ):
        super().__init__()
        self.jitter = 1e-6
        self.n_neighbors = n_neighbors
        self.knn_cov = knn_cov
        self.u_train_vals = None
        self.knn_scale = knn_scale


    def initialize(self,a_tch,b_tch,trainer):
        n_inputs, a_out,n_hidden = trainer.initialize_predictor_dims()
        self.shared = torch.nn.Linear(n_inputs, n_hidden)
        self.mean_hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.mean_linear = torch.nn.Linear(n_hidden, n_hidden)

        self.cho_hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.cho_elements_module = nn.Linear(
            n_hidden, n_hidden * (n_hidden - 1) // 2)

        self.cho_diag_hidden = torch.nn.Linear(n_hidden, n_hidden)
        self.cho_diag = torch.nn.Linear(n_hidden, n_hidden)


        t_radius = torch.tensor(2.)
        self.radius = nn.Parameter(t_radius, requires_grad=True)
        self.input_to_output = nn.Linear(n_inputs, 1, bias=True)
        if self.knn_cov:
            self.knn_fit(trainer)
        self.train()


    def forward(self, x,a_shape,b_shape,train_flag):

        shared_1 = self.shared(x)
        shared_2 = F.tanh(shared_1)      #leaky_relu(shared, negative_slope=0.01)

        mean_hidden_1 = self.mean_hidden(shared_2)
        mean_hidden_2 = F.tanh(mean_hidden_1)     #leaky_relu(mean_hidden, negative_slope=0.01)
        mean = self.mean_linear(mean_hidden_2)


        cho_hidden= self.cho_hidden(shared_2)
        cho_hidden1= F.tanh(cho_hidden)
        cho_elements = self.cho_elements_module(cho_hidden1)
        cho_elements = F.leaky_relu(cho_elements, negative_slope=0.01)
        cho = CorrCholeskyTransform()(cho_elements)

        cho_dh = self.cho_diag_hidden(shared_2)
        cho_dh1= F.tanh(cho_dh)
        cho_d = self.cho_diag(cho_dh1)
        cho_d1 = F.softplus(cho_d)
        diagonal_indices = torch.arange(cho.shape[2])

        cho[:, diagonal_indices, diagonal_indices] = cho_d1

        r = torch.abs(self.radius)

        cho_scaled =  r*cho  # torch.matmul(cho, y)
        x = self.input_to_output(x)
        #torch.clamp(radius, min = 1e-2)
        if self.knn_cov:
            new_a_tch = self.knn_predict(x)
            a_tch = (1-self.knn_scale)*cho_scaled + self.knn_scale*new_a_tch
        else:
            a_tch = cho_scaled
        return a_tch, mean, r

    def knn_fit(self,trainer):
        knn = NearestNeighbors(n_neighbors=self.n_neighbors)
        x = trainer.create_input_tensors(trainer.x_train_tch)
        knn.fit(x.detach())
        self.knn = knn
        self.u_train_vals = trainer.u_train_set

    def knn_predict(self,x):
        neighbors = self.knn.kneighbors(x.detach(), return_distance=False)
        atchs = []
        for i in range(x.shape[0]):
            atchs.append(sc.linalg.sqrtm(np.cov(self.u_train_vals[neighbors[i]].T)))
        atchs = np.stack(atchs)
        a_tch = torch.tensor(atchs, dtype=torch.double, requires_grad=True)
        return a_tch