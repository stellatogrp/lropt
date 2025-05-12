import numpy as np
import scipy as sc
import torch
from sklearn.neighbors import NearestNeighbors

import lropt.train.settings as settings

torch.set_default_dtype(settings.DTYPE)

class LinearPredictor(torch.nn.Module):

    def __init__(self,predict_mean = False, pretrain = False,
                 epochs = 100,lr = 0,knn_cov = False,
                 n_neighbors = 10, knn_scale = 1):
        super(LinearPredictor, self).__init__()
        self.predict = predict_mean
        self.pretrain = pretrain
        self.epochs = epochs
        self.lr = lr
        self.n_neighbors = n_neighbors
        self.knn_cov = knn_cov
        self.u_train_vals = None
        self.knn_scale = knn_scale

    def initialize(self,a_tch,b_tch,trainer):
        """Initialize the parameters"""
        num_in, a_out,b_out = trainer.initialize_predictor_dims()
        self.linear_mean = torch.nn.Linear(num_in,b_out)
        self.linear_cov = torch.nn.Linear(num_in,a_out)
        self.customize(a_tch,b_tch,trainer.settings.init_bias,trainer.settings.init_weight,trainer.settings.random_init)
        if self.predict:
            input_tensors = trainer.create_input_tensors(trainer.x_train_tch)
            self.gen_weights(input_tensors,trainer.u_train_tch)
        if self.knn_cov:
            self.knn_fit(trainer)
        self.train()
        if self.pretrain:
            self.pretrain_func(trainer)


    def customize(self,a_tch,b_tch,init_bias,init_weight,random_init):
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
        self.linear_mean.bias.data = b_tch
        if not random_init:
            with torch.no_grad():
                # torch_b = b_tch
                torch_a = a_tch.flatten()
                # torch_concat = torch.hstack([torch_a, torch_b])
            self.linear_mean.weight.data.fill_(0.000)
            self.linear_cov.weight.data.fill_(0.000)
            self.linear_mean.bias.data = b_tch
            self.linear_cov.bias.data = torch_a
            if init_weight is not None:
                self.linear_mean.weight.data = init_weight.clone().detach().requires_grad_(True)
            if init_bias is not None:
                self.linear_mean.bias.data = torch.tensor(
                    init_bias, dtype=torch.double, requires_grad=True
                )

    def gen_weights(self,input,output):
        """Set the weights of the predictor using lstsq
        Args:
        input
            context data to train on
        output
            uncertain parameter data to train on
        """
        input = input.detach().numpy()
        output = output.detach().numpy()
        # num_in = input.shape[1]
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
        # self.init_bias = np.hstack([a_tch.detach().numpy().flatten(),mults_mean_bias])
        # self.init_weight = np.vstack([np.zeros((m*m,num_in)),mults_mean_weight])
        self.linear_mean.weight.data = torch.tensor(
            mults_mean_weight, dtype=torch.double, requires_grad=True
        )
        self.linear_mean.bias.data = torch.tensor(
            mults_mean_bias, dtype=torch.double, requires_grad=True
        )

    def forward(self, x,a_shape,b_shape,train_flag):
        """create a_tch and b_tch using the predictor"""
        out_a = self.linear_cov(x)
        out_b = self.linear_mean(x)
        # raw_a = out[:, : a_shape[0] * a_shape[1]]
        # raw_b = out[:, a_shape[0] * a_shape[1] :]
        a_tch = out_a.view(out_b.shape[0], a_shape[0], a_shape[1])
        b_tch = out_b.view(out_b.shape[0], b_shape[0])
        if self.knn_cov:
            new_a_tch = self.knn_predict(x)
            a_tch = (1-self.knn_scale)*a_tch + self.knn_scale*new_a_tch
        if not train_flag:
            a_tch = a_tch.detach().clone()
            b_tch = b_tch.detach().clone()
        return a_tch, b_tch, 1

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

    def pretrain_func(self,trainer):
        # call it pre-training
        if self.pretrain:
            assert (len(trainer.x_train_tch) != 0) and (len(trainer.u_train_tch) != 0)
            pred_optimizer = torch.optim.SGD(
                self.linear_mean.parameters(),
                lr = self.lr)
            criterion = torch.nn.MSELoss()
            epochs=self.epochs
            for epoch in range(epochs):
                _,yhat,_=trainer.create_predictor_tensors(trainer.x_train_tch)
                loss=criterion(yhat,trainer.u_train_tch)
                pred_optimizer.zero_grad()
                loss.backward()
                pred_optimizer.step()
