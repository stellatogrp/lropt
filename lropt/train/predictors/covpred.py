import numpy as np
import torch
from scipy.optimize import fmin_l_bfgs_b

import lropt.train.settings as settings

torch.set_default_dtype(settings.DTYPE)

class CovPredictor(torch.nn.Module):

    def __init__(self,pretrain = False,lr=0.1,epochs=100):
        super(CovPredictor, self).__init__()
        self.pretrain = pretrain
        self.epochs = epochs
        self.lr = lr

    def initialize(self,a_tch,b_tch,trainer,epsilon=1e-3, lam_1=0.0, lam_2=0.0, lam_3=0.0):
        """Initialize the parameters"""
        X = trainer.create_input_tensors(trainer.x_train_tch)
        Y = trainer.u_train_tch
        X = X.detach().numpy()
        Y = Y.detach().numpy()
        T, p = X.shape
        T, n = Y.shape
        diag_rows, diag_cols = np.diag_indices(n)
        off_diag_cols, off_diag_rows = np.triu_indices(n, k=1)
        k = off_diag_rows.size
        def f(x):
            Aplus = x[:n*p].reshape(n, p)
            Aneg = x[n*p:n*p*2].reshape(n, p)
            bplus = x[n*p*2:n*(p*2+1)]
            C = x[n*(p*2+1):n*(p*2+1)+k*p].reshape(k, p)
            d = x[n*(p*2+1)+k*p:n*(p*2+1)+k*p+k]
            A = Aplus - Aneg
            b = (Aplus + Aneg) @ np.ones(p) + epsilon + bplus
            Areg = x[n*(p*2+1)+k*p+k:n*(p*2+1)+k*p+k+p*n].reshape(n, p)
            breg = x[n*(p*2+1)+k*p+k+p*n:n*(p*2+1)+k*p+k+p*n+n]

            pred = X @ Areg.T + breg

            L = np.zeros((T, n, n))

            L[:, diag_rows, diag_cols] = X @ A.T + b
            L[:, off_diag_rows, off_diag_cols] = X @ C.T + d

            f = -np.log(L[:, diag_rows, diag_cols]).sum() / T + \
                .5 * np.square((Y[:,:,None] * L).sum(axis=1) - pred).sum() / T + \
                lam_1 / 2 * (np.sum(np.square(A)) + np.sum(np.square(C))) + \
                lam_2 / 2 * (np.sum(np.square(b - 1)) + np.sum(np.square(d))) + \
                lam_3 / 2 * np.sum(np.square(Areg))

            L_grad = np.zeros((T, n, n))
            L_grad[:, diag_rows, diag_cols] = -1.0 / L[:, diag_rows, diag_cols]
            L_grad += Y[:,:,None] * (L.transpose(0,2,1) * Y[:,None,:]).sum(axis=2)[:,None,:]
            L_grad -= Y[:,:,None] * pred[:,None,:]

            pred_grad = -(Y[:,:,None] * L).sum(axis=1) + pred

            Aplus_grad = (L_grad[:, diag_rows, diag_cols][:,:,None] * (
                X[:,None,:] + 1)).sum(axis=0) / T + \
                    lam_1 * A + lam_2 * np.outer(b - 1, np.ones(p))
            Aneg_grad = (L_grad[:, diag_rows,
                                diag_cols][:,:,None] * (-X[:,None,:] + 1)
                                ).sum(axis=0) / T - \
                    lam_1 * A + lam_2 * np.outer(b - 1, np.ones(p))
            C_grad = (L_grad[:, off_diag_rows, off_diag_cols][:,:,None] *\
                      X[:,None,:]).sum(axis=0) / T + lam_1 * C

            bplus_grad = L_grad[:, diag_rows, diag_cols].sum(axis=0) / T + lam_2 * (b - 1)
            d_grad = L_grad[:, off_diag_rows, off_diag_cols].sum(axis=0) / T + lam_2 * d

            Areg_grad = pred_grad.T @ X / T + lam_3 * Areg
            breg_grad = pred_grad.sum(axis=0) / T

            grad = np.concatenate([
                Aplus_grad.flatten(),
                Aneg_grad.flatten(),
                bplus_grad.flatten(),
                C_grad.flatten(),
                d_grad.flatten(),
                Areg_grad.flatten(),
                breg_grad.flatten()
            ])
            return f, grad
        bounds = [(0, np.inf)] * (n*p) + [(0,np.inf)] * (n*p) + \
                [(0, np.inf)] * n + [(-np.inf, np.inf)] * k * p + [(-np.inf, np.inf)] * k + \
                [(-np.inf, np.inf)] * p * n + [(-np.inf, np.inf)] * n
        x = np.zeros(len(bounds))
        x[2*n*p:2*n*p+n] = 1 - epsilon
        x, fstar, info = fmin_l_bfgs_b(f, x, bounds=bounds)
        Aplus = x[:n*p].reshape(n, p)
        Aneg = x[n*p:n*p*2].reshape(n, p)
        bplus = x[n*p*2:n*(p*2+1)]
        C = x[n*(p*2+1):n*(p*2+1)+k*p].reshape(k, p)
        d = x[n*(p*2+1)+k*p:n*(p*2+1)+k*p+k]
        A = Aplus - Aneg
        b = (Aplus + Aneg) @ np.ones(p) + epsilon + bplus
        Areg = x[n*(p*2+1)+k*p+k:n*(p*2+1)+k*p+k+p*n].reshape(n, p)
        breg = x[n*(p*2+1)+k*p+k+p*n:n*(p*2+1)+k*p+k+p*n+n]
        self.A = torch.nn.Parameter(torch.tensor(A))
        self.b = torch.nn.Parameter(torch.tensor(b))
        self.C = torch.nn.Parameter(torch.tensor(C))
        self.d = torch.nn.Parameter(torch.tensor(d))
        self.Areg = torch.nn.Parameter(torch.tensor(Areg))
        self.breg = torch.nn.Parameter(torch.tensor(breg))

    def forward(self, x,a_shape,b_shape,train_flag):
        """create a_tch and b_tch using the predictor"""
        n = self.Areg.shape[0]
        N = x.shape[0]
        nu = torch.matmul(x, self.Areg.T) + self.breg
        diag_rows, diag_cols = np.diag_indices(n)
        off_diag_cols, off_diag_rows = np.triu_indices(n, k=1)
        # k = off_diag_rows.size
        L = torch.zeros((N, n, n),dtype = settings.DTYPE)
        L[:, diag_rows, diag_cols] = torch.matmul(x, self.A.T) + self.b
        L[:, off_diag_rows, off_diag_cols] = torch.matmul(x,self.C.T) + self.d
        init_shape = nu[0].shape[0]
        yhat = torch.stack([torch.linalg.solve_triangular(L[i].T,
                                        nu[i].view(init_shape,1),
                                        upper=True).view(
                                            init_shape,) for i in range(N)])
        return torch.stack([torch.linalg.inv(L[i]) for i in range(N)]), yhat, 1
