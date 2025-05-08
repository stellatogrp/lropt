import torch

import lropt.train.settings as settings
from lropt.train.predictors.linear import LinearPredictor

torch.set_default_dtype(settings.DTYPE)


class NNPredictor(LinearPredictor):

    def __init__(self,predict_mean = False,pretrain = False,
                 dropout_prob = 0.1,epochs = 100,lr = 0.1):
        super(NNPredictor, self).__init__(predict_mean,pretrain,epochs,lr)
        self.dropout_prob = dropout_prob

    def initialize(self,a_tch,b_tch,trainer):
        """Initialize the parameters"""
        num_in,num_a, num_b = trainer.initialize_predictor_dims()
        num_hidden = num_a + num_b
        self.linear = torch.nn.Linear(num_in,num_hidden)
        self.tanh = torch.nn.Tanh()
        # self.bn = torch.nn.BatchNorm1d(num_out)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(self.dropout_prob)
        self.linear1 = torch.nn.Linear(num_hidden,num_hidden)
        # self.customize(a_totsize,a_tch,b_tch,trainer.settings.init_bias,
        # trainer.settings.init_weight,trainer.settings.random_init)
        # if self.predict:
        #     input_tensors = trainer.create_input_tensors(trainer.x_train_tch)
        #     self.gen_weights(input_tensors,trainer.u_train_tch,a_tch)

    def forward(self, x,a_shape,b_shape,train_flag):
        """create a_tch and b_tch using the predictor"""
        out = self.linear(x)
        # out = self.tanh(out)
        out = self.relu(out)
        # out = self.bn(out)
        # out = self.dropout(out)
        # try batchnorm layer
        out = self.linear1(out)
        raw_a = out[:, : a_shape[0] * a_shape[1]]
        raw_b = out[:, a_shape[0] * a_shape[1] :]
        a_tch = raw_a.view(out.shape[0], a_shape[0], a_shape[1])
        b_tch = raw_b.view(out.shape[0], b_shape[0])
        if not train_flag:
            a_tch = torch.tensor(a_tch, requires_grad=False)
            b_tch = torch.tensor(b_tch, requires_grad=False)
        return a_tch, b_tch
