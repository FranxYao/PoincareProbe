import torch as th
from torch import nn
import geoopt as gt


class HyperGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, ball):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ball = ball

        k = (1 / hidden_size) ** 0.5
        self.w_z = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k)
        )
        self.w_r = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k)
        )
        self.w_h = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, hidden_size).uniform_(-k, k)
        )
        self.u_z = gt.ManifoldParameter(
            gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k)
        )
        self.u_r = gt.ManifoldParameter(
            gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k)
        )
        self.u_h = gt.ManifoldParameter(
            gt.ManifoldTensor(input_size, hidden_size).uniform_(-k, k)
        )
        self.b_z = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, manifold=self.ball).zero_()
        )
        self.b_r = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, manifold=self.ball).zero_()
        )
        self.b_h = gt.ManifoldParameter(
            gt.ManifoldTensor(hidden_size, manifold=self.ball).zero_()
        )

    def transition(self, W, h, U, x, hyp_b):
        W_otimes_h = self.ball.mobius_matvec(W, h)
        U_otimes_x = self.ball.mobius_matvec(U, x)
        Wh_plus_Ux = self.ball.mobius_add(W_otimes_h, U_otimes_x)

        return self.ball.mobius_add(Wh_plus_Ux, hyp_b)

    def forward(self, hyp_x, hidden):
        z = self.transition(self.w_z, hidden, self.u_z, hyp_x, self.b_z)
        z = th.sigmoid(self.ball.logmap0(z))

        r = self.transition(self.w_r, hidden, self.u_r, hyp_x, self.b_r)
        r = th.sigmoid(self.ball.logmap0(r))

        r_point_h = self.ball.mobius_pointwise_mul(hidden, r)
        h_tilde = self.transition(self.w_h, r_point_h, self.u_r, hyp_x, self.b_h)

        minus_h_oplus_htilde = self.ball.mobius_add(-hidden, h_tilde)
        new_h = self.ball.mobius_add(
            hidden, self.ball.mobius_pointwise_mul(minus_h_oplus_htilde, z)
        )

        return new_h


class HyperGRU(nn.Module):
    def __init__(self, input_size, hidden_size, ball, default_dtype=th.float64):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ball = ball
        self.default_dtype = default_dtype

        self.gru_cell = HyperGRUCell(hidden_size, hidden_size, ball=self.ball)

    def init_gru_state(self, batch_size, hidden_size, cuda_device):
        return th.zeros(
            (batch_size, hidden_size), dtype=self.default_dtype, device=cuda_device
        )

    def forward(self, inputs):
        hidden = self.init_gru_state(inputs.shape[0], self.hidden_size, inputs.device)
        outputs = []
        for x in inputs.transpose(0, 1):
            hidden = self.gru_cell(x, hidden)
            outputs += [hidden]
        return th.stack(outputs).transpose(0, 1)
