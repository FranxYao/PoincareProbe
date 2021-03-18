import math

import torch as th
from torch import nn
import geoopt as gt


class EuclideanProbe(nn.Module):
    def __init__(
        self, device, default_dtype=th.float32, layer_num: int = 10,
    ):
        super().__init__()
        self.device = device
        self.default_dtype = default_dtype
        self.layer_num = layer_num

        self.probe_dim = 768
        self.bound = 1 / math.sqrt(self.probe_dim)
        self.pos = nn.Parameter(data=th.zeros(self.probe_dim))
        self.neg = nn.Parameter(data=th.zeros(self.probe_dim))
        nn.init.uniform_(self.pos, -self.bound, self.bound)
        nn.init.uniform_(self.neg, -self.bound, self.bound)

        self.proj = nn.Parameter(data=th.zeros(self.probe_dim, self.probe_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, sequence_output):
        transformed = th.matmul(sequence_output, self.proj)
        pos_logits = (((self.neg - transformed) ** 2).sum(-1)).sum(-1)
        neg_logits = (((self.pos - transformed) ** 2).sum(-1)).sum(-1)

        return th.stack((neg_logits, pos_logits), dim=-1)


class PoincareProbe(nn.Module):
    def __init__(
        self, device, default_dtype=th.float64, layer_num: int = 10,
    ):
        super().__init__()
        self.device = device
        self.default_dtype = default_dtype
        self.ball = gt.Stereographic(-1)
        self.probe_dim = 64
        self.layer_num = layer_num

        self.bound = 1 / math.sqrt(self.probe_dim)
        pos = th.zeros(self.probe_dim).uniform_(-self.bound, self.bound)
        neg = th.zeros(self.probe_dim).uniform_(-self.bound, self.bound)
        pos = self.ball.expmap0(pos)
        neg = self.ball.expmap0(neg)
        self.pos = gt.ManifoldParameter(data=pos, manifold=self.ball)
        self.neg = gt.ManifoldParameter(data=neg, manifold=self.ball)

        self.proj = nn.Parameter(data=th.zeros(768, self.probe_dim))
        self.trans = nn.Parameter(data=th.zeros(self.probe_dim, self.probe_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        nn.init.uniform_(self.trans, -0.05, 0.05)

    def forward(self, sequence_output):
        transformed = th.matmul(sequence_output, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        pos_logits = self.ball.dist(self.neg, transformed).sum(-1)
        neg_logits = self.ball.dist(self.pos, transformed).sum(-1)

        return th.stack((neg_logits, pos_logits), dim=-1)


class EuclideanProbeFixed(nn.Module):
    def __init__(
        self, device, default_dtype=th.float32, layer_num: int = 10,
    ):
        super().__init__()
        self.device = device
        self.default_dtype = default_dtype
        self.layer_num = layer_num

        self.probe_dim = 768
        self.bound = 1 / math.sqrt(self.probe_dim)
        self.pos = th.ones(self.probe_dim).to(device) * self.bound
        self.neg = th.ones(self.probe_dim).to(device) * (-self.bound)

        self.proj = nn.Parameter(data=th.zeros(self.probe_dim, self.probe_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def forward(self, sequence_output):
        transformed = th.matmul(sequence_output, self.proj)
        pos_logits = (((self.neg - transformed) ** 2).sum(-1)).sum(-1)
        neg_logits = (((self.pos - transformed) ** 2).sum(-1)).sum(-1)

        return th.stack((neg_logits, pos_logits), dim=-1)


class PoincareProbeFixed(nn.Module):
    def __init__(
        self, device, default_dtype=th.float64, layer_num: int = 10,
    ):
        super().__init__()
        self.device = device
        self.default_dtype = default_dtype
        self.ball = gt.Stereographic(-1)
        self.probe_dim = 64
        self.layer_num = layer_num

        self.bound = 0.5 / math.sqrt(self.probe_dim)
        self.pos = self.ball.expmap0(th.ones(self.probe_dim).to(device) * self.bound)
        self.neg = self.ball.expmap0(th.ones(self.probe_dim).to(device) * (-self.bound))

        self.proj = nn.Parameter(data=th.zeros(768, self.probe_dim))
        self.trans = nn.Parameter(data=th.zeros(self.probe_dim, self.probe_dim))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        nn.init.uniform_(self.trans, -0.05, 0.05)

    def forward(self, sequence_output):
        transformed = th.matmul(sequence_output, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        pos_logits = self.ball.dist(self.neg, transformed).sum(-1)
        neg_logits = self.ball.dist(self.pos, transformed).sum(-1)

        return th.stack((neg_logits, pos_logits), dim=-1)
