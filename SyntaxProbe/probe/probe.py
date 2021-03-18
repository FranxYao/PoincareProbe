from abc import ABCMeta, abstractmethod

import torch as th
from torch import nn


class Probe(nn.Module):
    def __init__(self, device, dim_in: int, dim_out: int, default_dtype=th.float32):
        super().__init__()
        self.device = device
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.default_dtype = default_dtype

    @abstractmethod
    def project(self, batch):
        pass

    @abstractmethod
    def distance(self, transformed):
        pass

    @abstractmethod
    def depth(self, transformed):
        pass

    def forward(self, batch, task: str):
        """
        Computes a tensor for the task

        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        Returns:
            A tensor for the task
        """
        transformed = self.project(batch)

        if task == "distance":
            return self.distance(transformed)
        elif task == "depth":
            return self.depth(transformed)
        elif task == "both":
            return self.distance(transformed), self.depth(transformed)
