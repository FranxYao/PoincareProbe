import torch as th
from torch import nn
import geoopt as gt

from .probe import Probe
from .hyperrnn import HyperGRU


class PoincareProbeBase(Probe):
    def __init__(self, curvature: float, dim_hidden: int, **kwargs):
        super().__init__(**kwargs)

        self.ball = gt.Stereographic(curvature)
        self.dim_hidden = dim_hidden

    def distance(self, transformed):
        """
        Computes all n^2 pairs of distances after exponential map
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.

        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        Returns:
            A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        squared_distances = self.ball.dist2(transformed, transposed)
        return squared_distances

    def depth(self, transformed):
        """
        Computes all n depths after exponential map
        for each sentence in a batch.

        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        Returns:
            A tensor of depths of shape (batch_size, max_seq_len)
        """
        batchlen, seqlen, rank = transformed.size()
        norms = self.ball.dist0(transformed.reshape(batchlen * seqlen, 1, rank))
        norms = norms.reshape(batchlen, seqlen) ** 2
        return norms


class PoincareProbe(PoincareProbeBase):
    """
    Computes squared poincare distance or depth after exponential map
    and a projection by Mobius mat-vec-mul.
    
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, **kwargs):
        print("Constructing PoincareProbe")
        super().__init__(**kwargs)

        self.proj = nn.Parameter(data=th.zeros(self.dim_in, self.dim_hidden))
        self.trans = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))

        nn.init.uniform_(self.proj, -0.05, 0.05)
        nn.init.uniform_(self.trans, -0.05, 0.05)

    def project(self, batch):
        """
        Transforme batch via probe
        
        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        """
        transformed = th.matmul(batch, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        return transformed


class LocalPoincareProbe(PoincareProbeBase):
    """
    Computes squared poincare distance or depth by rnn with exponential map
    and a projection by Mobius mat-vec-mul.
    
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, **kwargs):
        print("Constructing LocalPoincareProbe")
        super().__init__(**kwargs)

        self.proj = nn.GRU(self.dim_in, self.dim_hidden, batch_first=True)
        self.trans = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))

        nn.init.uniform_(self.trans, -0.05, 0.05)

    def project(self, batch):
        """
        Transforme batch via probe
        
        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        """
        transformed, _ = self.proj(batch)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        return transformed


class LocalPoincareProbeWithHyperGRU(PoincareProbeBase):
    """
    Computes squared poincare distance or depth by rnn with exponential map
    and a projection by Mobius mat-vec-mul.
    
    For a batch of sentences, computes all n^2 pairs of distances
    for each sentence in the batch.
    """

    def __init__(self, **kwargs):
        print("Constructing LocalPoincareProbeWithHyperGRU")
        super().__init__(**kwargs)

        self.proj = nn.Parameter(data=th.zeros(self.dim_in, self.dim_hidden))
        self.trans = nn.HyperGRU(
            self.dim_hidden,
            self.dim_out,
            ball=self.ball,
            default_dtype=self.default_dtype,
        )

        nn.init.uniform_(self.proj, -0.05, 0.05)

    def project(self, batch):
        """
        Transforme batch via probe
        
        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        """
        transformed = th.matmul(batch, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed, _ = self.trans(transformed)
        return transformed


class PoincareProbeNoSquare(Probe):
    """
    Poincare distance directly corresponds to tree distance
    no squared distance used
    """

    def __init__(self, curvature: float, dim_hidden: int, **kwargs):
        print("Constructing PoincareProbeNoSquare")
        super().__init__(**kwargs)

        self.ball = gt.Stereographic(curvature)
        self.dim_hidden = dim_hidden

        self.proj = nn.Parameter(data=th.zeros(self.dim_in, self.dim_hidden))
        self.trans = nn.Parameter(data=th.zeros(self.dim_out, self.dim_hidden))

        nn.init.uniform_(self.proj, -0.05, 0.05)
        nn.init.uniform_(self.trans, -0.05, 0.05)

    def distance(self, transformed):
        """
        Computes all n^2 pairs of distances after exponential map
        for each sentence in a batch.

        Note that due to padding, some distances will be non-zero for pads.

        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        Returns:
            A tensor of distances of shape (batch_size, max_seq_len, max_seq_len)
        """
        batchlen, seqlen, rank = transformed.size()
        transformed = transformed.unsqueeze(2)
        transformed = transformed.expand(-1, -1, seqlen, -1)
        transposed = transformed.transpose(1, 2)
        squared_distances = self.ball.dist(transformed, transposed)
        return squared_distances

    def depth(self, transformed):
        """
        Computes all n depths after exponential map
        for each sentence in a batch.

        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        Returns:
            A tensor of depths of shape (batch_size, max_seq_len)
        """
        batchlen, seqlen, rank = transformed.size()
        norms = self.ball.dist0(transformed.reshape(batchlen * seqlen, 1, rank))
        norms = norms.reshape(batchlen, seqlen)
        return norms

    def project(self, batch):
        """
        Transforme batch via probe
        
        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        """
        transformed = th.matmul(batch, self.proj)
        transformed = self.ball.expmap0(transformed)
        transformed = self.ball.mobius_matvec(self.trans, transformed)
        return transformed
