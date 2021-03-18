import torch as th
from torch import nn

from .probe import Probe


class EuclideanProbeBase(Probe):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def distance(self, transformed):
        """
        Computes all n^2 pairs of distances after projection
        for each sentence in a batch.
        Note that due to padding, some distances will be non-zero for pads.
        Computes (B(h_i-h_j))^T(B(h_i-h_j)) for all i,j
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
        diffs = transformed - transposed
        squared_diffs = diffs.pow(2)
        squared_distances = th.sum(squared_diffs, -1)
        return squared_distances

    def depth(self, transformed):
        """ 
        Computes all n depths after projection
        for each sentence in a batch.

        Computes (Bh_i)^T(Bh_i) for all i

        Args:
            batch: a batch of word representations of the shape
            (batch_size, max_seq_len, representation_dim)
        Returns:
            A tensor of depths of shape (batch_size, max_seq_len)
        """
        batchlen, seqlen, rank = transformed.size()
        norms = th.bmm(
            transformed.reshape(batchlen * seqlen, 1, rank),
            transformed.reshape(batchlen * seqlen, rank, 1),
        )
        norms = norms.reshape(batchlen, seqlen)
        return norms


class EuclideanProbe(EuclideanProbeBase):
    """ 
    Computes squared L2 distance or depth after projection by a matrix.
    
    For a batch of sentences, computes all n^2 pairs of distances 
    for each sentence in the batch.
    """

    def __init__(self, **kwargs):
        print("Constructing EuclideanProbe")
        super().__init__(**kwargs)

        self.proj = nn.Parameter(data=th.zeros(self.dim_in, self.dim_out))
        nn.init.uniform_(self.proj, -0.05, 0.05)

    def project(self, batch):
        """
        Transforme batch via probe
        
        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        """
        transformed = th.matmul(batch, self.proj)
        return transformed


class LocalEuclideanProbe(EuclideanProbeBase):
    """ 
    Computes squared L2 distance or depth after projection by rnn.
    
    For a batch of sentences, computes all n^2 pairs of distances 
    for each sentence in the batch.
    """

    def __init__(self, **kwargs):
        print("Constructing LocalEuclideanProbe")
        super().__init__(**kwargs)

        self.proj = nn.GRU(self.dim_in, self.dim_out, batch_first=True)

    def project(self, batch):
        """
        Transforme batch via probe
        
        Args:
            batch: a batch of word representations of the shape
                (batch_size, max_seq_len, representation_dim)
        """
        transformed, _ = self.proj(batch)
        return transformed
