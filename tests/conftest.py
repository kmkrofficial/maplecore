"""Shared test fixtures. Tests never download models or datasets."""

import pytest
import torch
import torch.nn as nn

from maplecore.indexer import Block, Index


class DotModel(nn.Module):
    """Stand-in for MapleNet: logit = scale * <query, block>."""

    def __init__(self, dim: int, scale: float = 10.0):
        super().__init__()
        self.dim = dim
        self.scale = scale
        self.dummy = nn.Parameter(torch.zeros(1))  # so .to()/.parameters() work

    def forward(self, x):
        q, b = x[:, : self.dim], x[:, self.dim:]
        return self.scale * (q * b).sum(dim=1)


def make_index(num_blocks: int, dim: int = 16, seed: int = 0) -> Index:
    g = torch.Generator().manual_seed(seed)
    emb = torch.randn(num_blocks, dim, generator=g) * 0.05
    blocks = [Block(id=i, text=f"block {i}", start_char=i * 10, end_char=i * 10 + 10)
              for i in range(num_blocks)]
    return Index(blocks=blocks, embeddings=emb, chunk_size=10, embedding_model="test-encoder")


@pytest.fixture
def dot_model():
    return DotModel(dim=16)
