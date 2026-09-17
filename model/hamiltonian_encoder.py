"""
Hamiltonian coefficient encoder + normalizer, reproducing Section 3.2.2 and
Appendix A.4 of ADAPT-GQE (arXiv:2607.22468): a residual-MLP encoder mapping
a canonically-ordered vector of Pauli-term coefficients to the language
model's embedding dimension, used to inject molecular information at the
<ham> token positions.
"""
import numpy as np
import torch
import torch.nn as nn


class HamiltonianNormalizer:
    """Per-coefficient-position max-abs scaling to [-1, 1], fit on the
    TRAINING split only (Sec. 3.2.2). A structurally-zero term (max-abs 0
    across the whole training set) is left unscaled rather than divided by
    zero."""

    def __init__(self, scale: np.ndarray = None):
        self.scale = scale

    def fit(self, vectors: np.ndarray) -> "HamiltonianNormalizer":
        scale = np.abs(vectors).max(axis=0)
        scale[scale == 0] = 1.0
        self.scale = scale
        return self

    def transform(self, vectors: np.ndarray) -> np.ndarray:
        return vectors / self.scale

    def save(self, path):
        np.save(path, self.scale)

    @classmethod
    def load(cls, path) -> "HamiltonianNormalizer":
        return cls(scale=np.load(path))


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, ffn_mult, dropout):
        super().__init__()
        hidden = int(dim * ffn_mult)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class HamiltonianEncoder(nn.Module):
    """f_enc: R^d -> R^D. Input LayerNorm + linear projection to the
    internal hidden dim H, `depth` residual MLP blocks, final LayerNorm +
    linear projection to the text-module embedding dim D. Matches the
    per-problem-size hyperparameters in Appendix Table 3:
      12q:  d=1819, H=2048, depth=4, ffn_mult=2.0, dropout=0.2
      14q:  d=3382, H=3072, depth=4, ffn_mult=1.5, dropout=0.1 (eps=5mHa) / 0.2 (eps=16mHa)
      16q:  d=5793, H=4096, depth=4, ffn_mult=1.0, dropout=0.2
    D = 1024 for the Gemma backbone, 5120 for Nemotron.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, depth=4, ffn_mult=2.0, dropout=0.2):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, ffn_mult, dropout) for _ in range(depth)]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(x)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_norm(x)
        return self.output_proj(x)
