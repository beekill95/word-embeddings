from __future__ import annotations

import torch
from torch import nn


class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class SkipGram(Word2Vec):
    def __init__(self, vocab_size: int, embedding_size: int) -> None:
        super().__init__(vocab_size, embedding_size)

        self.linear = nn.Linear(embedding_size, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.linear(x)
        return x


class CBOW(Word2Vec):
    # TODO
    ...
