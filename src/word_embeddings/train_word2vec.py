from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchtext.vocab import Vocab
from training_loop import TrainingLoop, SimpleTrainingStep

from .data import build_vocab, get_dataset
from .word2vec import CBOW, SkipGram


def convert_line_to_indices(tokenizer, vocab: Vocab, line: str):
    return


def skipgram_collate(
    lines: list[list[int]],
    *,
    context_length: int,
    max_length: int = -1,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    pass


def train(
    model_type: Literal["skip-gram", "cbow"],
    *,
    dataset_name: Literal["wikitext-2", "wikitext-103"],
    embedding_size: int = 100,
    batch_size: int = 64,
    device: str = "cpu",
    epochs: int = 5,
):
    train_ds = get_dataset(dataset_name, "train")
    val_ds = get_dataset(dataset_name, "val")

    # Build vocabulary.
    vocab = build_vocab(train_ds)

    if model_type == "skip-gram":
        model = SkipGram(len(vocab), embedding_size)
    elif model_type == "cbow":
        model = CBOW(len(vocab), embedding_size)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # TODO: refactor this to reusable function.
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        # TODO: Implement cbow collate.
        collate_fn=lambda lines: skipgram_collate(lines, context_length=5),
    )
    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        # TODO: Implement cbow collate.
        collate_fn=lambda lines: skipgram_collate(lines, context_length=5),
    )

    loop = TrainingLoop(
        model,
        step=SimpleTrainingStep(),
        device=device,
    )
    histories = loop.fit(
        train_dataloader,
        val_dataloader,
        epochs=epochs,
    )

    return {"model": model, "vocab": vocab}, *histories
