from __future__ import annotations

from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from training_loop import TrainingLoop, SimpleTrainingStep

from ..data import build_vocab, get_dataset
from .model import CBOW, SkipGram


def skipgram_collate(
    lines: list[list[int]],
    *,
    context_length: int,
    max_length: int = 0,
    skip_short_lines: bool = True,
    randomly_truncate_long_lines: bool = True,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    inputs = []
    outputs = []

    context_window = 2 * context_length + 1

    for line in lines:
        if len(line) < context_window and skip_short_lines:
            continue
        elif (max_length > 0) and (len(line) > max_length):
            if randomly_truncate_long_lines:
                # TODO
                idx = torch.randint(0, len(line) - max_length)

            # Truncate line to max length.
            line = line[idx : idx + max_length]

        # Create input & output words, with input word at the center of the window,
        # and output words are every words in the context but not the input.
        # TODO: there might be a bug with skip_short_lines = False
        for idx in range(context_length, len(line) - context_length):
            context_indices = (
                line[idx - context_length : idx] + line[idx + 1 : idx + context_length]
            )
            inputs.extend([idx] * len(context_indices))
            outputs.extend(context_indices)

    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(outputs, dtype=torch.long),
    )


def train(
    model_type: Literal["skip-gram", "cbow"],
    *,
    dataset_name: Literal["wikitext-2", "wikitext-103"],
    embedding_size: int = 100,
    batch_size: int = 64,
    device: str = "cpu",
    epochs: int = 5,
):
    def convert_line_to_indices(line: str):
        return vocab(tokenizer(line))

    train_ds = get_dataset(dataset_name, "train")
    val_ds = get_dataset(dataset_name, "val")

    # Build vocabulary.
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(train_ds, tokenizer=tokenizer)

    # Convert strings to indices.
    train_ds = train_ds.map(lambda line: convert_line_to_indices(line["text"]))
    val_ds = val_ds.map(lambda line: convert_line_to_indices(line["text"]))

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