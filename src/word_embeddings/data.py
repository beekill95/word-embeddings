from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Literal
from itertools import chain

import datasets
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torchtext.vocab import build_vocab_from_iterator


def get_dataset(
    name: Literal["wikitext-2", "wikitext-103"],
    ds_type: Literal["train", "validation", "test"],
) -> datasets.Dataset:
    def not_empty(line: dict):
        return bool(line["text"])

    if name == "wikitext-2":
        ds = datasets.load_dataset("wikitext", "wikitext-2-v1")
    elif name == "wikitext-103":
        ds = datasets.load_dataset("wikitext", "wikitext-103-v1")
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    assert isinstance(ds, datasets.DatasetDict), "Unexpected return dataset type."

    return ds[ds_type].with_format("torch").filter(not_empty)


def build_vocab(
    ds: datasets.Dataset,
    *,
    tokenizer: Callable[[str], Iterable[str]] | None = None,
    special_tokens: list[str] | None = None,
    min_freq: int = 10,
    max_tokens: int | None = None,
) -> Vocab:
    if tokenizer is None:
        tokenizer = get_tokenizer("basic_english")

    if special_tokens is None:
        special_tokens = ["<unk>"]

    return build_vocab_from_iterator(
        chain(tokenizer(line["text"]) for line in ds),
        min_freq=min_freq,
        special_first=True,
        special_tokens=special_tokens,
        max_tokens=max_tokens,
    )
