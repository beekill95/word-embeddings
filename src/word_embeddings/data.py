from __future__ import annotations

from typing import Literal

import datasets


def get_dataset(
    name: Literal["wikitext-2", "wikitext-103"],
    ds_type: Literal["train", "validation", "test"],
) -> datasets.Dataset:
    if name == "wikitext-2":
        ds = datasets.load_dataset("wikitext", "wikitext-2-v1")
    elif name == "wikitext-103":
        ds = datasets.load_dataset("wikitext", "wikitext-103-v1")
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    assert isinstance(ds, datasets.DatasetDict), "Unexpected return dataset type."

    return ds[ds_type].with_format("torch")
