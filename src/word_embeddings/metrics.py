from __future__ import annotations

from typing import Iterable

import torch
from torcheval.metrics import Metric


class TopKMultilabelAccuracy(Metric[torch.Tensor]):
    eps = 1e-9

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)

        self.threshold = threshold
        self._add_state(
            "intersection",
            torch.tensor(0.0, dtype=torch.float, device=device),
        )
        self._add_state(
            "union",
            torch.tensor(0.0, dtype=torch.float, device=device),
        )

    @torch.inference_mode()
    def compute(self) -> torch.Tensor:
        return self.intersection / (self.union + self.eps)

    @torch.inference_mode()
    def update(self, ypred: torch.Tensor, ytrue: torch.Tensor) -> torch.Tensor:
        """
        Update the internal state
        """
        assert ypred.shape[0] == ytrue.shape[0]
        intersection, union = self._update(
            ypred=ypred,
            ytrue=ytrue,
            threshold=self.threshold,
        )

        self.intersection += intersection
        self.union += union

    @torch.inference_mode()
    def merge_state(
        self,
        metrics: Iterable[TopKMultilabelAccuracy],
    ) -> TopKMultilabelAccuracy:
        for metric in metrics:
            self.intersection += metric.intersection
            self.union += metric.union

        return self

    @staticmethod
    @torch.inference_mode
    @torch.jit.script
    def _update(
        *,
        ypred: torch.Tensor,
        ytrue: torch.Tensor,
        threshold: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        intersection = torch.tensor(0, dtype=torch.float, device=ypred.device)
        union = torch.tensor(0, dtype=torch.float, device=ytrue.device)

        for i in range(ypred.shape[0]):
            true = ytrue[i]
            # FIXME: check model's output shape
            pred = torch.nonzero(ypred[i].squeeze() > threshold).squeeze()

            unique_indices, indices_count = torch.unique(
                torch.concat((true, pred)),
                return_counts=True,
            )

            intersection += torch.sum(indices_count > 1)
            union += unique_indices.shape[0]

        return intersection, union
