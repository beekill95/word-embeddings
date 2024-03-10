from __future__ import annotations

import torch
from torch import nn


class MultiLabelSoftMarginLoss(nn.Module):
    """
    Similar to torch's MultiLabelSoftMarginLoss,
    but accept `target` as class indices instead of binary tensor.
    """

    def __init__(self, num_classes: int, *args, logits: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_classes = num_classes
        self.logits = logits

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]

        loss = torch.tensor(0.0, dtype=inputs.dtype, device=inputs.device)

        for i in range(batch_size):
            # FIXME: check the outputs of the model. it should be (N, C),
            # but it actually is (N, 1, C).
            pred = inputs[i].squeeze()
            labels = targets[i]

            mask = torch.ones(self.num_classes, dtype=bool, device=inputs.device)
            mask[labels] = False

            if not self.logits:
                positive_pred = pred[labels]
                negative_pred = pred[mask]
            else:
                expo_pred = torch.exp(-pred)
                positive_pred = 1.0 / (1.0 + expo_pred[labels])
                negative_pred = expo_pred[mask] / (1.0 + expo_pred[mask])

            loss += torch.mean(torch.log(positive_pred)) + torch.mean(
                torch.log(negative_pred)
            )

        return -loss / batch_size
