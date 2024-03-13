from __future__ import annotations


import torch
from torch import nn
from torch.masked import masked_tensor


class MultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, logits: bool = True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.logits = logits

    def forward(self, inputs: torch.Tensor, targets: torch.BoolTensor) -> torch.Tensor:
        if self.logits:
            inputs = torch.sigmoid(inputs)

        positive_preds = masked_tensor(inputs, targets, requires_grad=True)
        negative_preds = masked_tensor(inputs, ~targets, requires_grad=True)

        positive_loss = torch.mean(torch.log(positive_preds), dim=1)
        negative_loss = torch.mean(torch.log(-negative_preds + 1.0), dim=1)

        loss = -torch.mean(positive_loss + negative_loss).get_data()
        return loss
