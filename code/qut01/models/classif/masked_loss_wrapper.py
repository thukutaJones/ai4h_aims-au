from typing import Union

import torch
from torch import Tensor, nn
from torch.nn.modules.loss import _WeightedLoss


class MaskedLossWrapper(_WeightedLoss):
    """A wrapper class for a PyTorch loss function, used to ignore some values in the targets."""

    def __init__(self, loss_fun_to_wrap: Union[nn.Module, callable], ignore_index: int, reduction: str = "mean"):
        """Initialize the MaskedLossWrapper.

        Args:
            loss_fun_to_wrap (nn.Module): The loss function to wrap.
            ignore_index (int): values (in the targets) indicating the indices to be ignored.
            reduction (str): Reduction to apply - only mean and non are supported.
        """
        super().__init__()
        self.loss_to_wrap = loss_fun_to_wrap
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """Forward pass of the loss computation.

        Args:
            input (Tensor): The input tensor.
            target (Tensor): The target tensor.

        Returns:
            Tensor: Computed loss.
        """
        unmasked_input_location = target >= 0

        fixed_targets = target.clone()
        fixed_targets[fixed_targets == self.ignore_index] = 0

        non_masked_loss = self.loss_to_wrap(input, fixed_targets)
        assert non_masked_loss.shape == target.shape
        masked_loss = non_masked_loss * unmasked_input_location

        if self.reduction == "mean":
            return torch.mean(masked_loss)
        elif self.reduction == "none":
            return masked_loss
        else:
            raise NotImplementedError(f"reduction {self.reduction} not supported.")
