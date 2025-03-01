import typing

import numpy as np
import torch

from qut01.data.batch_utils import BatchDictType


class ConvertItemTensorsToItems(torch.nn.Module):
    """Converts single-item numpy arrays into the `.item()` version of those arrays."""

    def __init__(
        self,
        keys_to_ignore: typing.Optional[typing.Sequence[str]] = (),
    ):
        """Initializes the transform and validates internal settings."""
        super().__init__()
        if not keys_to_ignore:
            keys_to_ignore = []
        assert isinstance(keys_to_ignore, typing.Sequence)
        self.keys_to_ignore = keys_to_ignore

    def forward(self, batch: BatchDictType) -> BatchDictType:
        """Converts single-item numpy arrays into the `.item()` version of those arrays.

        Args:
            batch: the batch dictionary containing potential arrays to be transformed.

        Returns:
            The updated batch dictionary.
        """
        for key in batch.keys():
            if key in self.keys_to_ignore:
                continue
            val = batch[key]
            if not isinstance(val, np.ndarray):
                continue
            if val.size != 1:
                continue
            batch[key] = val.item()
        return batch

    def __repr__(self) -> str:
        """Returns a string representation of the transform object."""
        return f"{self.__class__.__name__}({self.keys_to_ignore=})"
