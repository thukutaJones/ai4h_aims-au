import torch

from qut01.models.classif.masked_loss_wrapper import MaskedLossWrapper


def test_forward__no_masked_elements():
    input = torch.Tensor([1, 2, 3, 4])
    target = torch.Tensor([1, 2, 3, 4])
    mlw = MaskedLossWrapper(loss_fun_to_wrap=torch.add, ignore_index=-1, reduction="none")
    assert mlw(input, target).tolist() == [2, 4, 6, 8]


def test_forward__masked_elements():
    input = torch.Tensor([1, 2, 3, 4])
    target = torch.Tensor([1, 2, 3, -1])
    mlw = MaskedLossWrapper(loss_fun_to_wrap=torch.add, ignore_index=-1, reduction="none")
    assert mlw(input, target).tolist() == [2, 4, 6, 0]
