import typing

import hydra
import omegaconf
import torchvision

import qut01.data.transforms.collate
import qut01.data.transforms.convert_item_tensors
import qut01.data.transforms.identity
import qut01.data.transforms.samplers
import qut01.data.transforms.wrappers
from qut01.data.batch_utils import BatchTransformType
from qut01.data.transforms.convert_item_tensors import ConvertItemTensorsToItems
from qut01.data.transforms.identity import Identity
from qut01.data.transforms.samplers import SentenceSampler
from qut01.data.transforms.wrappers import IterableDataset


def validate_or_convert_transform(
    transform: typing.Optional[BatchTransformType],
) -> BatchTransformType:
    """Validates or converts the given transform object to a proper (torchvision-style) object.

    Args:
        transform: a callable object, DictConfig of a callable object (or a list of those), or a
            list of such objects that constitute the transformation pipeline.

    Returns:
        The "composed" (assembled, and ready-to-be-used) batch transformation pipeline.
    """
    if transform is None:
        transform = [Identity()]  # if nothing is provided, assume that's a shortcut for the identity func
    if isinstance(transform, (dict, omegaconf.DictConfig)) and "_target_" in transform:
        t = hydra.utils.instantiate(transform)
        assert callable(t), f"instantiated transform object not callable: {type(t)}"
        transform = [t]
    if callable(transform):
        transform = [transform]
    assert isinstance(transform, typing.Sequence), (
        "transform must be provided as a callable object, as a DictConfig for a callable object, or as "
        f"a sequence of such DictConfig/callable objects; instead, we got: {type(transform)}"
    )
    out_t = []
    for t in transform:
        if isinstance(t, (dict, omegaconf.DictConfig)) and "_target_" in t:
            t = hydra.utils.instantiate(t)
        assert callable(t), f"transform object not callable: {type(t)}"
        out_t.append(t)
    if len(out_t) == 0:
        out_t = [Identity()]  # there are no transforms to apply, return an identity function
    return torchvision.transforms.Compose(out_t)
