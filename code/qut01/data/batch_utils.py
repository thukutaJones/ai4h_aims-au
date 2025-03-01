import typing

import numpy as np
import torch

BatchDictType = typing.Dict[str, typing.Any]
"""Default type used to represent a data batch loaded by a data parser/loader and fed to a model."""

_BatchTransformType = typing.Callable[[BatchDictType], BatchDictType]

BatchTransformType = typing.Union[_BatchTransformType, typing.Sequence[_BatchTransformType], None]
"""Default type used to represent a callable object or function that transforms a data batch."""

batch_size_key: str = "batch_size"
"""Default batch dictionary key (string) used to store/fetch the batch size."""

batch_id_key: str = "batch_id"
"""Default batch dictionary key (string) used to store/fetch the batch identifier."""

batch_index_key: str = "index"
"""Default batch dictionary key (string) used to store/fetch the batch index."""


def get_batch_size(batch: "BatchDictType") -> int:
    """Checks the provided batch dictionary and attempts to return the batch size.

    If the given dictionary does not contain a `batch_size` attribute that we can interpret, we will
    throw an exception. Otherwise, if that attribute is an integer or a tensor/array, the resulting
    batch size will be returned.

    If the batch size is stored as an array, we will assume that it is as the result of collating
    the loaded batches of multiple parsers/dataloaders; we will therefore sum all values in that
    array (where each value should be the batch size of a single collated chunk) in order to return
    the total batch size.
    """
    if batch is None or not batch:
        return 0
    assert batch_size_key in batch, "could not find the mandatory 'batch_size' key in the given batch dictionary!"
    batch_size = batch[batch_size_key]
    # we'll try to interpret this potential object in any way we can...
    if isinstance(batch_size, int):
        pass  # nothing to do, it's good as-is!
    elif np.issubdtype(type(batch_size), np.integer):
        batch_size = int(batch_size)  # in case we're using numpy ints (might break slicing)
    elif isinstance(batch_size, np.ndarray):
        assert np.issubdtype(batch_size.dtype, np.integer), f"invalid batch size array type: {batch_size.dtype}"
        batch_size = int(batch_size.astype(np.int64).sum())
    elif isinstance(batch_size, torch.Tensor):
        batch_size = batch_size.long().sum().item()
    else:
        raise NotImplementedError(f"cannot handle batch size type: {type(batch_size)}")
    assert batch_size >= 0, f"found an invalid batch size! ({batch_size})"
    return batch_size


def get_batch_id(
    batch: typing.Optional["BatchDictType"] = None,
    batch_id_prefix: typing.Optional[str] = None,
    batch_index_key_: typing.Optional[str] = None,
    dataset_name: typing.Optional[str] = None,
    index: typing.Optional[typing.Hashable] = None,
) -> typing.Hashable:
    """Checks the provided batch dictionary and attempts to return its batch identifier.

    If the given dictionary does not contain a 'batch_id' attribute that we can return, we will
    create such an id with the provided prefix/dataset/index info. If no info is available, we will
    throw an exception.

    Args:
        batch: the batch dictionary from which we'll return the batch identifier (if it's already
            there), or from which we'll gather data in order to build a batch identifier.
        batch_id_prefix: a prefix used when building batch identifiers. Will be ignored if a batch
            identifier is already present in the `batch`.
        batch_index_key_: an attribute name (key) under which we should be able to find the "index"
            of the provided batch dictionary. Will be ignored if a batch identifier is already
            present in the `batch`. If necessary yet `None`, we will at least try the default
            `batch_index_key` value before throwing an exception.
        dataset_name: an extra name to add when building batch identifiers. Will be ignored if a
            batch identifier is already present in the `batch`.
        index: the hashable index that corresponds to the integer or unique ID used to fetch the
            targeted batch dictionary from a dataset parser. Constitutes the basis for the creation
            of a batch identifier. If not specified, the `batch_index_key` must be provided so that
            we can find the actual index from a field within the batch dictionary.  Will be ignored
            if a batch identifier is already present in the `batch`.

    Returns:
        The (hopefully) unique batch identifier used to reference this batch elsewhere.
    """
    if batch is None or not batch or batch_id_key not in batch:
        if index is None:
            if batch_index_key_ is None:
                batch_index_key_ = batch_index_key  # fallback to module-wide default
            assert isinstance(batch, typing.Dict) and batch_index_key_ in batch, (
                "batch dict did not contain a batch identifier, and we need at least an index to "
                f"build such an identifier!\n (provide it via the `{batch_index_key_}` dict key in"
                f"the data parser, or implement your own transform to add it)"
            )
            index = batch[batch_index_key_]
        if isinstance(index, np.ndarray):
            assert index.ndim == 1 and index.size > 0, "index should be non-empty 1d vector"
            index = tuple(index)
            if len(index) == 1:
                index = index[0]
        assert isinstance(index, typing.Hashable), f"bad index for batch identifier: {type(index)}"
        prefix = f"{batch_id_prefix}_" if batch_id_prefix else ""
        dataset = f"{dataset_name}_" if dataset_name else ""
        if isinstance(index, int) or np.issubdtype(type(index), np.integer):
            index = f"batch{index:08d}"
        return f"{prefix}{dataset}{index}"
    else:
        batch_id = batch[batch_id_key]
        assert isinstance(batch_id, typing.Hashable), f"found batch id has bad type: {type(batch_id)}"
    return batch_id


def get_batch_index(
    batch: "BatchDictType",
    batch_index_key_: typing.Optional[str] = None,
) -> typing.Hashable:
    """Checks the provided batch dictionary and attempts to return the batch index.

    If the given dictionary does not contain a `batch_index` attribute that we can interpret, we
    will throw an exception.


    Args:
        batch: the batch dictionary from which we'll return the batch index.
        batch_index_key_: an attribute name (key) under which we should be able to find the "index"
            of the provided batch dictionary. If `None`, will default to the module-defined value.
    """
    assert isinstance(batch, typing.Dict), f"invalid batch type: {type(batch)}"
    if batch_index_key_ is None:
        batch_index_key_ = batch_index_key
    assert batch_index_key_ in batch, f"batch dict does not contain key: {batch_index_key_}"
    batch_index = batch[batch_index_key_]
    if isinstance(batch_index, np.ndarray):
        assert batch_index.ndim == 1 and batch_index.size > 0, "index should be non-empty 1d vector"
        batch_index = tuple(batch_index)
        if len(batch_index) == 1:
            batch_index = batch_index[0]
    assert isinstance(batch_index, typing.Hashable), f"invalid batch index type: {type(batch_index)}"
    return batch_index
