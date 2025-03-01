import copy
import math
import typing

import numpy as np
import torch.utils.data


class IterableDataset(torch.utils.data.IterableDataset):
    """Wraps a dataset so that items provided in varying-size arrays are returned one at a time.

    If shuffling is not activated, the items will be returned in the order defined by the arrays of
    the wrapped dataset as if those arrays were concatenated. Otherwise, an item buffer will be
    filled up to a specified size, shuffled, and slowly emptied repeatedly until all items have
    been returned.

    Note: only ONE data loader should use this object at a time because of its internal buffer and
    because of the state variables required to iterate over the wrapped dataset.

    Args:
        dataset_to_wrap: dataset to wrap for array item retrieval.
        target_array_keys: list of keys for target arrays to retrieve items from.
        constant_copy_keys: list of keys that contain other values/tensors to copy for each item.
        shuffle: specifies whether to shuffle the items inside the internal buffer.
        buffer_size: size of the item buffer to use. The buffer will be filled with up to that many
            items loaded from wrapped dataset arrays, shuffled (if needed), and them emptied one
            item at a time. Once the size is lower or equal to `buffer_refill_threshold`, the
            buffer is refilled. The process is repeated until all items have been returned.
        buffer_refill_threshold: the threshold for the number of items in the buffer under which it
            will be refilled. Defaults to zero, meaning the buffer will always be fully emptied
            before being refilled.
        shuffle_seed: random seed used to initialize the RNG used for shuffling.
    """

    def __init__(
        self,
        dataset_to_wrap: torch.utils.data.Dataset,
        target_array_keys: typing.List[str],
        constant_copy_keys: typing.Optional[typing.List[str]] = None,
        shuffle: bool = False,
        buffer_size: int = 0,  # in number of dataset array items
        buffer_refill_threshold: int = 0,  # in number of dataset array items
        shuffle_seed: typing.Optional[int] = None,
    ) -> None:
        """Initializes the dataset wrapper while validating settings."""
        assert hasattr(dataset_to_wrap, "__getitem__") and hasattr(
            dataset_to_wrap, "__len__"
        ), f"invalid dataset type: {type(dataset_to_wrap)}"
        self.dataset = dataset_to_wrap
        assert len(target_array_keys) > 0, "need to specify at least one target array key!"
        self.target_array_keys = target_array_keys
        if constant_copy_keys is None:
            constant_copy_keys = []
        self.constant_copy_keys = constant_copy_keys
        assert len(set(self.target_array_keys) & set(self.constant_copy_keys)) == 0
        self.shuffle = shuffle
        assert buffer_size >= 0, f"invalid buffer size: {buffer_size}"
        self.buffer_size = buffer_size
        assert buffer_refill_threshold >= 0, f"invalid buffer refill threshold: {buffer_refill_threshold}"
        self.buffer_refill_threshold = buffer_refill_threshold
        self.shuffle_seed = shuffle_seed
        self._reset_iterator()

    def _reset_iterator(self) -> None:
        """Resets the iterator internal state + buffer (if one is used)."""
        dataset_size = len(self.dataset)  # noqa
        self._buffer = []
        # TODO: also update this to work with distributed setup (i.e. with world size > 1)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            self._remaining_idxs = list(range(dataset_size))
        else:  # in a worker process, share the workload by splitting indices across all workers
            idx_count_per_worker = int(math.ceil(dataset_size / float(worker_info.num_workers)))
            idx_start = worker_info.id * idx_count_per_worker
            idx_end = min(idx_start + idx_count_per_worker, dataset_size)
            self._remaining_idxs = list(range(idx_start, idx_end))
        if self.shuffle:
            self._shuffle_rng = np.random.default_rng(self.shuffle_seed)
            self._shuffle_rng.shuffle(self._remaining_idxs)

    def __iter__(self) -> "IterableDataset":
        """Get an iterator for the items in the target arrays of the wrapper dataset."""
        self._reset_iterator()
        return self

    def __next__(self) -> typing.Dict[str, typing.Any]:
        """Get the next item from the next array of the wrapped dataset."""
        if len(self._buffer) <= self.buffer_refill_threshold and self._remaining_idxs:  # refill time
            while self._remaining_idxs:  # there are still items to fetch from the dataset
                curr_idx = self._remaining_idxs.pop(0)
                curr_items = list(self._yield_items_for_idx(curr_idx))
                if not curr_items:
                    continue
                self._buffer.extend(curr_items)
                if len(self._buffer) >= self.buffer_size:
                    break
            if self.shuffle:
                self._shuffle_rng.shuffle(self._buffer)  # refill done, shuffle the buffer
        if not self._buffer and not self._remaining_idxs:
            raise StopIteration  # we're at the total end, no more refills
        # buffer is ready, return the next item
        return self._buffer.pop(0)

    def _yield_items_for_idx(
        self,
        index: int,
    ) -> typing.Generator[typing.Dict[str, typing.Any], None, None]:
        """Yields all items for the specified dataset index while deep copying constants."""
        assert index <= len(self.dataset), f"invalid index: {index}"  # noqa
        batch = self.dataset[index]
        assert isinstance(batch, dict), f"unexpected batch type: {type(batch)}"
        array_data, constant_data = {}, {}
        for key in self.constant_copy_keys:
            assert key in batch, f"missing constant value '{key}' from batch"
            constant_data[key] = batch[key]
        array_length = None
        for key in self.target_array_keys:
            assert key in batch, f"missing target array '{key}' from batch"
            if array_length is None:
                array_length = len(batch[key])
            else:
                assert len(batch[key]) == array_length, "mismatched array length(s)"
            array_data[key] = batch[key]
        for item_idx in range(array_length):
            output = {
                **{key: copy.deepcopy(val) for key, val in constant_data.items()},
                **{key: array[item_idx] for key, array in array_data.items()},
            }
            yield output
