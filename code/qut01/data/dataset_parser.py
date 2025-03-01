"""Implements a basic dataset parser interface for the "raw" deeplake dataset format."""
import datetime
import functools
import pathlib
import typing

import deeplake
import deeplake.util.pretty_print
import lightning.pytorch.core.mixins as pl_mixins
import numpy as np
import torch.utils.data.dataset

import qut01.data.annotations.keys
import qut01.data.batch_utils
import qut01.data.classif_utils
import qut01.data.statement_utils
import qut01.utils.logging

if typing.TYPE_CHECKING:
    from qut01.data.batch_utils import BatchDictType, BatchTransformType
    from qut01.data.statement_utils import StatementProcessedData

logger = qut01.utils.logging.get_logger(__name__)

default_dataset_name = "statements.20231129.deeplake"
"""Default name given to the deeplake dataset folder on local storage."""

dataset_default_branch_name = "main"
"""Name given to the base deeplake dataset branch that contains the raw data only."""

dataset_annotated_branch_name = "annotated"
"""Name given to the deeplake dataset branch that contains the criteria annotations."""

dataset_validated_branch_name = "annotated-and-validated"
"""Name given to the deeplake dataset branch that contains VALIDATED criteria annotations.

Annotations are validated by having them carefully reviewed by experts to make sure they are as
accurate as possible. Once a statement contains only validated annotations, it is included in the
gold evaluation set. Few annotations will ever be validated; see the `split_utils` module for more
information.
"""


class DataParser(torch.utils.data.dataset.Dataset, pl_mixins.HyperparametersMixin):
    """Basic dataset parser interface for the raw deeplake dataset format.

    Note that we make this class inherit from Lightning's `HyperparametersMixin` interface in order
    to save/restore constructor parameters. This allows us to build and return new data parser
    instances constructed with the original hyperparameters whenever we e.g. create a "filtered"
    version of this object.

    Args:
        dataset_path_or_object: path to the deeplake dataset to be read, or deeplake dataset
            object to be wrapped by this reader.
        dataset_branch: branch name to checkout after opening the dataset (if any). If one is
            specified, it MUST exist, or the constructor will throw.
        save_hyperparams: toggles whether hyperparameters should be saved in this class. This
            should be `False` when this class is derived, and the `save_hyperparameters` function
            should be called in the derived constructor.
        ignored_tensors: provides the list of tensor names that should be ignored (and not loaded)
            when preparing and returning statement batch data.
        pickle_dir_path: path to the pickle dump directory where validated annotations are dumped
            as backups. If `None`, we will use the default framework directory.
        add_processed_data_to_batch: specifies whether statement processed data (annotations and
            extracted sentences) should be created and returned in batch dictionaries or not.
        use_processed_data_cache: specifies whether statement processed data (annotations and
            extracted sentences) should be cached internally once created.
        sentence_source_text_tensor: name of the tensor to extract raw statement sentences from.
            Should correspond to a raw string extracted via either ABBYY or fitz.
        batch_transforms: configuration dictionary or list of transformation operations that
            will be applied to the "raw" tensor batch data read by this class. These should be
            callable objects that expect to receive a batch dictionary, and that also return
            a batch dictionary.
        batch_id_prefix: string used as a prefix in the batch identifiers generated for the
            data samples read by this parser.
        extra_deeplake_kwargs: extra parameters sent to the deeplake dataset constructor.
            Should not be used if an already-opened dataset is provided.

    TODO: add option to drop/ignore statements that do not have a full set of (valid) annotations
          across their meta group? (otherwise, sampling random sentences might land on relevant ones)
    """

    def __init__(
        self,
        dataset_path_or_object: typing.Union[typing.AnyStr, pathlib.Path, deeplake.Dataset],
        dataset_branch: typing.Optional[str] = None,
        save_hyperparams: bool = True,  # turn this off in derived classes
        ignored_tensors: typing.Optional[typing.List[str]] = None,
        pickle_dir_path: typing.Optional[pathlib.Path] = None,  # none = default framework path
        dump_found_validated_annots_as_pickles: bool = False,  # dump to pickles as backup, if needed
        load_validated_annots_from_pickles: bool = False,  # load from pickles (backups), if needed
        add_processed_data_to_batch: bool = False,
        use_processed_data_cache: bool = True,
        sentence_source_text_tensor: str = "fitz/text",
        processed_data_key: str = "processed_data",
        batch_transforms: "BatchTransformType" = None,
        batch_id_prefix: typing.Optional[str] = None,
        **extra_deeplake_kwargs,  # cannot be used with a dataset object (only with path)
    ):
        """Parses a deeplake archive or wraps an already-opened object.

        Note: we should NOT call `self.save_hyperparameters` in this class constructor if it is not
        intended to be used as the FINAL derivation before being instantiated into an object; in other
        words, if you intend on using this class as an interface, turn `save_hyperparams` OFF! See
        these links for more information:
            https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
            https://github.com/Lightning-AI/lightning/issues/16206
        """
        super().__init__()
        if save_hyperparams:
            self.save_hyperparameters(
                ignore=["dataset_path_or_object", "extra_deeplake_kwargs"],
                logger=False,
            )
        if isinstance(dataset_path_or_object, deeplake.Dataset):
            assert not extra_deeplake_kwargs, "dataset is already opened, can't use extra kwargs"
            dataset = dataset_path_or_object
        else:
            dataset = deeplake.load(
                dataset_path_or_object,
                read_only=True,
                check_integrity=True,
                **extra_deeplake_kwargs,
            )
        self.dataset = dataset
        if dataset_branch is not None:
            self.dataset.checkout(dataset_branch)
        self.statement_ids = self.dataset.statement_id.numpy().flatten().tolist()
        assert len(self.statement_ids) == len(self.dataset), "unexpected ids-to-dataset mismatch"
        self.ignored_tensors = [] if ignored_tensors is None else ignored_tensors
        self.pickle_dir_path = pickle_dir_path
        self.dump_found_validated_annots_as_pickles = dump_found_validated_annots_as_pickles
        self.load_validated_annots_from_pickles = load_validated_annots_from_pickles
        self.processed_data_key = processed_data_key
        self.batch_id_prefix = batch_id_prefix or ""
        self.batch_transforms = qut01.data.transforms.validate_or_convert_transform(batch_transforms)
        self.add_processed_data_to_batch = add_processed_data_to_batch
        self.use_processed_data_cache = use_processed_data_cache
        self._processed_data_cache = {}  # sidx to processed data obj mapping
        assert sentence_source_text_tensor in self.tensor_names
        self.sentence_source_text_tensor = sentence_source_text_tensor
        self.text_sentences_cache, self.annotation_objects_cache = None, None

    @property
    def batch_id_key(self) -> str:
        """Returns the batch dictionary key that stores the statement identifiers.

        Note: the different between statement "identifiers" (or IDs) and "indices" is that the
        identifiers are the numbers used to refer to the statements on the Modern Slavery register,
        and the numbers used to refer to the statements as part of this dataset are the "indices".
        """
        return qut01.data.batch_utils.batch_id_key

    @property
    def batch_size_key(self) -> str:
        """Returns the batch dictionary key that stores the statement count."""
        return qut01.data.batch_utils.batch_size_key

    @property
    def batch_index_key(self) -> str:
        """Returns the batch dictionary key that stores the statement's dataset-level index.

        Note: the different between statement "identifiers" (or IDs) and "indices" is that the
        identifiers are the numbers used to refer to the statements on the Modern Slavery register,
        and the numbers used to refer to the statements as part of this dataset are the "indices".
        """
        return qut01.data.batch_utils.batch_index_key

    def __len__(self) -> int:
        """Returns the total size (in terms of data batch count) of the dataset."""
        return len(self.dataset)

    def get_tensor_data(
        self,
        index: typing.Hashable,
        meta_only: bool = False,  # for metadata-only queries, i.e. without raw pdf data or text
    ) -> "BatchDictType":
        """Returns a single data tensor batch loaded from the dataset at the given index."""
        index = self._validate_or_convert_index(index)
        if meta_only:
            batch = self._get_meta_tensor_batch(index)
        else:
            batch = self._get_full_tensor_batch(index)
            if self.add_processed_data_to_batch:
                processed_data = self.get_processed_data(
                    index=index,
                    statement_tensor_data=batch,
                )
                batch[self.processed_data_key] = processed_data
        if self.batch_transforms:
            batch = self.batch_transforms(batch)
        assert isinstance(batch, typing.Dict), f"unexpected post-transform batch type: {type(batch)}"
        if self.batch_index_key not in batch:
            batch[self.batch_index_key] = index
        if self.batch_size_key not in batch:
            batch[self.batch_size_key] = self._get_batch_size_from_index(batch, index)
        if self.batch_id_key not in batch:
            batch[self.batch_id_key] = self._get_batch_id_for_index(batch, index)
        return batch

    def get_processed_data(
        self,
        index: typing.Hashable,
        statement_tensor_data: typing.Optional["BatchDictType"] = None,
    ) -> "StatementProcessedData":
        """Returns a processed data object from the tensor data loaded at a given index."""
        index = self._validate_or_convert_index(index)
        if self.use_processed_data_cache and index in self._processed_data_cache:
            statement_data = self._processed_data_cache[index]
        else:
            if statement_tensor_data is None:
                statement_tensor_data = self._get_full_tensor_batch(index)
            statement_data = qut01.data.statement_utils.StatementProcessedData.create(
                statement_tensor_data=statement_tensor_data,
                source_text_tensor=self.sentence_source_text_tensor,
                pickle_dir_path=self.pickle_dir_path,
                dump_found_validated_annots_as_pickles=self.dump_found_validated_annots_as_pickles,
                load_validated_annots_from_pickles=self.load_validated_annots_from_pickles,
            )
            if self.use_processed_data_cache:
                self._processed_data_cache[index] = statement_data
        return statement_data

    def __getitem__(self, index: typing.Hashable, meta_only: bool = False) -> "BatchDictType":
        """Returns a single data batch loaded from the dataset at the given index."""
        batch = self.get_tensor_data(index=index, meta_only=meta_only)
        return batch

    def _validate_or_convert_index(self, index: typing.Hashable) -> typing.Hashable:
        """Validates or converts (if needed) the data batch index used to fetch a data batch."""
        # by default, this impl does NOT support slicing, and we assume all indices are ints
        if np.issubdtype(type(index), np.integer):
            # convert numpy ints here if needed, as some datasets might not be familiar w/ those
            index = int(index)  # noqa
        assert isinstance(index, int), f"unsupported index type for base parser: {type(index)}"
        assert 0 <= index < len(self), f"invalid data batch index being queried: {index}"
        return index

    def _get_full_tensor_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch loaded from the deeplake dataset at a specified index."""
        # the following line will fetch the corresponding batch data for the given index across
        # all deeplake tensors stored in the dataset (assuming they're all indexed identically)
        data = self.dataset[index]  # noqa
        # we will now convert all these tensors to numpy arrays, which might not be adequate for
        # funky datasets composed of non-stackable arrays (e.g. when using images with different
        # shapes, when loading json data, when decoding raw bytes, etc.) ... in those cases, you
        # should derive and implement your own version of this function to do your own unpacking!
        batch = {
            tensor_name: data[tensor_name].numpy()
            for tensor_name in self.tensor_names
            if tensor_name not in self.ignored_tensors
        }
        # as a bonus, we provide the index used to fetch the batch with the default key
        batch[self.batch_index_key] = index
        # finally, fill in the int-based statement id and annotation counts for this batch
        batch["potential_annotation_count"] = self.get_potential_annotation_counts(batch)
        batch["statement_id"] = self.statement_ids[index]
        return batch

    def _get_meta_tensor_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch containing only metadata and annotations.

        In contrast with the `_get_full_tensor_batch` function, this function should be able to provide a
        much faster access to metadata and annotations if only those are required for analysis.
        """
        target_tensor_prefixes = ["metadata", "annotations"]
        target_tensors = [t for t in self.tensor_names if any([t.startswith(p) for p in target_tensor_prefixes])]
        batch = {tensor_name: self.dataset[tensor_name][index].numpy() for tensor_name in target_tensors}
        batch[self.batch_index_key] = index
        batch["potential_annotation_count"] = self.get_potential_annotation_counts(batch)
        batch["statement_id"] = self.statement_ids[index]
        return batch

    def _get_batch_size_from_index(  # noqa
        self,
        batch: "BatchDictType",
        index: typing.Hashable,
    ) -> int:
        """Returns the expected batch size for the given batch + (validated, converted) index."""
        if self.batch_size_key in batch:
            return batch[self.batch_size_key]
        assert isinstance(index, int) and 0 <= index < len(self.dataset)
        return 1

    def _get_batch_id_for_index(
        self,
        batch: "BatchDictType",
        index: typing.Hashable,
    ) -> typing.Hashable:
        """Returns the unique batch identifier for the batch + (validated, converted) index."""
        if self.batch_id_key in batch:
            return batch[self.batch_id_key]
        # since this parses modern slavery statements with unique identifiers from the AMSR, we'll
        # return those ids (with a potential prefix) as references
        assert isinstance(index, int) and 0 <= index < len(self.dataset)
        if not self.batch_id_prefix or self.batch_id_prefix.endswith("_"):
            return f"{self.batch_id_prefix}statement_{self.statement_ids[index]}"
        else:
            return f"{self.batch_id_prefix}_statement_{self.statement_ids[index]}"

    def get_potential_annotation_counts(
        self,
        batch: "BatchDictType",
    ) -> typing.Dict[str, int]:
        """Returns the number of annotations potentially available for a single statement.

        Note: the "potential" part means that we have not confirmed that the annotations exist and
        are not badly formatted; they might never be created if there is an issue with them.
        """
        annot_counts = {}
        for class_name in qut01.data.classif_utils.ANNOT_META_CLASS_NAMES:
            annot_flag_to_check = qut01.data.annotations.keys.get_potentially_annotated_flag_key(class_name)
            annot_counts[class_name] = 0
            if annot_flag_to_check not in batch:
                continue  # can't even confirm that the data is annotated for this group, so skip
            annotated_flag = batch[annot_flag_to_check]
            if isinstance(annotated_flag, deeplake.Tensor):
                annotated_flag = annotated_flag.numpy()
            if isinstance(annotated_flag, np.ndarray):
                assert annotated_flag.size == 1
                annotated_flag = annotated_flag.item()
            assert isinstance(annotated_flag, bool)
            if not annotated_flag:
                continue  # this data is not annotated at all for this annotation (meta-)class
            # this data is annotated, just need to look at the dataset info to see how many times
            annot_counts[class_name] = self.dataset.info["annotation_count"][class_name]
        return annot_counts

    def get_validated_annotation_flags(
        self,
        batch: "BatchDictType",
    ) -> typing.Dict[str, int]:
        """Returns whether annotations are validated or not for a single statement.

        Being "validated" means that the annotation was reviewed and updated at least once by an
        expert. We determine this based on the `last_update` flag being set to anything but an
        empty string for a particular annotation.
        """
        valid_flags = {}
        for class_name in qut01.data.classif_utils.ANNOT_CLASS_NAMES:
            last_update_key = qut01.data.annotations.keys.get_annotation_last_update_key(class_name)
            valid_flags[class_name] = False
            if last_update_key not in batch:
                continue
            last_update = batch[last_update_key]
            if isinstance(last_update, deeplake.Tensor):
                last_update = last_update.numpy()
            if isinstance(last_update, np.ndarray):
                assert last_update.size == 1
                last_update = last_update.item()
            assert isinstance(last_update, str)
            if last_update == "":
                continue
            assert datetime.datetime.fromisoformat(last_update)
            valid_flags[class_name] = True
        return valid_flags

    def get_potentially_annotated_statement_ids(self) -> typing.Dict[str, typing.List[int]]:
        """Returns the identifiers (ints) of all potentially annotated statements.

        This function will throw an exception if the dataset is not checked out on a branch with
        annotations.

        Note: the "potential" part means that we have not confirmed that the annotations exist and
        are not badly formatted; they might never be created if there is an issue with them.
        """
        annot_statement_ids = {}
        for class_name in qut01.data.classif_utils.ANNOT_META_CLASS_NAMES:
            annot_flag_to_check = qut01.data.annotations.keys.get_potentially_annotated_flag_key(class_name)
            assert annot_flag_to_check in self.dataset.tensors, (
                f"missing mandatory tensor '{annot_flag_to_check}' when checking for potential annotations;"
                "\b\t(make sure your dataset is checked out on a branch that contains annotations!)"
            )
            valid_mask = self.dataset[annot_flag_to_check].numpy().flatten()
            assert len(valid_mask) == len(self.statement_ids)
            annot_statement_ids[class_name] = np.asarray(self.statement_ids)[valid_mask].tolist()
        return annot_statement_ids

    def get_validated_statement_ids(self) -> typing.Dict[str, typing.List[int]]:
        """Returns the identifiers (ints) of all statements with validated annotations.

        This function will throw an exception if the dataset is not checked out on a branch with
        validated annotations, or if the parser is configured to load pickles at runtime if any are
        found.
        """
        assert not self.load_validated_annots_from_pickles, (
            "cannot ask for list of validated statement IDs while also trying to load them from pickles\n"
            "\t(this implementation does NOT look through potential pickle files, so some might be missing)"
        )
        valid_statement_ids = {}
        for class_name in qut01.data.classif_utils.ANNOT_CLASS_NAMES:
            last_update_key = qut01.data.annotations.keys.get_annotation_last_update_key(class_name)
            assert last_update_key in self.dataset.tensors, (
                f"missing mandatory tensor '{last_update_key}' when checking for validated annotations;"
                "\b\t(make sure your dataset is checked out on a branch that contains validation flags!)"
            )
            valid_mask = [last_update != "" for last_update in self.dataset[last_update_key].numpy().flatten()]
            assert len(valid_mask) == len(self.statement_ids)
            valid_statement_ids[class_name] = np.asarray(self.statement_ids)[valid_mask].tolist()
        return valid_statement_ids

    def get_fully_validated_statement_ids(self) -> typing.List[int]:
        """Returns the identifiers (ints) of all statements with fully validated annotations.

        This function will throw an exception if the dataset is not checked out on a branch with
        validated annotations.
        """
        valid_sids_per_type = self.get_validated_statement_ids()
        fully_valid_sids = functools.reduce(set.intersection, [set(sids) for sids in valid_sids_per_type.values()])
        return list(fully_valid_sids)

    def update_tensors(
        self,
        statement_id: int,
        tensor_data: typing.Dict[str, typing.Any],
    ) -> None:
        """Updates tensors associated with a particular statement in the dataset.

        The statement is expected to be given via its IDENTIFIER, not its index!

        Note: this may cause unexpected behavior when updating tensors with caching enabled. An
        exception will also be thrown if the dataset is not opened in writing mode. Finally, note
        that if more than one annotation exists for the target statement, all of them will be
        removed and replaced by the provided data.
        """
        assert not self.dataset.read_only, "cannot update dataset content in read-only mode!"
        assert (
            self.dataset.branch == qut01.data.dataset_parser.dataset_validated_branch_name
        ), "we should only try to update the dataset as part of the annotation validation process?"
        assert statement_id in self.statement_ids, f"unknown statement id: {statement_id}"
        self.dataset.info["updated_on"] = datetime.datetime.now().isoformat()
        self.dataset.info["repo_version"] = qut01.__version__
        statement_idx = self.statement_ids.index(statement_id)
        for tensor_dataset_name, tensor_val in tensor_data.items():
            assert isinstance(tensor_dataset_name, str), f"invalid dataset name: {tensor_dataset_name}"
            assert tensor_dataset_name in self.dataset, f"unknown tensor dataset name: {tensor_dataset_name}"
            tensor_dataset = self.dataset[tensor_dataset_name]
            assert len(tensor_dataset) == len(self.statement_ids)
            if tensor_dataset.htype.startswith("sequence"):
                # multi-annotator-tensor-dataset: we must empty the existing values and replace them all
                if not isinstance(tensor_val, list):
                    # note: since deeplake is a bit dumb and won't replace the stored list with another one,
                    #       we need to replace all elements of the original list by copies of the new value
                    #       (we assume that the extra copies are discarded when the validated data is loaded)
                    orig_elem_count = len(tensor_dataset[statement_idx].numpy())
                    tensor_val = [tensor_val] * orig_elem_count
                tensor_dataset[statement_idx] = tensor_val
            else:
                # single-annotator-tensor-dataset: we just replace the value directly
                tensor_dataset[statement_idx] = tensor_val

    @property
    def tensor_info(self) -> typing.Dict[str, typing.Any]:
        """Returns the dictionary of tensor info objects (deeplake-defined) from the dataset.

        The returned objects can help downstream processing stages figure out what kind of data they
        will be receiving from this parser.
        """
        return {k: v.info for k, v in self.dataset.tensors.items()}

    @property
    def tensor_names(self) -> typing.List[str]:
        """Names of the data tensors that will be provided in the loaded batches.

        Note that additional tensors and other attributes may be loaded as well, but these are the
        'primary' fields that should be expected by downstream processing stages.
        """
        return list(self.dataset.tensors.keys())

    @property
    def info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information parsed from the deeplake object."""
        return dict(self.dataset.info)

    @property
    def dataset_name(self) -> str:
        """Returns the dataset commit id used to identify this particular dataset."""
        return str(self.dataset.commit_id)

    def summary(self) -> None:
        """Prints a summary of the deeplake dataset using the default logger.

        Note: this might take a while (minutes) with huge datasets!
        """
        logger.info(f"dataset object: {str(self.dataset)}")
        logger.info(f"dataset info: {self.dataset.info}")
        logger.info(f"dataset length: {len(self)}")
        logger.info(f"dataset branches: {self.dataset.branches}")
        logger.info(f"dataset current branch: {self.dataset.branch}")
        logger.info(f"dataset current commit: {self.dataset.commit_id}")
        logger.info(deeplake.util.pretty_print.summary_dataset(self.dataset))

    def filter(
        self,
        function: typing.Union[typing.Callable, str],
        **filter_kwargs,
    ):
        """Filters the dataset in accordance of filter function `f(x: sample) -> bool`.

        See `deeplake.core.dataset.Dataset.filter` for more information.
        """
        if self.dataset.min_len != self.dataset.max_len:
            raise NotImplementedError("cannot filter variable length datasets")
        filtered_dataset = self.dataset.filter(function=function, **filter_kwargs)
        return self.__class__(dataset_path_or_object=filtered_dataset, **self.hparams)

    def get_dataloader(
        self,
        # arguments for dataloader wrapper: deeplake.enterprise.dataloader.DeepLakeDataLoader.batch
        batch_size: int = 1,
        drop_last: bool = False,
        # arguments for dataloader wrapper: deeplake.enterprise.dataloader.DeepLakeDataLoader.shuffle
        shuffle: bool = False,
        shuffle_buffer_size: int = 2048,
        # arguments for dataloader wrapper: deeplake.enterprise.dataloader.DeepLakeDataLoader.pytorch
        num_workers: int = 0,
        collate_fn: typing.Optional[typing.Callable] = None,
        tensors: typing.Optional[typing.List[str]] = None,
        num_threads: typing.Optional[int] = None,
        prefetch_factor: int = 2,
        distributed: bool = False,
        return_index: bool = True,
        decode_method: typing.Optional[typing.Dict[str, str]] = None,
        persistent_workers: bool = False,
        # arguments for dataloader construction: deeplake.enterprise.dataloader.dataloader
        ignore_errors: bool = False,
        verbose: bool = False,
        # arguments to toggle the dataloader type
        use_optimized_dataloader: bool = False,
    ) -> torch.utils.data.DataLoader:
        """Returns a deeplake data loader for this data parser object.

        Derived classes may implement/use more complex collate, wrapper, or transform functions. By
        default, we simply forward the default settings to deeplake's dataloader creator.
        """
        if collate_fn is None:
            from qut01.data.transforms.collate import default_collate

            collate_fn = default_collate
        tensors = tensors if tensors is not None else self.tensor_names
        tensors = [t for t in tensors if t not in self.ignored_tensors]
        if use_optimized_dataloader:
            dataloader = self.dataset.dataloader(ignore_errors=ignore_errors, verbose=verbose).batch(
                batch_size=batch_size,
                drop_last=drop_last,
            )
            if self.batch_transforms:
                dataloader = dataloader.transform(self.batch_transforms)
            if shuffle:
                dataloader = dataloader.shuffle(shuffle=shuffle, buffer_size=shuffle_buffer_size)
            dataloader = dataloader.pytorch(
                num_workers=num_workers,
                collate_fn=collate_fn,
                tensors=tensors,
                num_threads=num_threads,
                prefetch_factor=prefetch_factor,
                distributed=distributed,
                return_index=return_index,
                decode_method=decode_method,
                persistent_workers=persistent_workers,
            )
        else:
            assert not distributed, "missing distributed implementation with fallback dataloader"
            dataloader = deeplake.integrations.pytorch.pytorch.dataset_to_pytorch(
                self.dataset,
                num_workers=num_workers,
                batch_size=batch_size,
                drop_last=drop_last,
                collate_fn=collate_fn,
                pin_memory=False,
                shuffle=shuffle,
                buffer_size=shuffle_buffer_size,
                use_local_cache=False,
                transform=self.batch_transforms,
                tensors=tensors,
                return_index=return_index,
                pad_tensors=False,
                decode_method=decode_method,
                persistent_workers=persistent_workers,
                # cache_size=?
            )
        return dataloader


def get_dataloader(
    parser: DataParser,
    **deeplake_pytorch_dataloader_kwargs,
) -> torch.utils.data.DataLoader:
    """Returns a deeplake data loader for the given data parser object.

    This will call the `get_dataloader` function from the parser class itself, which may be derived
    from the base class for some datasets.
    """
    assert isinstance(
        parser, DataParser
    ), f"invalid data parser type to use the deeplake dataloader getter: {type(parser)}"
    return parser.get_dataloader(**deeplake_pytorch_dataloader_kwargs)


def get_default_deeplake_dataset_path() -> pathlib:
    """Returns the framework's default path where we should find the deeplake dataset."""
    return qut01.utils.config.get_data_root_dir() / default_dataset_name


def get_deeplake_dataset(
    dataset_path: typing.Optional[pathlib.Path] = None,  # none = fall back to framework default
    checkout_branch: typing.Optional[str] = None,
    read_only: bool = True,
    **deeplake_kwargs,
) -> deeplake.Dataset:
    """Opens and returns a deeplake dataset object while printing basic information about it."""
    if dataset_path is None:
        dataset_path = get_default_deeplake_dataset_path()
    assert dataset_path.exists(), f"could not locate deeplake dataset at: {dataset_path}"
    logger.info(f"parsing dataset from: {dataset_path.absolute()}")
    dataset = deeplake.load(dataset_path, check_integrity=True, read_only=read_only, **deeplake_kwargs)
    if checkout_branch and dataset.branch != checkout_branch:
        dataset.checkout(checkout_branch)
    dataset.log()
    logger.info(dataset.metadata)
    logger.info(f"dataset size: {len(dataset)}")
    logger.info("dataset info:")
    for info_key, info_val in dataset.info.items():
        logger.info(f"\t{info_key}: {info_val}")
    logger.info("dataset tensors:")
    for tensor_key, tensor_val in dataset.tensors.items():
        logger.info(f"\t{tensor_key}: {tensor_val.htype}  {tensor_val.shape}")
    logger.info("dataset branches:")
    current_branch_commits = len(dataset.commits)
    current_branch_str = f"(current, {current_branch_commits} commits)"
    for branch in dataset.branches:
        logger.info(f"\t'{branch}'  {current_branch_str if dataset.branch == branch else ''}")
    return dataset


def prepare_dataset_for_validation(
    dataset: deeplake.Dataset,
    restart_from_raw_annotations: bool,
    bypass_user_confirmation: bool,
) -> None:
    """Prepares the deeplake dataset for validation (with the proper branch checkout + init)."""
    assert not dataset.read_only, "dataset is read-only, but we must write to it during validation"
    if restart_from_raw_annotations or dataset_validated_branch_name not in dataset.branches:
        if dataset_validated_branch_name in dataset.branches:
            if bypass_user_confirmation:
                confirmed = True
            else:
                choice = input(f"\nOverwrite {dataset_validated_branch_name} branch? (y/N):\n")
                confirmed = choice.strip().lower() in ["y", "yes"]
            if not confirmed:
                print("Branch overwrite aborted, exiting app!")
                exit(-1)
            dataset.delete_branch(dataset_validated_branch_name)
        dataset.checkout(dataset_validated_branch_name, create=True)
        # we create a new tensor that will hold the validation datetime, per criteria, per statement
        for annot_name in qut01.data.classif_utils.ANNOT_CLASS_NAMES:
            annot_last_update_key = qut01.data.annotations.keys.get_annotation_last_update_key(annot_name)
            tensor = dataset.create_tensor(name=annot_last_update_key, htype="text")
            tensor.extend([""] * len(dataset.statement_id))  # no datetime = 'not validated yet'
    else:
        dataset.checkout(dataset_validated_branch_name)


if __name__ == "__main__":
    import qut01.utils.logging

    qut01.utils.logging.setup_logging_for_analysis_script()
    dataset_ = get_deeplake_dataset()
    parser_ = DataParser(dataset_)
    parser_.summary()
