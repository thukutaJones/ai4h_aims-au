"""Implements a data module for the MNIST train/valid/test loaders.

See the following URL for more info on this dataset:
https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html

(this should only be used for testing/debugging purposes)
"""
import pathlib
import typing

import hydra.utils
import lightning.pytorch.core.mixins as pl_mixins
import torch
import torch.utils.data
import torchvision

import qut01.data.datamodules.base
import qut01.data.transforms
import qut01.utils.logging

if typing.TYPE_CHECKING:
    import qut01

logger = qut01.utils.logging.get_logger(__name__)


class DataModule(qut01.data.datamodules.base.BaseDataModule):
    """Example of LightningDataModule for the MNIST dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This Lightning interface allows you to share a full dataset without explaining how to download,
    split, transform, and process the data. More info here:
        https://lightning.ai/docs/pytorch/stable/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: typing.Union[typing.AnyStr, pathlib.Path],
        dataparser_configs: typing.Optional[qut01.utils.DictConfig] = None,
        dataloader_configs: typing.Optional[qut01.utils.DictConfig] = None,
        train_val_split: typing.Tuple[int, int] = (55_000, 5_000),
    ):
        """Initializes the MNIST data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.

        Args:
            data_dir: directory where the MNIST dataset is located (or where it will be downloaded).
            dataparser_configs: configuration dictionary of data parser settings, separated by
                data subset type, which may contain for example definitions for data transforms.
            dataloader_configs: configuration dictionary of data loader settings, separated by data
                subset type, which may contain for example batch sizes and worker counts.
            train_val_split: sample split counts to use when separating the train/valid data.
        """
        self.save_hyperparameters(logger=False)
        super().__init__(dataparser_configs=dataparser_configs, dataloader_configs=dataloader_configs)
        assert data_dir is not None, "invalid data dir (must be specified, will download if needed)"
        pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
        assert len(train_val_split) == 2 and sum(train_val_split) == 60_000
        self._internal_data_transforms = [  # we'll apply these to all tensors from the orig parser
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ]
        self.data_train: typing.Optional[torch.utils.data.Dataset] = None
        self.data_valid: typing.Optional[torch.utils.data.Dataset] = None
        self.data_test: typing.Optional[torch.utils.data.Dataset] = None

    @property
    def _base_dataparser_configs(self) -> qut01.utils.DictConfig:
        """Returns the 'base' (class-specific-default) configs for the data parsers."""
        return {
            "_default_": {  # all data parsers will wrap the torchvision mnist dataset parser
                "_target_": "qut01.data.datamodules.mnist.DataParser",
                "batch_transforms": [],
            },
            # bonus: we will also give all parsers have a nice prefix
            "train": {"batch_id_prefix": "train"},
            "valid": {"batch_id_prefix": "valid"},
            "test": {"batch_id_prefix": "test"},
        }

    @property
    def num_classes(self) -> int:
        """Returns the number of labels/classes in the dataset."""
        return 10

    def prepare_data(self) -> None:
        """Downloads the MNIST data to the dataset directory, if needed.

        This method is called only from a single device, so the data will only be downloaded once.
        """
        torchvision.datasets.MNIST(self.hparams.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.hparams.data_dir, train=False, download=True)

    def setup(self, stage: typing.Optional[str] = None) -> None:
        """Loads the MNIST data under the train/valid/test parsers.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`, so be
        careful not to execute the random split twice! The `stage` can be used to differentiate
        whether it's called before `trainer.fit()` or `trainer.test()`.
        """
        # load datasets only if they're not loaded already
        if self.data_train is None:
            orig_train_parser = torchvision.datasets.MNIST(
                root=self.hparams.data_dir,
                train=True,
                transform=torchvision.transforms.Compose(self._internal_data_transforms),
            )
            # NOTE: we regenerate a split here using the original mnist train set for train+valid
            train_parser, valid_parser = torch.utils.data.random_split(
                dataset=orig_train_parser,
                lengths=self.hparams.train_val_split,
                generator=torch.Generator().manual_seed(self.split_seed),
            )
            train_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "train")
            self.data_train = hydra.utils.instantiate(train_parser_config, train_parser)
            valid_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "valid")
            self.data_valid = hydra.utils.instantiate(valid_parser_config, valid_parser)
            orig_test_parser = torchvision.datasets.MNIST(
                root=self.hparams.data_dir,
                train=False,
                transform=torchvision.transforms.Compose(self._internal_data_transforms),
            )
            test_parser_config = self._get_subconfig_for_subset(self.dataparser_configs, "test")
            self.data_test = hydra.utils.instantiate(test_parser_config, orig_test_parser)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the MNIST training set data loader."""
        assert self.data_train is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_train, subset_type="train")

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the MNIST validation set data loader."""
        assert self.data_valid is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_valid, subset_type="valid")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the MNIST testing set data loader."""
        assert self.data_test is not None, "parser unavailable, call `setup()` first!"
        return self._create_dataloader(self.data_test, subset_type="test")


class DataParser(torch.utils.data.dataset.Dataset, pl_mixins.HyperparametersMixin):
    """Base interface used to wrap the MNIST data parser."""

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        batch_transforms: "qut01.data.BatchTransformType" = None,
        batch_id_prefix: typing.Optional[str] = None,
    ):
        """Parses the MNIST dataset (or a subset of it, provided by the datamodule)."""
        super().__init__()
        self.save_hyperparameters(ignore="dataset", logger=False)
        self._dataset_name = "MNIST"
        self._tensor_names: typing.List[str] = ["data", "target"]
        self._batch_id_prefix = batch_id_prefix or ""
        self.batch_transforms = qut01.data.transforms.validate_or_convert_transform(batch_transforms)
        self.dataset = dataset

    def __len__(self) -> int:
        """Returns the total size of the dataset."""
        return len(self.dataset)  # noqa

    def __getitem__(self, index: typing.Hashable) -> typing.Dict[str, typing.Any]:
        """Returns a single data batch loaded from the dataset at the given index."""
        batch = self._get_raw_batch(index)
        if self.batch_transforms:
            batch = self.batch_transforms(batch)
        assert isinstance(batch, typing.Dict), f"unexpected post-transform batch type: {type(batch)}"
        return batch

    def _get_raw_batch(
        self,
        index: typing.Hashable,
    ) -> typing.Any:
        """Returns a single data batch loaded from the dataset at a specified index."""
        batch = self.dataset[index]  # load the batch data using the base class impl
        assert isinstance(batch, tuple) and len(batch) == 2
        batch = {
            "data": batch[0],
            "target": batch[1],
            qut01.data.batch_size_key: 1,
            qut01.data.batch_id_key: f"{self._batch_id_prefix}mnist{index:06d}",
            qut01.data.batch_index_key: index,
            "class_names": [str(c) for c in range(10)],
        }
        return batch

    @property
    def tensor_names(self) -> typing.List[str]:
        """Names of the data tensors that will be provided in the loaded batches."""
        return self._tensor_names

    @property
    def dataset_info(self) -> typing.Dict[str, typing.Any]:
        """Returns metadata information parsed from the dataset (if any)."""
        return dict(name=self.dataset_name)

    @property
    def dataset_name(self) -> str:
        """Returns the dataset name used to identify this particular dataset."""
        return self._dataset_name

    def summary(self, *args, **kwargs) -> None:
        """Prints a summary of the dataset using the default logger.

        This function should be easy-to-call (parameter-free, if possible) and fast-to-return (takes
        seconds or tens-of-seconds at most) in order to remain friendly to high-level users. What it
        does specifically is totally up to the derived class.

        All outputs should be sent to the default logger.
        """
        import qut01.utils.logging

        logger = qut01.utils.logging.get_logger(__name__)
        logger.info(self)
        logger.info(f"dataset_name={self.dataset_name}, length={len(self)}")


def _main(data_root_path: pathlib.Path) -> None:
    mnist_data_folder = data_root_path / "mnist"
    datamodule = DataModule(data_dir=mnist_data_folder)
    datamodule.prepare_data()
    datamodule.setup()
    dataloader = datamodule.train_dataloader()
    print(f"training data loader has {len(dataloader)} batches")
    batch = next(iter(dataloader))
    print(f"batch keys: {batch.keys()}")


if __name__ == "__main__":
    import qut01.utils.config
    import qut01.utils.logging

    qut01.utils.logging.setup_logging_for_analysis_script()
    data_root_path_ = qut01.utils.config.get_data_root_dir()
    _main(data_root_path_)
