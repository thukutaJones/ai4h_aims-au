"""Implements a data module for the QUT01-AIMS annotated sentences dataset.

Note: the code below assumes that the dataset has been prepared with the following scripts:

    qut01/data/scripts/parse_raw_statements_data.py  (to prepare the raw PDF data)
    qut01/data/scripts/add_annotations_to_raw_dataset.py  (to add annotations)

The result should be a deeplake dataset located in the "data" directory of the project, i.e.
the directory you should have configured as an environment variable during framework setup. You
may also directly download the dataset from the QUT01-AIMS google drive shared folder. Unzip the
deeplake dataset in the "data" directory of the project, and it should be ready to be used by
this datamodule.
"""
import typing

import torch

import qut01.data.transforms.wrappers
import qut01.utils.logging
from qut01.data.datamodules.statement_sampler import DataModule as StatementDataModule

if typing.TYPE_CHECKING:
    import torch.utils.data

logger = qut01.utils.logging.get_logger(__name__)


class DataModule(StatementDataModule):
    """Datamodule for the QUT01-AIMS annotated dataset.

    This implementation will create data loaders that iterate over individual sentences (prepared
    with or without extra context) associated with classification labels based on annotations to
    provide examples for the criterion-wise classification of text as "relevant" or "irrelevant",
    and as "positive" or "negative" evidence.

    The `prepare_data` method of this datamodule will make sure that the dataset is available and
    ready to be used, and prepare the data split based on known statement entities/trademarks. The
    `setup` method will create the actual dataset parsers and dataloaders (should be called on the
    device where the data will ultimately be used).

    Args:
        sentence_buffer_size: the size of the buffer used in the internal dataset wrapper.
        save_hyperparams: toggles whether hyperparameters should be saved in this class. This
            should be `False` when this class is derived, and the `save_hyperparameters` function
            should be called in the derived constructor.
        kwargs: additional keyword arguments passed to the base (statement sampler) constructor.
    """

    def __init__(
        self,
        sentence_buffer_size: int,
        shuffle_train_buffer: bool = True,
        save_hyperparams: bool = True,  # turn this off in derived classes
        **kwargs,
    ):
        """Initializes the data module.

        Note: it might look like we're not using the provided args at all, but we are actually
        automatically saving those to the `hparams` attribute (via the `save_hyperparameters`
        function) in order to use them later.
        """
        if save_hyperparams:
            # this line allows us to access hparams with `self.hparams` + auto-stores them in checkpoints
            self.save_hyperparameters(logger=False)  # logger=False since we don't need duplicated logs
        super().__init__(**kwargs)
        assert sentence_buffer_size > 0, f"invalid buffer size: {sentence_buffer_size}"
        self.sentence_buffer_size = sentence_buffer_size
        self.shuffle_train_buffer = shuffle_train_buffer

    def _get_subset_parser(
        self,
        subset_sids: typing.List[int],
        subset_type: str,
    ) -> typing.Union[torch.utils.data.Dataset, torch.utils.data.IterableDataset]:
        """Creates and returns a data subset parser which can be used in a dataloader."""
        parser = super()._get_subset_parser(subset_sids, subset_type)
        if subset_type == "train":
            parser = qut01.data.transforms.wrappers.IterableDataset(
                dataset_to_wrap=parser,
                target_array_keys=["sentence_data"],
                shuffle=self.shuffle_train_buffer,
                buffer_size=self.sentence_buffer_size,
                buffer_refill_threshold=self.sentence_buffer_size // 2,
            )
        else:
            parser = qut01.data.transforms.wrappers.IterableDataset(
                dataset_to_wrap=parser,
                target_array_keys=["sentence_data"],
            )
        return parser


if __name__ == "__main__":
    import qut01.utils.config

    qut01.utils.logging.setup_logging_for_analysis_script()
    config_ = qut01.utils.config.init_hydra_and_compose_config(
        overrides=["data=sentence_sampler.yaml"],
    )
    qut01.data.datamodules.statement_sampler._local_main(config_)  # noqa
