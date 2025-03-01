"""Implements generic classifier modules based on Lightning."""
import typing
from typing import Mapping, Union

import hydra
import lightning.pytorch.loggers as pl_loggers
import omegaconf
import torch
import torch.nn.functional
import torch.utils.tensorboard.writer
import torchmetrics
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torchmetrics import Metric, MetricCollection

import qut01.data
import qut01.utils
from qut01.models.utils import BaseModel

logger = qut01.utils.logging.get_logger(__name__)
TorchModuleOrDictConfig = typing.Union[torch.nn.Module, qut01.utils.DictConfig]


class GenericClassifier(BaseModel):
    """Example of LightningModule used for image classification tasks.

    This class is derived from the framework's base model interface, and it implements all the
    extra goodies required for automatic rendering/logging of predictions. The input data and
    classification label attributes required to ingest and evaluate predictions are assumed to be
    specified via keys in the loaded batch dictionaries. The exact keys should be specified to the
    constructor of this class.

    This particular implementation expects to get a "backbone" encoder configuration alongside a
    "head" classification layer configuration. Embeddings generated in the forward pass are also
    reshaped automatically, based on the assumption that we generate one embedding per image.

    For more information on the role and responsibilities of the LightningModule, see:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

    For more information on the base class, see:
        `qut01.models.utils.BaseModel`
    """

    def __init__(
        self,
        encoder: TorchModuleOrDictConfig,
        head: typing.Optional[TorchModuleOrDictConfig],
        loss_fn: typing.Optional[TorchModuleOrDictConfig],
        metrics: typing.Optional[qut01.utils.DictConfig],
        optimization: typing.Optional[qut01.utils.DictConfig],
        num_output_classes: int,
        num_input_channels: int,
        input_key: str = "input",
        label_key: str = "label",
        ignore_mask_key: typing.Optional[str] = None,
        ignore_index: typing.Optional[int] = None,
        save_hyperparams: bool = True,  # turn this off in derived classes
        **kwargs,
    ):
        """Initializes the LightningModule and its submodules, loss, metrics, and optimizer.

        Note: we favor passing everything in as dict configs that can be used to instantiate
        modules directly as this seems to be the 'cleanest' way to log everything needed to
        reinstantiate the model from scratch without having to serialize the modules directly...

        Args:
            encoder: dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the backbone encoder of the model. If a config is provided, it
                will be used to instantiate the backbone encoder via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object.
            head: optional dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the backbone encoder of the model. If a config is provided, it
                will be used to instantiate the classifier via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object. If nothing is provided, we will
                assume that the backbone encoder already possesses a classifier, and will
                compute the loss directly on the backbone's output.
            loss_fn: optional dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the loss function of the model. If a config is provided, it
                will be used to instantiate the loss function via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object. If nothing is provided, we will
                assume that the model already implements its own loss, and a derived class will be
                computing it in its own override of the `generic_step` function.
            metrics: dict-based configuration that corresponds to the metrics to be instantiated
                during training/validation/testing. It must be possible to return the result
                of instantiating this config via `hydra.utils.instantiate` directly to a
                `torchmetrics.MetricCollection` object. If no config is provided, it will default
                to an accuracy metric only.
            optimization: dict-based configuration that can be used to instantiate the
                optimizer used by the trainer. This config is assumed to be formatted according
                to Lightning's format. See the base class's `configure_optimizers` for more info.
            num_output_classes: number of unique classes (categories) to be predicted.
            num_input_channels: number of channels in the images to be loaded.
            input_key: key used to fetch the input data tensor from the loaded batch dictionaries.
            label_key: key used to fetch the class label tensor from the loaded batch dictionaries.
            ignore_mask_key: key used to fetch the dontcare (ignore) mask tensor from the loaded
                batch dictionaries. None = not used.
            ignore_index: value used to indicate dontcare predictions in label arrays. None = not
                used. Can only be used when working with "hard" labels, i.e. labels specified via
                class indices.
            save_hyperparams: toggles whether hyperparameters should be saved in this class. This
                should be `False` when this class is derived, and the `save_hyperparameters`
                function should be called in the derived constructor.
        """
        assert num_output_classes >= 1, f"invalid number of output classes: {num_output_classes}"
        assert num_input_channels >= 1, f"invalid number of input channels: {num_input_channels}"
        self.num_output_classes = num_output_classes
        self.num_input_channels = num_input_channels
        self.input_key, self.label_key = input_key, label_key
        self.ignore_index, self.ignore_mask_key = ignore_index, ignore_mask_key
        assert isinstance(metrics, (dict, omegaconf.DictConfig)), f"incompatible metrics type: {type(metrics)}"
        self._metrics_config = metrics
        if save_hyperparams:
            # this line allows us to access hparams with `self.hparams` + auto-stores them in checkpoints
            self.save_hyperparameters(logger=False)  # logger=False since we don't need duplicated logs
        super().__init__(
            optimization=optimization,
            **kwargs,
        )
        if isinstance(encoder, (dict, omegaconf.DictConfig)):
            encoder = hydra.utils.instantiate(encoder)
        assert isinstance(encoder, torch.nn.Module), f"incompatible encoder type: {type(encoder)}"
        self.encoder = encoder
        if head is not None:  # if none, we will just not use it, and return encoder logits directly
            if isinstance(head, (dict, omegaconf.DictConfig)):
                head = hydra.utils.instantiate(head)
            assert isinstance(head, torch.nn.Module), f"incompatible head type: {type(head)}"
        self.head = head

        if loss_fn is not None:  # if none, user will have to override generic_step to provide their own
            if isinstance(loss_fn, (dict, omegaconf.DictConfig)):
                loss_fn = hydra.utils.instantiate(loss_fn)
            assert isinstance(loss_fn, torch.nn.Module), f"incompatible loss_fn type: {type(loss_fn)}"

        self.loss_fn = loss_fn

    def configure_metrics(self) -> torchmetrics.MetricCollection:
        """Configures and returns the metric objects to update when given predictions + labels."""
        metrics = hydra.utils.instantiate(self._metrics_config)
        assert isinstance(metrics, (dict, omegaconf.DictConfig)), f"invalid metric dict type: {type(metrics)}"
        if isinstance(metrics, omegaconf.DictConfig):
            metrics = omegaconf.OmegaConf.to_container(
                cfg=metrics,
                resolve=True,
                throw_on_missing=True,
            )
        assert all([isinstance(k, str) for k in metrics.keys()])
        assert all([isinstance(v, torchmetrics.Metric) for v in metrics.values()])
        return torchmetrics.MetricCollection(metrics)

    def log_dict(
        self, dictionary: Union[Mapping[str, typing.Union[Metric, Tensor, Union[int, float]]], MetricCollection], **kwd
    ):
        assert self.labels is not None

        for k, v in dictionary.items():
            if v.numel() == 1:
                super().log(name=k, value=v, **kwd)
            else:
                assert v.ndim == 1
                for i in range(v.shape[0]):
                    super().log(name=f"{k}/{self.labels[i]}", value=v[i], **kwd)

    def forward(self, batch: qut01.data.BatchDictType) -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        assert self.input_key in batch, f"missing mandatory '{self.input_key}' tensor from batch"
        input_tensor = batch[self.input_key]
        assert input_tensor.ndim >= 2
        batch_size, tensor_shape = input_tensor.shape[0], input_tensor.shape[1:]
        assert batch_size == qut01.data.get_batch_size(batch)
        embed = self.encoder(input_tensor)
        assert embed.ndim >= 2 and embed.shape[0] == batch_size
        embed = torch.flatten(embed, start_dim=1)
        if self.head is not None:
            logits = self.head(embed)
        else:
            logits = embed
        assert logits.ndim == 2 and logits.shape == (batch_size, self.num_output_classes)
        return logits

    def _generic_step(
        self,
        batch: qut01.data.BatchDictType,
        batch_idx: int,
    ) -> typing.Dict[str, typing.Any]:
        """Runs a generic version of the forward + evaluation step for the train/valid/test loops.

        In comparison with the regular `forward()` function, this function will compute the loss and
        return multiple outputs used to update the metrics based on the assumption that the batch
        dictionary also contains info about the target labels. This means that it should never be
        used in production, as we would then try to access labels that do not exist.

        This generic+default implementation will break if the forward pass returns more than a
        single prediction tensor, or if the target labels need to be processed or transformed in any
        fashion before being sent to the loss.
        """
        assert self.loss_fn is not None, "missing impl in derived class, no loss function defined!"
        preds = self(batch)  # this will call the 'forward' implementation above and return preds
        assert self.label_key in batch, f"missing mandatory '{self.label_key}' tensor from batch"
        target = batch[self.label_key]
        ignore_mask = None

        if self.ignore_mask_key is not None and self.ignore_mask_key in batch:
            ignore_mask = batch[self.ignore_mask_key]
        if ignore_mask is not None:
            loss = self.loss_fn(preds, target, ignore_mask)
        else:
            loss = self.loss_fn(preds, target)
        return {
            "loss": loss,  # mandatory for training loop, optional for validation/testing
            "preds": preds.detach(),  # used in metrics, logging, and potentially even returned to user
            "targets": target,  # so that metric update functions have access to the tensor itself
            "ignore_mask": ignore_mask,
            "ignore_index": self.ignore_index,
            qut01.data.batch_size_key: qut01.data.get_batch_size(batch),  # so that logging can use it
        }

    def _render_and_log_samples(
        self,
        loop_type: str,  # 'train', 'valid', or 'test'
        batch: qut01.data.BatchDictType,
        batch_idx: int,
        sample_idxs: typing.List[int],
        sample_ids: typing.List[typing.Hashable],
        outputs: typing.Dict[str, typing.Any],
        dataloader_idx: int = 0,
    ) -> typing.Any:
        """Renders and logs specific samples from the current batch using available loggers.

        Note: we assume that we are doing predictions for sentences taken from the AIMS dataset,
        that the predictions are for a binary class setup (negative/positive) and are logits
        (NOT probabilities!), and that we can display the sentences along with the predicted class
        probabilities via tensorboard or Comet. If any of these assumptions are not met, this will
        throw an exception.
        """
        assert len(sample_idxs) == len(sample_ids)
        if not isinstance(self.logger, (pl_loggers.TensorBoardLogger, pl_loggers.CometLogger)):
            return
        for sample_idx, sample_id in zip(sample_idxs, sample_ids):
            # note: this assumes we're running with preprocessing operations for the AIMS dataset!
            assert "sentence_orig_text" in batch, "missing required batch field: 'sentence_orig_text'"
            sentence_orig_text = batch["sentence_orig_text"][sample_idx]
            assert "text" in batch, "missing required batch field: 'text'"
            text = batch["text"][sample_idx]
            class_names = batch["class_names"]
            targets = batch[self.label_key][sample_idx].cpu().numpy()
            assert targets.shape == (len(class_names),), f"unexpected targets shape: {targets.shape}"
            dontcare = batch[f"{self.label_key}_dontcare_mask"][sample_idx].cpu().numpy()
            preds = outputs["preds"][sample_idx].cpu()  # should be CxN or N (C classes, N labels)
            # note: we assume we're doing binary classification, otherwise this base class can't do rendering
            assert preds.shape == (len(class_names),), f"unexpected preds shape: {preds.shape}"
            preds = torch.sigmoid(preds).numpy()  # we also assume that the model outputs logits, not probs
            output_str = f'TARGET TEXT: "{sentence_orig_text}"\n\n'
            output_str += f'FULL TEXT: "{text}"\n\n'
            for class_idx, class_name in enumerate(class_names):
                if not dontcare[class_idx]:
                    output_str += (
                        f"\n\n{class_name} {self.label_key}: "
                        f"target={targets[class_idx]:.02f}, "
                        f"pred={preds[class_idx]:.02f}"
                    )
            for loggr in self.loggers:
                if isinstance(loggr, pl_loggers.TensorBoardLogger):
                    assert hasattr(loggr, "experiment") and loggr.experiment is not None
                    assert isinstance(loggr.experiment, torch.utils.tensorboard.writer.SummaryWriter)
                    text_tag = f"{loop_type}/{sample_id}"
                    loggr.experiment.add_text(text_tag, output_str, global_step=self.global_step)
                else:
                    assert isinstance(loggr, pl_loggers.CometLogger)
                    assert hasattr(loggr, "experiment") and loggr.experiment is not None
                    metadata = dict(sample_id=sample_id, loop_type=loop_type)
                    loggr.experiment.log_text(output_str, step=self.global_step, metadata=metadata)
        return None

    def on_validation_batch_end(
        self,
        outputs: typing.Optional[STEP_OUTPUT],
        batch: qut01.data.BatchDictType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Completes the forward + evaluation step for the validation loop.

        Args:
            See parent class.

        Returns:
            Nothing.
        """
        if self.labels is None:
            self.labels = batch["class_names"]
        super().on_validation_batch_end(
            outputs=outputs, batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx
        )

    def on_test_batch_end(
        self,
        outputs: typing.Optional[STEP_OUTPUT],
        batch: qut01.data.BatchDictType,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Completes the forward + evaluation step for the testing loop.

        Args:
            See parent class.

        Returns:
            Nothing.
        """
        if self.labels is None:
            self.labels = batch["class_names"]
        super().on_test_batch_end(outputs=outputs, batch=batch, batch_idx=batch_idx, dataloader_idx=dataloader_idx)
