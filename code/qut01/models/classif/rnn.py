"""Implements an RNN-based classifier module based on Lightning."""
import math
import typing

import hydra
import omegaconf
import torch
import torch.nn.functional

import qut01.data
import qut01.utils
from qut01.models.classif.base import GenericClassifier

logger = qut01.utils.logging.get_logger(__name__)
TorchModuleOrDictConfig = typing.Union[torch.nn.Module, qut01.utils.DictConfig]


class RNNClassifier(GenericClassifier):
    """Example of LightningModule used for RNN-based text classification.

    This class is derived from the framework's base model interface; the input data and class label
    attributes required to ingest and evaluate predictions are assumed to be specified via keys in
    the loaded batch dictionaries. The exact keys should be specified to the constructor.

    For more information on the role and responsibilities of the LightningModule, see:
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html

    For more information on the base class, see:
        `qut01.models.classif.GenericClassifier`
        `qut01.models.utils.BaseModel`
    """

    def __init__(
        self,
        embedding: TorchModuleOrDictConfig,
        encoder: TorchModuleOrDictConfig,
        head: typing.Optional[TorchModuleOrDictConfig],
        output_tensor_extra_dims: typing.Optional[typing.List[int]] = None,
        freeze_embedding_model: bool = False,
        freeze_encoder_model: bool = False,
        input_key: str = "text_token_ids",
        input_attention_mask_key: str = "text_attention_mask",
        save_hyperparams: bool = True,  # turn this off in derived classes
        **kwargs,
    ):
        """Initializes the LightningModule and its submodules, loss, metrics, and optimizer.

        Note: we favor passing everything in as dict configs that can be used to instantiate
        modules directly as this seems to be the 'cleanest' way to log everything needed to
        reinstantiate the model from scratch without having to serialize the modules directly...

        Args:
            embedding: dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the token embedding layer of the model. If a config is provided, it
                will be instantiated via `hydra.utils.instantiate`.
            encoder: dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the recurrent layer of the model. If a config is provided, it
                will be instantiated via `hydra.utils.instantiate`.
            head: dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the classification head of the model. If a config is provided, it
                will be used to instantiate the classifier via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object.
            output_tensor_extra_dims: optional list of extra output tensor dimensions to expect
                after the classification head. If some dimensions are provided but not found in the
                tensor provided by the head, the output tensor will be reshaped to fit these.
            freeze_embedding_model: specifies whether the embedding layer's parameters should be
                frozen or not.
            freeze_encoder_model: specifies whether the embedding layer's parameters should be
                frozen or not.
            model_config_dim_attrib_name: the attribute name for the model's output dimension
                count (or embedding size), stored inside its internal config.
            input_key: key used to fetch the input data tensor from the loaded batch dictionaries.
            input_attention_mask_key: key used to fetch the sentence attention mask tensor from the
                loaded batch dictionary.
            save_hyperparams: toggles whether hyperparameters should be saved in this class. This
                should be `False` when this class is derived, and the `save_hyperparameters`
                function should be called in the derived constructor.

        See the base class constructor for more info on the other arguments.
        """
        if isinstance(embedding, (dict, omegaconf.DictConfig)):
            embedding = hydra.utils.instantiate(embedding)
        assert isinstance(embedding, torch.nn.Module), f"bad embedding type: {type(embedding)}"
        if freeze_embedding_model:
            for param in embedding.parameters():
                param.requires_grad = False
        if isinstance(encoder, (dict, omegaconf.DictConfig)):
            encoder = hydra.utils.instantiate(encoder)
        assert isinstance(encoder, torch.nn.Module), f"bad encoder type: {type(encoder)}"
        if freeze_encoder_model:
            for param in encoder.parameters():
                param.requires_grad = False
        if not output_tensor_extra_dims:
            output_tensor_extra_dims = []
        output_tensor_extra_dims = tuple(output_tensor_extra_dims)
        assert head is not None, "head should be configured (can't leave empty!)"
        if save_hyperparams:
            # this line allows us to access hparams with `self.hparams` + auto-stores them in checkpoints
            self.save_hyperparameters(logger=False)  # logger=False since we don't need duplicated logs
        super().__init__(
            encoder=encoder,
            head=head,
            num_input_channels=1,
            input_key=input_key,
            save_hyperparams=False,
            **kwargs,
        )
        self.embedding = embedding
        self.output_tensor_extra_dims = output_tensor_extra_dims
        self.input_attention_mask_key = input_attention_mask_key

    def forward(self, batch: qut01.data.BatchDictType) -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        assert all(
            [k in batch for k in [self.input_key, self.input_attention_mask_key]]
        ), "missing at least one of the mandatory input keys from the loaded batch"
        input_tensor = batch[self.input_key]
        assert input_tensor.ndim == 2
        batch_size, seq_len = input_tensor.shape
        assert batch_size == qut01.data.get_batch_size(batch)
        # TODO: figure out if we can/should use the input attention mask here?
        embeddings = self.embedding(input_tensor)
        assert embeddings.ndim == 3 and embeddings.shape[:2] == (batch_size, seq_len)
        _, (hidden_state, _) = self.encoder(embeddings)
        assert hidden_state.ndim == 3 and hidden_state.shape[1] == batch_size  # (n_layers x batch_size x hidden_dim)
        # we attach the classifier to the last layer's hidden state
        logits = self.head(hidden_state[-1])
        return logits
