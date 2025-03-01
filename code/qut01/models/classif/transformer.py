"""Implements hf-transformers-based classifier module based on Lightning."""
import math
import typing

import hydra
import omegaconf
import torch
import torch.nn.functional
import transformers
from peft import LoraConfig, get_peft_model

import qut01.data
import qut01.utils
from qut01.models.classif.base import GenericClassifier

logger = qut01.utils.logging.get_logger(__name__)
TorchModuleOrDictConfig = typing.Union[torch.nn.Module, qut01.utils.DictConfig]


class TransformerClassifier(GenericClassifier):
    """Example of LightningModule used for huggingface-transformers-based text classification.

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
        pretrained_model: TorchModuleOrDictConfig,
        head: typing.Optional[TorchModuleOrDictConfig],
        model_output_param_name: str,
        pretrained_model_fine_tuning_mode: dict,
        output_tensor_extra_dims: typing.Optional[typing.List[int]] = None,
        model_config_dim_attrib_name: str = "dim",
        input_key: str = "text_token_ids",
        input_attention_mask_key: str = "text_attention_mask",
        output_embedding_strategy: str = "cls_token",
        save_hyperparams: bool = True,  # turn this off in derived classes
        **kwargs,
    ):
        """Initializes the LightningModule and its submodules, loss, metrics, and optimizer.

        Note: we favor passing everything in as dict configs that can be used to instantiate
        modules directly as this seems to be the 'cleanest' way to log everything needed to
        reinstantiate the model from scratch without having to serialize the modules directly...

        Args:
            pretrained_model: dict-based configuration or `transformers.PreTrainedModel`-compatible
                object that corresponds to the backbone encoder of the model. If a config is
                provided, it will be used to instantiate the backbone encoder via
                `hydra.utils.instantiate`.
            head: optional dict-based configuration or `torch.nn.Module`-compatible object that
                corresponds to the classification head of the model. If a config is provided, it
                will be used to instantiate the classifier via `hydra.utils.instantiate`
                to create a `torch.nn.Module`-compatible object. If nothing is provided, we will
                assume that the backbone encoder already possesses a classifier, and will
                compute the loss directly on the backbone's output.
            model_output_param_name: name of the function to call on the output from the model
                to extract the logits (if the classification is already done), or the top
                embeddings (if the classification nis still to be done).
            output_tensor_extra_dims: optional list of extra output tensor dimensions to expect
                after the classification head. If some dimensions are provided but not found in the
                tensor provided by the head, the output tensor will be reshaped to fit these.
            pretrained_model_fine_tuning_mode: specifies how to fine tune the pre-trained model.
            model_config_dim_attrib_name: the attribute name for the model's output dimension
                count (or embedding size), stored inside its internal config.
            input_key: key used to fetch the input data tensor from the loaded batch dictionaries.
            input_attention_mask_key: key used to fetch the sentence attention mask tensor from the
                loaded batch dictionary.
            output_embedding_strategy: specifies which output embedding strategy to use to convert
                the token sequence into a fixed-dimension embedding. Can be `cls_token` or
                `target_max_pool`. Has no effect if not using a classification head.
            save_hyperparams: toggles whether hyperparameters should be saved in this class. This
                should be `False` when this class is derived, and the `save_hyperparameters`
                function should be called in the derived constructor.

        See the base class constructor for more info on the other arguments.
        """
        if save_hyperparams:
            # this line allows us to access hparams with `self.hparams` + auto-stores them in checkpoints
            self.save_hyperparameters(logger=False)  # logger=False since we don't need duplicated logs
        if isinstance(pretrained_model, (dict, omegaconf.DictConfig)):
            pretrained_model = hydra.utils.instantiate(pretrained_model)
        assert isinstance(
            pretrained_model, transformers.PreTrainedModel
        ), f"incompatible pretrained model type: {type(pretrained_model)}"
        assert hasattr(
            pretrained_model.config, model_config_dim_attrib_name
        ), f"missing model confg attribute: {model_config_dim_attrib_name}"
        model_embedding_size = getattr(pretrained_model.config, model_config_dim_attrib_name)
        if not output_tensor_extra_dims:
            output_tensor_extra_dims = []
        output_tensor_extra_dims = tuple(output_tensor_extra_dims)
        if isinstance(head, (dict, omegaconf.DictConfig)):
            # assume that the head config needs the input dim count as a constructor argument
            head = hydra.utils.instantiate(head, model_embedding_size)

        if pretrained_model_fine_tuning_mode["type"] == "frozen":
            for name, child in pretrained_model.named_children():
                if name not in pretrained_model_fine_tuning_mode["except"]:
                    for param in child.parameters():
                        param.requires_grad = False
        elif pretrained_model_fine_tuning_mode["type"] == "lora":
            config = LoraConfig(
                modules_to_save=["classifier"],
            )
            pretrained_model = get_peft_model(pretrained_model, config)
        elif pretrained_model_fine_tuning_mode["type"] == "full":
            pass  # nothing to do
        else:
            raise ValueError(
                f"pretrained_model_fine_tuning_mode.type=" f"{pretrained_model_fine_tuning_mode} not supported"
            )

        super().__init__(
            encoder=pretrained_model,
            head=head,
            num_input_channels=1,
            input_key=input_key,
            save_hyperparams=False,
            **kwargs,
        )
        self.model_embedding_size = model_embedding_size
        self.output_tensor_extra_dims = output_tensor_extra_dims
        self.input_attention_mask_key = input_attention_mask_key
        self.model_output_param_name = model_output_param_name
        assert output_embedding_strategy in ["cls_token", "target_max_pool"]
        self.output_embedding_strategy = output_embedding_strategy

    def forward(self, batch: qut01.data.BatchDictType) -> torch.Tensor:
        """Forwards batch data through the model, similar to `torch.nn.Module.forward()`."""
        assert all(
            [k in batch for k in [self.input_key, self.input_attention_mask_key]]
        ), "missing at least one of the mandatory input keys from the loaded batch"
        input_tensor = batch[self.input_key]
        input_attn_mask = batch[self.input_attention_mask_key]
        assert input_tensor.ndim >= 2
        batch_size, tensor_shape = input_tensor.shape[0], input_tensor.shape[1:]
        assert batch_size == qut01.data.get_batch_size(batch)
        model_output = self.encoder(
            input_ids=input_tensor,
            attention_mask=input_attn_mask,
        )
        output_tensor = getattr(model_output, self.model_output_param_name)

        if self.head is not None:
            assert output_tensor.shape == (batch_size, *tensor_shape, self.model_embedding_size)
            if self.output_embedding_strategy == "cls_token":
                # we will rely on the CLS token to provide the embedding to attach to the classifier
                # (code below assumes that the CLS token is either at a specified or default location)
                assert "text_cls_token_indices" in batch, "text_cls_token_indices should be explicitly specified"
                cls_indices = batch.get("text_cls_token_indices")
                batch_indices = torch.arange(batch_size)
                expected_cls_tokens = input_tensor[batch_indices, cls_indices]
                assert len(torch.unique(expected_cls_tokens)) == 1  # should have single CLS token id?
                output_embedding = output_tensor[batch_indices, cls_indices]  # B x N x C => B x C
            elif self.output_embedding_strategy == "target_max_pool":
                # we will max-pool all the embeddings of tokens that belong to the target sentence
                # (if the text is only the target sentence without context, this max-pools across all tokens)
                target_token_mask = batch["text_target_token_mask"]
                output_embedding_list = []
                for sample_idx in range(batch_size):  # number of target tokens varies for each sample
                    target_embeddings = output_tensor[sample_idx, target_token_mask[sample_idx]]  # N x C => L x C
                    (output_embedding, _) = torch.max(target_embeddings, dim=0)  # L x C => C
                    output_embedding_list.append(output_embedding)  # stack along batch dim
                output_embedding = torch.stack(output_embedding_list)  # back to B x C
            else:
                raise ValueError(f"{self.output_embedding_strategy} is not a supported output_embedding_strategy")
            return self.head(output_embedding)
        else:
            return output_tensor
