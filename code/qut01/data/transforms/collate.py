import dataclasses
import typing

import hydra.utils
import numpy as np
import omegaconf
import tokenizers
import torch
import torch.utils.data
import transformers

import qut01.data.batch_utils
import qut01.utils.config

if typing.TYPE_CHECKING:
    from qut01.data.batch_utils import BatchDictType
    from qut01.data.sentence_utils import SampleListType


logger = qut01.utils.logging.get_logger(__name__)


def default_collate(
    batches: typing.List["BatchDictType"],
    keys_to_batch_manually: typing.Sequence[str] = (),
    keys_to_ignore: typing.Sequence[str] = (),
) -> "BatchDictType":
    """Performs the default collate function while manually handling some given special cases."""
    assert isinstance(batches, (list, tuple)) and all(
        [isinstance(b, dict) for b in batches]
    ), f"unexpected type for batch array provided to collate: {type(batches)}"
    assert all(
        [len(np.setxor1d(list(batches[idx].keys()), list(batches[0].keys()))) == 0 for idx in range(1, len(batches))]
    ), "not all batches have the same sets of keys! (implement your own custom collate fn!)"
    avail_batch_keys = list(batches[0].keys())
    output = dict()
    # first step: look for the keys that we need to batch manually, and handle those
    default_keys_to_batch_manually = [
        qut01.data.batch_utils.batch_id_key,  # can be any hashable object type
    ]
    keys_to_batch_manually = {*keys_to_batch_manually, *default_keys_to_batch_manually}
    for key in keys_to_batch_manually:
        if key in avail_batch_keys:
            output[key] = [b[key] for b in batches]
    keys_to_skip_or_already_done = {*keys_to_ignore, *keys_to_batch_manually}
    output.update(
        torch.utils.data.default_collate(
            [{k: v for k, v in b.items() if k not in keys_to_skip_or_already_done} for b in batches]
        )
    )
    if qut01.data.batch_utils.batch_size_key not in output:
        output[qut01.data.batch_utils.batch_size_key] = len(batches)
    return output


@dataclasses.dataclass
class _InternalTokenizerOutput:
    token_ids_tensor: typing.Optional[torch.Tensor]
    attn_mask_tensor: typing.Optional[torch.Tensor]
    cls_token_indices: typing.Optional[torch.Tensor]
    target_token_mask: typing.Optional[torch.Tensor]
    samples_to_drop: typing.List[int]


class StatementSentenceCollater(torch.nn.Module):
    """Batch collate class to use when applying tokenizers across many sentences.

    Args:
        tokenizer: tokenizer object to use for sentence encoding. Assumed to be based on the
            huggingface `tokenizers.Tokenizer` or `transformers.AutoTokenizer` interface.
        apply_ignore_index: defines the 'ignore index' label value to use when applying 'dontcare'
            (ignore) masks onto the target tensors. If `None`, no masking or label updates are
            performed. Otherwise, we will replace label values in relevance/evidence tensors
            with this value. Can only be used if the relevance/evidence tensors are based on
            HARD labels, i.e. if they directly contain class indices as integers.
        input_sentence_data_key: key to use when fetching the sentence (sample) data objects from
            the loaded batch dictionary data.
        autotokenizer_ids_key: key to use when fetching encoded token ids from a tokenizer not
            derived from the `tokenizer.Tokenizer` interface.
        autotokenizer_attn_mask_key: key to use when fetching attention masks from a tokenizer not
            derived from the `tokenizer.Tokenizer` interface.
        keys_to_batch_manually: list of keys that should be kept from the input statement data
            dictionaries and batched (as a list) to be added to the output batch dictionary.
        prompt_text_to_prepend: text of the prompt (or any context for the model) to be prepended
            over every example.

    Output:
        A batch dictionary containing manually collated items along with the sentence and target
        tensors.
    """

    def __init__(
        self,
        tokenizer: typing.Optional[typing.Union[qut01.utils.config.DictConfig, tokenizers.Tokenizer]],
        apply_ignore_index: typing.Optional[int] = None,
        input_sentence_data_key: str = "sentence_data",  # input
        autotokenizer_ids_key: str = "input_ids",  # internal (for tokenizer)
        autotokenizer_attn_mask_key: str = "attention_mask",  # internal (for tokenizer)
        move_cls_token_to_target_sentence: bool = True,
        keys_to_batch_manually: typing.Optional[typing.Sequence[str]] = (),
        prompt_text_to_prepend: str = "",
    ):
        """Initializes the module by validating settings."""
        super().__init__()
        if tokenizer is not None:
            if isinstance(tokenizer, (dict, omegaconf.DictConfig)):
                tokenizer = hydra.utils.instantiate(tokenizer)
            if isinstance(tokenizer, tokenizers.Tokenizer):
                tokenizer.enable_padding()
                logger.info(f"tokenizer vocabulary size: {tokenizer.get_vocab_size()}")
        self.tokenizer = tokenizer
        self.apply_ignore_index = apply_ignore_index
        self.input_sentence_data_key = input_sentence_data_key
        self.autotokenizer_ids_key = autotokenizer_ids_key
        self.autotokenizer_attn_mask_key = autotokenizer_attn_mask_key
        self.move_cls_token_to_target_sentence = move_cls_token_to_target_sentence
        if not keys_to_batch_manually:
            keys_to_batch_manually = []
        self.keys_to_batch_manually = keys_to_batch_manually
        self.prompt_text_to_prepend = prompt_text_to_prepend.strip()

    def forward(
        self,
        batches: typing.List["BatchDictType"],
    ) -> "BatchDictType":
        """Collates the given batches into a new batch dict with the required tensors."""
        assert all([isinstance(b, dict) for b in batches]), "unexpected batch dict type"
        assert all([self.input_sentence_data_key in b for b in batches]), "missing sentence data obj"
        sentence_data = []
        for batch in batches:
            if isinstance(batch[self.input_sentence_data_key], list):
                sentence_data.extend(batch[self.input_sentence_data_key])
            else:
                sentence_data.append(batch[self.input_sentence_data_key])
        output_batch = self._gather_and_collate_inputs_and_targets(sentence_data=sentence_data)
        if output_batch is None:
            return None  # noqa  (happens when we truncate with big context windows and small batch sizes)
        assert not (set(output_batch.keys()) & set(self.keys_to_batch_manually)), "bad key overlap?"
        for key in self.keys_to_batch_manually:
            output_batch[key] = [b[key] for b in batches]
        return output_batch

    def _encode_with_tokenizer(
        self,
        text_batch_to_encode: typing.List[str],
        text_target_mask_batch: typing.Optional[typing.List[typing.List[bool]]] = None,  # used w/ context only
    ) -> _InternalTokenizerOutput:
        """Returns the outputs of the tokenizer, i.e. token ids and attention masks.

        If no tokenizer is defined, all outputs will be `None` instead.

        If using a target mask (meaning we have a target sentence inside a wider text block), we
        might want to reposition the CLS token at a new location. If true, it may be that placing
        the CLS token at the very beginning of the target text is impossible: this is because the
        encoded text is TRUNCATED automatically by the tokenizer so that we do not exceed the
        maximum token sequence length of the model. In such cases, we identify the sample as
        "to be dropped", and it should be ignored from the collate function in the parent caller.
        """
        assert text_target_mask_batch is None or len(text_batch_to_encode) == len(text_target_mask_batch)
        sample_count = len(text_batch_to_encode)
        if self.tokenizer is None:
            return _InternalTokenizerOutput(
                token_ids_tensor=None,
                attn_mask_tensor=None,
                cls_token_indices=None,
                target_token_mask=None,
                samples_to_drop=[],
            )
        if isinstance(self.tokenizer, tokenizers.Tokenizer):
            raise NotImplementedError  # old implementation below; does not handle CLS token...
            # encodings = self.tokenizer.encode_batch(text_batch_to_encode)
            # token_ids_tensor = torch.as_tensor([e.ids for e in encodings])
            # attn_mask_tensor = torch.as_tensor([e.attention_mask for e in encodings])
        encodings = self.tokenizer(text_batch_to_encode, padding=True, truncation=True, return_tensors="pt")
        assert isinstance(
            encodings, (dict, transformers.tokenization_utils_base.BatchEncoding)
        ), f"unrecognized tokenizer output: {type(encodings)}"
        assert (
            self.autotokenizer_ids_key in encodings
        ), f"tokenizer output not found: {self.autotokenizer_attn_mask_key}"
        token_ids_tensor = encodings[self.autotokenizer_ids_key]
        assert (
            self.autotokenizer_attn_mask_key in encodings
        ), f"tokenizer output not found: {self.autotokenizer_attn_mask_key}"
        attn_mask_tensor = encodings[self.autotokenizer_attn_mask_key]
        cls_token_indices = torch.as_tensor([0] * sample_count, dtype=torch.int64)
        target_token_mask = torch.zeros_like(token_ids_tensor, dtype=torch.bool)
        samples_to_drop = []

        if samples_to_drop and len(samples_to_drop) < sample_count:
            # if we identified samples we need to drop, get rid of their corresponding output tensors
            token_ids_tensor = torch.stack(
                [token_ids_tensor[sidx] for sidx in range(sample_count) if sidx not in samples_to_drop],
            )
            attn_mask_tensor = torch.stack(
                [attn_mask_tensor[sidx] for sidx in range(sample_count) if sidx not in samples_to_drop],
            )
            cls_token_indices = torch.stack(
                [cls_token_indices[sidx] for sidx in range(sample_count) if sidx not in samples_to_drop],
            )
            target_token_mask = torch.stack(
                [target_token_mask[sidx] for sidx in range(sample_count) if sidx not in samples_to_drop],
            )
        elif len(samples_to_drop) == sample_count:  # we dropped everything, no need to re-stack the tensors
            token_ids_tensor, attn_mask_tensor, cls_token_indices, target_token_mask = None, None, None, None
        return _InternalTokenizerOutput(
            token_ids_tensor=token_ids_tensor,
            attn_mask_tensor=attn_mask_tensor,
            cls_token_indices=cls_token_indices,
            target_token_mask=target_token_mask,
            samples_to_drop=samples_to_drop,
        )

    def _gather_and_collate_inputs_and_targets(
        self,
        sentence_data: "SampleListType",
    ) -> "BatchDictType":
        """Prepares tensors of input token sequences and target criteria labels.

        The shape of the output token sequence tensor should be `(N, S)`, where N is the number of
        sequences (i.e. independent examples to classify), and S is the maximum (padded or cut off)
        sequence length. The output target label tensors should contain hard (0 or 1) or soft
        (float values between 0 and 1) labels corresponding to the relevance or evidence of the
        sequence for each of the targeted classes (or criteria). The shape of these tensors should
        be `(N, C)` where N is the number of sequences, and C is the number of target classes (or
        criteria).

        All tensors are returned along with a class names list and a label types list in a batch
        dictionary that may also contain additional manually specified items (via the `keys_to_batch_manually`
        setting).
        """
        if not len(sentence_data):
            return None  # noqa -- garbage in, garbage out; this should not happen, skip the batch
        assert all([isinstance(sdata, qut01.data.sentence_utils.SentenceData) for sdata in sentence_data])
        assert len({sdata.label_strategy for sdata in sentence_data}) == 1, "should have static label types?"
        assert len({sdata.target_criteria for sdata in sentence_data}) == 1, "can only have one class setup"
        target_criteria = sentence_data[0].target_criteria
        label_strategy = sentence_data[0].label_strategy

        # first, fetch the full text to analyze (with potential prompt+context) and encode it
        text_batch = [s.text for s in sentence_data]
        text_target_mask_batch = [s.target_text_mask for s in sentence_data]
        text_batch, text_target_mask_batch = self._add_prompt_if_available(text_batch, text_target_mask_batch)
        tokenized_text = self._encode_with_tokenizer(text_batch, text_target_mask_batch)
        if tokenized_text.samples_to_drop:  # we might drop samples if the target is out-of-bounds after truncation
            sentence_data = [s for sidx, s in enumerate(sentence_data) if sidx not in tokenized_text.samples_to_drop]
        if not sentence_data:
            return None  # noqa -- again, this is a waste of compute, but it's hard to avoid

        # next, fetch the TARGET SENTENCES (only! no context! no prompt!) and encode those
        target_sentences_batch = [s.target_text for s in sentence_data]
        tokenized_sentences = self._encode_with_tokenizer(target_sentences_batch)

        # time to batch and convert the target labels into tensors used for training/evaluation
        if label_strategy.startswith("hard"):
            relevance_labels = torch.as_tensor([s.relevance_labels for s in sentence_data]).float()
            evidence_labels = torch.as_tensor([s.evidence_labels for s in sentence_data]).float()
        else:
            assert self.apply_ignore_index is None, "cannot apply ignore index when using soft labels!"
            relevance_labels = torch.as_tensor([s.relevance_labels for s in sentence_data]).float()
            evidence_labels = torch.as_tensor([s.evidence_labels for s in sentence_data]).float()
        relevance_dontcare_mask = torch.as_tensor([s.relevance_dontcare_mask for s in sentence_data])
        evidence_dontcare_mask = torch.as_tensor([s.evidence_dontcare_mask for s in sentence_data])
        if self.apply_ignore_index is not None:
            assert relevance_labels.shape == relevance_dontcare_mask.shape
            assert evidence_labels.shape == evidence_dontcare_mask.shape
            relevance_labels[relevance_dontcare_mask] = self.apply_ignore_index
            evidence_labels[evidence_dontcare_mask] = self.apply_ignore_index

        # we create unique batch identifiers by combining the statement identifiers with target sentence idxs
        batch_ids = []
        for sdata in sentence_data:
            if len(sdata.orig_sentence_idxs) == 1:
                sentence_idx_id = f"{sdata.orig_sentence_idxs[0]:05d}"
            else:
                sentence_idx_id = "s[" + ",".join([str(idx) for idx in sdata.orig_sentence_idxs]) + "]"
            batch_ids.append(f"statement{sdata.statement_id:05d}:sentence{sentence_idx_id}")

        # all done, store all the batched tensors into a dictionary off for the regular collate fn
        output = {
            "sentence_token_ids": tokenized_sentences.token_ids_tensor,
            "sentence_attention_mask": tokenized_sentences.attn_mask_tensor,
            "sentence_orig_text": target_sentences_batch,
            "sentence_orig_idxs": [s.orig_sentence_idxs for s in sentence_data],
            "text_token_ids": tokenized_text.token_ids_tensor,
            "text_attention_mask": tokenized_text.attn_mask_tensor,
            "text_cls_token_indices": tokenized_text.cls_token_indices,
            "text_target_token_mask": tokenized_text.target_token_mask,
            "text_target_mask": text_target_mask_batch,
            "text": text_batch,
            "relevance": relevance_labels,
            "evidence": evidence_labels,
            "relevance_dontcare_mask": relevance_dontcare_mask,
            "evidence_dontcare_mask": evidence_dontcare_mask,
            "statement_id": [s.statement_id for s in sentence_data],
            "class_names": target_criteria,
            "dropped_sample_count": len(tokenized_text.samples_to_drop),
            qut01.data.batch_utils.batch_size_key: len(sentence_data),
            qut01.data.batch_utils.batch_id_key: batch_ids,
        }
        return output

    def _add_prompt_if_available(
        self,
        text_batch,
        text_target_mask_batch,
    ) -> typing.Tuple[typing.List[str], typing.List[typing.List[bool]]]:
        if self.prompt_text_to_prepend:
            prompt = f"{self.prompt_text_to_prepend} "
            text_batch = [prompt + s for s in text_batch]
            text_target_mask_batch = [([False] * len(prompt)) + m for m in text_target_mask_batch]
        return text_batch, text_target_mask_batch
