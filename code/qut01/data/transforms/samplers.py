"""Contains functions used to process/extract samples from processed statement data.

TODO: could add data augmentation operations based on dropout, masking, or random replacement
      of context? (or based on random sentence merging? translations?)
"""
import itertools
import typing

import numpy as np
import torch

import qut01.data.sentence_utils
import qut01.data.statement_utils
from qut01.data.annotations.chunks import ChunkOrigin

if typing.TYPE_CHECKING:
    from qut01.data import ClassifSetupType, LabelStrategyType
    from qut01.data.batch_utils import BatchDictType
    from qut01.data.sentence_utils import SampleListType, SentenceData
    from qut01.data.statement_utils import StatementProcessedData


supported_sample_strategies = tuple(
    [
        "all",  # all sentences will be individually sampled (whether annotated or not)
        "subchunk",  # all sentences inside annotated text chunks will be individually sampled
        "chunk",  # annotated text chunks (i.e. text chunks split by '//') will be used as samples
    ]
)
"""Name of the sentence sampling strategies that can be used when generating sentence data."""

SampleStrategyType = typing.Literal[*supported_sample_strategies]  # noqa


class SentenceSampler(torch.nn.Module):
    """Sentence data sampling module that operates on preprocessed statement data.

    This module will extract sentence data from statements and return that data (with target labels
    and other information) as a list of `SentenceData` objects in the loaded batch dictionaries.
    See the `SentenceData` class for more details on the extracted information. Each `SentenceData`
    instance corresponds to an example (or 'sample') that a model may try to generate label
    predictions for: there are up to 11 classes (one for each type of annotation), and two types
    of labels (relevance and evidence), for a total of up to 22 predictions per example.

    Sentences that do not possess any associated annotation will also be sampled and turned into
    classification examples with target labels. These examples will however always possess
    'irrelevant' labels for all criteria,

    Args:
        classif_setup: specifies the classification setup to use in order to generate label lists
            for each sentence. If `None`, then all sentences of the statements will be considered
            and extracted, but many of them might not have any label lists (as annotations may be
            missing). If `any`, all sentences that have at least one type of annotation will be
            extracted. Otherwise, only sentences with annotations matching the provided class setup
            will be loaded.
        label_strategy: specifies the strategy to use when generating sentence labels. Hard labels
            correspond to boolean values, and soft labels to float values between 0 and 1.
        sample_strategy: specifies the strategy to use when sampling sentences to be extracted from
            the raw processed statement data.
        context_word_count: specifies the number of individual words to be added before/after the
            target sentence as context (if any).
        randomly_merge_irrelevant_sentences: specifies whether to randomly merge irrelevant
            sentences together, i.e. sentences that are not associated with any annotation, and
            that should never be considered relevant (or provide evidence) for any criterion even
            when randomly merged together. This is a form of data augmentation.
        random_subsample_count: specifies a number of sentences to (randomly) keep for each
            statement. Optional; if `None`, then all sentences are kept.
        use_chunks_from_viz_elems: specifies whether supporting text chunks that originate from
            visual element (tables and figures) should be sampled from or not.
        sentence_join_token: string used to join preprocessed sentences back together.
        input_statement_data_key: key to use when looking for the processed statement data in the
            loaded batch dictionaries.
        output_sentence_data_key: key to use when inserting the processed sentence data back into
            the loaded batch dictionaries. Should not overlap with an existing key!
    """

    def __init__(
        self,
        classif_setup: "ClassifSetupType",
        label_strategy: "LabelStrategyType",
        sample_strategy: "SampleStrategyType",
        context_word_count: int = 0,
        left_context_boundary_token=None,
        right_context_boundary_token=None,
        randomly_merge_irrelevant_sentences: bool = False,
        random_subsample_count: typing.Optional[int] = None,
        use_chunks_from_viz_elems: bool = False,
        sentence_join_token: str = ". ",
        input_statement_data_key: str = "processed_data",
        output_sentence_data_key: str = "sentence_data",
    ):
        """Initializes the module by validating settings."""
        super().__init__()
        assert (
            classif_setup is None or classif_setup in qut01.data.supported_classif_setups
        ), f"invalid classif target: {classif_setup}"
        self.classif_setup = classif_setup
        self.target_criteria = tuple(
            qut01.data.classif_utils.convert_classif_setup_to_list_of_criteria(
                classif_setup=classif_setup,
            ),
        )
        assert label_strategy in qut01.data.supported_label_strategies, f"invalid label strategy: {label_strategy}"
        self.label_strategy = label_strategy
        assert sample_strategy in qut01.data.supported_sample_strategies, f"invalid sample strategy: {sample_strategy}"
        self.sample_strategy = sample_strategy
        assert context_word_count >= 0, f"invalid context word count: {context_word_count}"

        self.context_word_count = context_word_count
        if context_word_count > 0:
            if left_context_boundary_token is None or right_context_boundary_token is None:
                raise ValueError(
                    "context_word_count > 0 means we are using context. "
                    "Please provide both left_context_boundary_token and "
                    "right_context_boundary_token"
                )
        self.left_context_boundary_token = left_context_boundary_token
        self.right_context_boundary_token = right_context_boundary_token

        self.randomly_merge_irrelevant_sentences = randomly_merge_irrelevant_sentences
        assert (
            random_subsample_count is None or random_subsample_count > 0
        ), f"invalid random subsampled sentence count: {random_subsample_count}"
        self.random_subsample_count = random_subsample_count
        self.use_chunks_from_viz_elems = use_chunks_from_viz_elems
        self.sentence_join_token = sentence_join_token
        self.input_statement_data_key = input_statement_data_key
        self.output_sentence_data_key = output_sentence_data_key

    def forward(self, batch: "BatchDictType") -> "BatchDictType":
        """Processes the given batch dictionary to extract its sentence data samples."""
        assert (
            self.input_statement_data_key in batch
        ), f"missing statement data in batch dict under expected key: {self.input_statement_data_key}"
        assert (
            self.output_sentence_data_key not in batch
        ), f"unexpected item in batch dict under sentence data key: {self.output_sentence_data_key}"
        statement_data = batch[self.input_statement_data_key]
        assert isinstance(statement_data, qut01.data.statement_utils.StatementProcessedData)
        if self.sample_strategy == "subchunk":
            samples = self._get_samples_from_annotation_subchunks(statement_data)
        elif self.sample_strategy == "chunk":
            samples = self._get_samples_from_annotation_chunks(statement_data)
        else:  # self.sample_strategy == "all":
            samples = self._get_samples_from_all_sentences(statement_data)
        samples = self._update_sample_list_with_missing_irrelevant_sentences(samples, statement_data)
        if self.random_subsample_count is not None:
            picked_sample_idxs = np.random.permutation(len(samples))[: self.random_subsample_count]
            samples = [samples[idx] for idx in picked_sample_idxs]
        batch[self.output_sentence_data_key] = samples
        return batch

    def _get_samples_from_annotation_chunks(
        self,
        statement_data: "StatementProcessedData",
    ) -> "SampleListType":
        """Prepares and returns sentence data samples for all annotated chunks in the statement."""
        output_samples = []
        for chunk in statement_data.annotation_chunks:
            if chunk.annotation.name not in self.target_criteria:
                continue
            if not self.use_chunks_from_viz_elems and chunk.chunk_origin != ChunkOrigin.main_body:
                continue
            output_samples.append(
                self._create_sentence_data_obj(
                    statement_data=statement_data,
                    target_sentence_idxs=chunk.matched_sentences_orig_idxs,
                    target_sentences=[statement_data.sentences[sidx] for sidx in chunk.matched_sentences_orig_idxs],
                )
            )
        return output_samples

    def _get_samples_from_annotation_subchunks(
        self,
        statement_data: "StatementProcessedData",
    ) -> "SampleListType":
        """Prepares and returns sentence data samples for all chunk sentences in the statement."""
        output_samples = []
        for chunk in statement_data.annotation_chunks:
            if chunk.annotation.name not in self.target_criteria:
                continue
            if not self.use_chunks_from_viz_elems and chunk.chunk_origin != ChunkOrigin.main_body:
                continue
            for orig_sentence_idx in chunk.matched_sentences_orig_idxs:
                output_samples.append(
                    self._create_sentence_data_obj(
                        statement_data=statement_data,
                        target_sentence_idxs=[orig_sentence_idx],
                        target_sentences=[statement_data.sentences[orig_sentence_idx]],
                    )
                )
        return output_samples

    def _get_samples_from_all_sentences(
        self,
        statement_data: "StatementProcessedData",
    ) -> "SampleListType":
        """Prepares and returns sentence data samples for all sentences in the statement."""
        output_samples = []
        for sentence_idx, sentence in enumerate(statement_data.sentences):
            if not self.use_chunks_from_viz_elems:
                sentence_chunks = statement_data.sentence_annotation_chunks[sentence_idx]
                if sentence_chunks and any([c.chunk_origin != ChunkOrigin.main_body for c in sentence_chunks]):
                    continue
            output_samples.append(
                self._create_sentence_data_obj(
                    statement_data=statement_data,
                    target_sentence_idxs=[sentence_idx],
                    target_sentences=[sentence],
                )
            )
        return output_samples

    def _create_sentence_data_obj(
        self,
        statement_data: "StatementProcessedData",
        target_sentence_idxs: typing.List[int],
        target_sentences: typing.List[str],
    ) -> "SentenceData":
        """Creates and returns a sentence data object filled with necessary labels/metadata.

        Note: this is where extra context might be added to create an "example" where we try to
        classify the target sentences with extra words. If no extra context is required, then
        only the target sentences provided as input will be included in the generated example.
        """
        assert len(target_sentences) == 1
        # fetch the relevance/evidence labels for the target sentence(s)
        relevance_labels, relevance_dontcare = qut01.data.sentence_utils.generate_binary_labels(
            statement_data=statement_data,
            orig_sentence_idxs=target_sentence_idxs,
            target_criteria=self.target_criteria,
            label_type="relevance",
            label_strategy=self.label_strategy,
        )
        evidence_labels, evidence_dontcare = qut01.data.sentence_utils.generate_binary_labels(
            statement_data=statement_data,
            orig_sentence_idxs=target_sentence_idxs,
            target_criteria=self.target_criteria,
            label_type="evidence",
            label_strategy=self.label_strategy,
        )
        # update the evidence dontcare mask to ignore all cases where relevance is null
        evidence_dontcare = tuple(
            [orig_dontcare or not rel_label for orig_dontcare, rel_label in zip(evidence_dontcare, relevance_labels)]
        )
        # combine the target sentences into a contiguous text block with separators tokens
        target_text = self.sentence_join_token.join(target_sentences)
        output_text = target_text
        # finally, add context (if needed) while keeping mask updated to only focus on target sentences
        if self.context_word_count:
            full_text = self.sentence_join_token.join(statement_data.sentences)
            left = full_text[0 : full_text.index(target_sentences[0])].split(" ")
            right = full_text[full_text.index(target_sentences[-1]) + len(target_sentences[-1]) :].split(" ")
            to_add_left = left[-self.context_word_count // 2 :]
            to_add_right = right[: self.context_word_count // 2]
            output_text = (
                " ".join(to_add_left) + f" {self.left_context_boundary_token} {target_text} "
                f"{self.right_context_boundary_token}" + " ".join(to_add_right)
            ).strip()

        target_mask = [True for _ in range(len(output_text))]  # FIXME: not useful anymore
        output = qut01.data.sentence_utils.SentenceData(
            text=output_text,  # contains the 'target' sentence augmented with more context words
            target_text_mask=target_mask,  # identifies the 'target' sentence vs the context
            label_strategy=self.label_strategy,
            sample_strategy=self.sample_strategy,
            context_word_count=self.context_word_count,
            orig_sentence_idxs=target_sentence_idxs,
            statement_id=statement_data.id,
            relevance_labels=relevance_labels,
            relevance_dontcare_mask=relevance_dontcare,
            evidence_labels=evidence_labels,
            evidence_dontcare_mask=evidence_dontcare,
            target_criteria=self.target_criteria,
        )
        return output

    def _update_sample_list_with_missing_irrelevant_sentences(
        self,
        samples: "SampleListType",
        statement_data: "StatementProcessedData",
    ) -> "SampleListType":
        """Updates the given sample list with missing irrelevant sentences."""
        sampled_sentence_idxs = [idx for sample in samples for idx in sample.orig_sentence_idxs]
        missing_sentence_idxs = [
            idx for idx in range(len(statement_data.sentences)) if idx not in sampled_sentence_idxs
        ]  # these are indices for 'irrelevant' sentences that we'll also consider
        if self.randomly_merge_irrelevant_sentences:
            raise NotImplementedError  # TODO!
            # also: determine if we should merge towards relevant sentence length stats?
        else:
            for sentence_idx in missing_sentence_idxs:
                if len(statement_data.sentence_annotation_chunks[sentence_idx]) > 0:
                    # it's annotated for at least one chunk, so it's not "irrelevant"
                    continue
                samples.append(
                    self._create_sentence_data_obj(
                        statement_data=statement_data,
                        target_sentence_idxs=[sentence_idx],
                        target_sentences=[statement_data.sentences[sentence_idx]],
                    )
                )
        return samples
