"""Contains defines, types, and dataclasses used to define/store processed sentence data."""
import dataclasses
import typing

import numpy as np

if typing.TYPE_CHECKING:
    from qut01.data import SampleStrategyType
    from qut01.data.classif_utils import CriteriaNameType
    from qut01.data.statement_utils import StatementProcessedData

    ClassLabelArrayType = typing.Tuple[CriteriaNameType, ...]


supported_label_types = tuple(
    [
        "relevance",  # whether the text is RELEVANT for the target annotation/criterion or not
        "evidence",  # whether the text shows that the entity FULFILS THEIR REQUIREMENTS or not
    ]
)
"""Name of the labels for which we can generate target tensors (per criterion/annotation type)."""

LabelType = typing.Literal[*supported_label_types]  # noqa

supported_label_strategies = tuple(
    [
        "hard_union",  # assign label as positive for sentence if ANY annotator thinks it should be
        "hard_intersection",  # assign label as positive for sentence if ALL annotators think it should be
        "soft",  # assign label as the average of 0/1 values assigned by all annotators
    ]
)
"""Name of the label generation strategies that can be used when generating target tensors."""

LabelStrategyType = typing.Literal[*supported_label_strategies]  # noqa
HardOrSoftLabelArrayType = typing.Tuple[typing.Union[bool, float], ...]
MaskLabelArrayType = typing.Tuple[bool, ...]


@dataclasses.dataclass
class SentenceData:
    """Holds references, metadata, and labels related to a sentence extracted from a statement."""

    text: str
    """Sentence(s) to be analyzed.

    These sentences should have been preprocessed (cleaned up of bad characters, newlines, etc.),
    but not tokenized yet, as that is a model-dependant operation.

    The string may contain MORE than just a single sentence if we merged multiple sentences from the
    same annotation chunk, or if we concatenated neighboring words (as extra context) from the
    statement. When it contains more than the target sentence, use the 'target_text_mask' attribute
    of this class to identify the sentence that should be targeted for classification.
    """

    target_text_mask: typing.List[bool]
    """Character-wise boolean attention mask indicating context vs target for all chars in `text`.

    If the character is part of the original sentence to be analyzed (as opposed to its context),
    then its corresponding boolean value in this mask is set to `True`.
    """

    label_strategy: LabelStrategyType
    """Name of the label generation strategy that was used to generate the label lists below."""

    sample_strategy: "SampleStrategyType"
    """Name of the sentence sampling strategy that was used to generate this sentence data."""

    context_word_count: int
    """Number of individual words added before/after the target sentence as context (if any)."""

    orig_sentence_idxs: typing.List[int]
    """Indices of the target text data matched to the original sentences of the statement."""

    statement_id: int
    """Identifier of the statement that contains this sentence."""

    relevance_labels: HardOrSoftLabelArrayType
    """Array of binary labels specifying whether the target text is relevant for each criteria.

    The length of this array corresponds to the number of criteria (classes) in `target_criteria`.

    When annotations exist, these labels can be hard positives/negatives values (booleans), or soft
    labels (floats between 0 and 1) depending on the label generation strategy. When annotations do
    not exist in the statement, the RELEVANCE label defaults to FALSE (0), but should be ignored.
    See the `relevance_dontcare_mask` attribute for more information.
    """

    relevance_dontcare_mask: MaskLabelArrayType
    """Array of binary labels specifying whether the relevance labels (above) should be ignored.

    The length of this array corresponds to the number of criteria (classes) in `target_criteria`.

    Relevance labels may not be valid if they are associated to criteria which are not annotated
    at all in the statement; those cases will be masked as "don't care" (i.e. ignored). This mask
    may be used when training or evaluating classifiers to avoid these invalid cases.
    """

    evidence_labels: HardOrSoftLabelArrayType
    """Array of binary labels specifying whether the target text is tied to a no/yes label.

    The length of this array corresponds to the number of criteria (classes) in `target_criteria`.

    When annotations exist, these labels can be hard positives/negatives values (booleans), or soft
    labels (floats between 0 and 1) depending on the label generation strategy. When annotations do
    not exist in the statement, or when the RELEVANCE label is FALSE (0), the EVIDENCE label
    defaults to FALSE (0). See the `evidence_dontcare_mask` attribute for more info.
    """

    evidence_dontcare_mask: MaskLabelArrayType
    """Array of binary labels specifying whether the evidence labels (above) should be ignored.

    The length of this array corresponds to the number of criteria (classes) in `target_criteria`.

    Evidence labels may not be valid if they are associated to criteria which are not annotated
    at all in the statement. Also, the evidence label can only ever be FALSE (0) when the relevance
    of the same sentence is FALSE (0); those cases will be masked as "don't care" (i.e. ignored).
    This mask may be used when training or evaluating classifiers to avoid these invalid cases.
    """

    target_criteria: "ClassLabelArrayType"
    """Array of criteria (class) names that map to the label indices in the above arrays.

    These classes correspond to types of annotations, which are themselves named after criteria;
    refer to `supported_classif_setups` in the `qut01.data.annotations.classes` module for more
    information.
    """

    @property
    def target_text(self) -> str:
        """Returns the target text to be classified inside the above `text` string.

        Note that when no context words are added, this is the same as `text`.
        """
        return "".join([self.text[idx] for idx in np.where(self.target_text_mask)[0]])


SampleType = SentenceData
SampleListType = typing.List[SentenceData]


def generate_binary_labels(
    statement_data: "StatementProcessedData",
    orig_sentence_idxs: typing.List[int],
    target_criteria: "ClassLabelArrayType",
    label_type: LabelType,
    label_strategy: LabelStrategyType,
) -> typing.Tuple[HardOrSoftLabelArrayType, MaskLabelArrayType]:
    """Generates a list of per-criterion relevance or evidence labels for the given sentences.

    Also returns the dontcare (ignore) mask to use alongside the label array when training or
    evaluating classifiers. The returned label array and mask will have a first dimension that
    matches the number of criteria we are studying. Each item in that list is either a boolean/float
    label or a list of booleans (for the dontcare mask).
    """
    assert label_type in supported_label_types
    assert label_strategy in supported_label_strategies
    output_labels, output_dontcare_mask = [], []
    for criterion_name in target_criteria:
        annotator_count = statement_data.annotation_counts.get(criterion_name, 0)
        if annotator_count == 0:
            # when the statement is not annotated, both label types default to False/0
            output_labels.append(False if label_strategy.startswith("hard") else 0.0)
            output_dontcare_mask.append(True)
            continue
        sentence_labels = []
        for orig_sidx in orig_sentence_idxs:
            annotation_count = statement_data.sentence_annotation_counts[orig_sidx].get(criterion_name, 0)
            annotation_chunks = [
                chunk
                for chunk in statement_data.sentence_annotation_chunks[orig_sidx]
                if chunk.annotation.name == criterion_name
            ]
            assert annotation_count <= len(annotation_chunks)
            assert annotation_count <= annotator_count
            if label_strategy == "hard_union":
                if label_type == "relevance":
                    # if the sentence has been extracted as supporting text by ANY annotator, it's relevant
                    label = annotation_count > 0
                else:  # label_type == "evidence"
                    # if the sentence is positive evidence according to ANY annotator, it keeps that label
                    label = any([c.annotation.label.name == "YES" for c in annotation_chunks])
            elif label_strategy == "hard_intersection":
                if label_type == "relevance":
                    # if the sentence has been extracted as supporting text by ALL annotators, it's relevant
                    label = annotation_count == annotator_count
                else:  # label_type == "evidence"
                    # if the sentence is positive evidence according to ALL annotators, it keeps that label
                    label = all([c.annotation.label.name == "YES" for c in annotation_chunks])
            else:  # label_strategy == "soft":
                if label_type == "relevance":
                    # the proportion of annotators that extracted it as supporting text give the soft label weight
                    label = annotation_count / annotator_count
                else:
                    # the proportion of annotators that labeled it as YES give the soft label weight
                    label = sum([c.annotation.label.name == "YES" for c in annotation_chunks]) / len(annotation_chunks)
            sentence_labels.append(label)
        # merge all sentence labels into one global label for all sentences
        if label_strategy == "hard_union":
            output_labels.append(any(sentence_labels))
        elif label_strategy == "hard_intersection":
            output_labels.append(all(sentence_labels))
        else:  # if label_strategy == "soft":
            output_labels.append(sum(sentence_labels) / len(sentence_labels))
        output_dontcare_mask.append(False)
    return tuple(output_labels), tuple(output_dontcare_mask)
