import typing

import qut01.data.preprocess_utils

if typing.TYPE_CHECKING:
    from qut01.data.annotations.classes import AnnotationsBase


def compute_inter_annotator_agreement(
    annotation1: "AnnotationsBase",
    annotation2: "AnnotationsBase",
) -> float:
    """Computes and returns the IoU-based inter-annotator-agreement (IAA) between two annotations.

    The approach used here is the Intersection over Union (IoU), also known as the Jaccard Index.

    Note that this assumes that the two input annotations are of the same statement, are of the same
    type, and share labels. If they are not of the same statement or type, an exception will be
    thrown. If they do not share labels (e.g. if one says "YES" and the other "NO"), the IAA will be
    returned as zero. This function will only look at the label value and its supporting text, and
    it will NOT consider any other annotation fields at all.

    The current implementation does NOT use any kind of soft matching across annotated sentences, so
    a single mismatched character (beyond those removed/cleaned via preprocessing) will alter the
    output score.
    """
    assert annotation1.name == annotation2.name, f"annotation type mismatch ({annotation1.name} vs {annotation2.name})"
    assert (
        annotation1.statement_id == annotation2.statement_id
    ), f"annotation statement mismatch ({annotation1.statement_id} vs {annotation2.statement_id})"
    if annotation1.label != annotation2.label:  # noqa
        return 0.0
    annot_sentences1, _ = qut01.data.preprocess_utils.get_preprocessed_sentences(
        raw_text=annotation1.supporting_text,
        text_source=qut01.data.preprocess_utils.TextSource.ANNOTATORS,
    )
    annot_sentences2, _ = qut01.data.preprocess_utils.get_preprocessed_sentences(
        raw_text=annotation2.supporting_text,
        text_source=qut01.data.preprocess_utils.TextSource.ANNOTATORS,
    )
    return compute_intersection_over_union(
        set1=set(annot_sentences1),
        set2=set(annot_sentences2),
    )


def compute_intersection_over_union(
    set1: typing.Set[typing.Hashable],
    set2: typing.Set[typing.Hashable],
    zero_division_outcome: float = 1.0,
) -> float:
    """Computes and returns the intersection-over-union (IoU), or Jaccard index/score.

    This is a generic implementation of this function that works on two sets of hashable elements,
    where elements are either present or not across the two sets, defining the intersection or not.
    The union is defined as the set of all elements present in either input set.
    """
    intersection = set1 & set2
    union = set1 | set2
    if not union:
        return zero_division_outcome
    iou = len(intersection) / len(union)
    return iou
