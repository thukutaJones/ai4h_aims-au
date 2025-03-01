"""Contains defines and utility functions for the QUT01-AIMS classification task setup."""
import typing

ANNOT_APPROVAL_CLASS_NAME = "approval"
ANNOT_SIGNATURE_CLASS_NAME = "signature"
ANNOT_C1_REPENT_CLASS_NAME = "c1 (reporting entity)"
ANNOT_C2_STRUCT_CLASS_NAME = "c2 (structure)"
ANNOT_C2_OPS_CLASS_NAME = "c2 (operations)"
ANNOT_C2_SUPPCH_CLASS_NAME = "c2 (supply chains)"
ANNOT_C3_RISK_CLASS_NAME = "c3 (risk description)"
ANNOT_C4_MITIG_CLASS_NAME = "c4 (risk mitigation)"
ANNOT_C4_REMED_CLASS_NAME = "c4 (remediation)"
ANNOT_C5_EFFECT_CLASS_NAME = "c5 (effectiveness)"
ANNOT_C6_CONSULT_CLASS_NAME = "c6 (consultation)"
ANNOT_CLASS_NAMES = tuple(
    [
        ANNOT_APPROVAL_CLASS_NAME,
        ANNOT_SIGNATURE_CLASS_NAME,
        ANNOT_C1_REPENT_CLASS_NAME,
        ANNOT_C2_STRUCT_CLASS_NAME,
        ANNOT_C2_OPS_CLASS_NAME,
        ANNOT_C2_SUPPCH_CLASS_NAME,
        ANNOT_C3_RISK_CLASS_NAME,
        ANNOT_C4_MITIG_CLASS_NAME,
        ANNOT_C4_REMED_CLASS_NAME,
        ANNOT_C5_EFFECT_CLASS_NAME,
        ANNOT_C6_CONSULT_CLASS_NAME,
    ]
)
ANNOT_APPROVSIGNC1_META_CLASS_NAME = "a-s-c1"
ANNOT_APPROVSIGNC1_CLASS_NAMES = tuple(
    [
        ANNOT_APPROVAL_CLASS_NAME,
        ANNOT_SIGNATURE_CLASS_NAME,
        ANNOT_C1_REPENT_CLASS_NAME,
    ]
)
ANNOT_C2C3C4C5C6_META_CLASS_NAME = "c2-c3-c4-c5-c6"
ANNOT_C2C3C4C5C6_CLASS_NAMES = tuple(
    [
        ANNOT_C2_STRUCT_CLASS_NAME,
        ANNOT_C2_OPS_CLASS_NAME,
        ANNOT_C2_SUPPCH_CLASS_NAME,
        ANNOT_C3_RISK_CLASS_NAME,
        ANNOT_C4_MITIG_CLASS_NAME,
        ANNOT_C4_REMED_CLASS_NAME,
        ANNOT_C5_EFFECT_CLASS_NAME,
        ANNOT_C6_CONSULT_CLASS_NAME,
    ]
)
ANNOT_META_CLASS_NAMES = tuple(
    [
        ANNOT_APPROVSIGNC1_META_CLASS_NAME,
        ANNOT_C2C3C4C5C6_META_CLASS_NAME,
    ]
)
ANNOT_CLASS_NAME_TO_META_CLASS_NAME = {
    class_name: (
        ANNOT_APPROVSIGNC1_META_CLASS_NAME
        if class_name in ANNOT_APPROVSIGNC1_CLASS_NAMES
        else ANNOT_C2C3C4C5C6_META_CLASS_NAME
    )
    for class_name in ANNOT_CLASS_NAMES
}
ANNOT_META_CLASS_NAME_TO_CLASS_NAMES_MAP = {
    ANNOT_APPROVSIGNC1_META_CLASS_NAME: ANNOT_APPROVSIGNC1_CLASS_NAMES,
    ANNOT_C2C3C4C5C6_META_CLASS_NAME: ANNOT_C2C3C4C5C6_CLASS_NAMES,
}

CriteriaNameType = typing.Literal[*ANNOT_CLASS_NAMES]  # noqa

supported_classif_setups = [
    "any",  # generate binary (relevance/evidence) classification target tensors for all 11 criteria
    *ANNOT_META_CLASS_NAMES,  # special setups for criteria subgroups
    *ANNOT_CLASS_NAMES,  # special setups for individual criteria
]
"""Name of the annotation groups across which we can generate target classification labels."""

ClassifSetupType = typing.Literal[*supported_classif_setups]  # noqa


def convert_classif_setup_to_list_of_criteria(
    classif_setup: ClassifSetupType,
) -> typing.List[str]:
    """Converts the classification target for a set of annotations into a list of annotation names.

    This is useful when specifying a classification setup for specific annotations (e.g. approval,
    signature, and C1) as a simple string inside configs (e.g. "a-s-c1"): this function will return
    the list of full annotation names involved with the classification setup. The order of the
    resulting list also maps directly to the target tensor dimensions that should be created in the
    processed data transformation operations.
    """
    assert classif_setup in supported_classif_setups, f"unexpected classif setup: {classif_setup}"
    if classif_setup in ANNOT_META_CLASS_NAMES:
        annot_names = ANNOT_META_CLASS_NAME_TO_CLASS_NAMES_MAP[classif_setup]
    else:
        if classif_setup == "any":
            annot_names = ANNOT_CLASS_NAMES
        else:
            annot_names = [classif_setup]
    return annot_names


classif_setup_to_criteria_count_map = {
    name: len(convert_classif_setup_to_list_of_criteria(name)) for name in supported_classif_setups
}
"""Map of number of classification targets (criteria) that a model should predict.

The map is indexed by the classification setup, defined above, and returns how many annotations are
associated with a particular setup. To get the exact dimension count for your classifier's output
layer, multiply the count returned here by two (for relevance and evidence).
"""
