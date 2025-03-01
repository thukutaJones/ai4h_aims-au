"""Contains defines and utility functions used to parse annotated labels from CSVs."""
import typing

from qut01.data.classif_utils import (
    ANNOT_APPROVAL_CLASS_NAME,
    ANNOT_APPROVSIGNC1_META_CLASS_NAME,
    ANNOT_C1_REPENT_CLASS_NAME,
    ANNOT_C2_OPS_CLASS_NAME,
    ANNOT_C2_STRUCT_CLASS_NAME,
    ANNOT_C2_SUPPCH_CLASS_NAME,
    ANNOT_C2C3C4C5C6_META_CLASS_NAME,
    ANNOT_C3_RISK_CLASS_NAME,
    ANNOT_C4_MITIG_CLASS_NAME,
    ANNOT_C4_REMED_CLASS_NAME,
    ANNOT_C5_EFFECT_CLASS_NAME,
    ANNOT_C6_CONSULT_CLASS_NAME,
    ANNOT_CLASS_NAME_TO_META_CLASS_NAME,
    ANNOT_CLASS_NAMES,
    ANNOT_SIGNATURE_CLASS_NAME,
)

if typing.TYPE_CHECKING:
    from qut01.data.classif_utils import CriteriaNameType

# below are defines for keys that hold annotated values in the repackaged dataset
# (see the repackaging scripts and CSVs for more info; these are unlikely useful outside this file)

# note that vars with a list of strings instead of a single string correspond to different versions
# (the a-s-c1 annotations and the c2-c3-c4-c5-c6 annotations had different column names...)

_ANNOT_PREFIX = "annotations/"
_LAST_UPDATE_SUFFIX = "_last_update"
_HAS_DATE_SUFFIX = "_date"
_DATE_TEXT_SUFFIX = "_Date_Text"
_DATE_TEXT_SCANNED_SUFFIX = "_Date_Scanned"
_NO_SUPPORT_INFO_SUFFIXES = ["_No_supporting_information", "_no_supporting_information"]
_YES_LABEL_TEXT_SUFFIXES = ["_Yes_text", "_text"]
_NO_LABEL_TEXT_SUFFIXES = ["_No_text", "_no_text"]
_UNCLEAR_LABEL_TEXT_SUFFIXES = ["_Unclear_text", "_unclear_text"]
_SCANNED_LABEL_SUFFIXES = ["_Scanned", "_scanned"]
_JOINT_OPT_LABEL_SUFFIX = "_Joint_Options"
_HAS_IMAGE_LABEL_SUFFIX = "_Image"
_VIZELEM_LABEL_SUFFIXES = ["_Visual_Element", "_visual_element"]
_VIZELEM_PAGE_SUFFIXES = [_VIZELEM_LABEL_SUFFIXES[0] + "_Page", _VIZELEM_LABEL_SUFFIXES[1] + "_page"]
_VIZELEM_SCANNED_SUFFIXES = [_VIZELEM_LABEL_SUFFIXES[0] + "_Scanned", _VIZELEM_LABEL_SUFFIXES[1] + "_scanned"]
_VIZELEM_TEXT_SUFFIXES = [_VIZELEM_LABEL_SUFFIXES[0] + "_Text", _VIZELEM_LABEL_SUFFIXES[1] + "_text"]
_VIZELEM_TYPE_SUFFIXES = [_VIZELEM_LABEL_SUFFIXES[0] + "_Type", _VIZELEM_LABEL_SUFFIXES[1] + "_type"]
_ADDVIZELEM_LABEL_SUFFIXES = ["_Additional_Visual_Element", "_add_element"]
_ADDVIZELEM_PAGE_SUFFIXES = [_ADDVIZELEM_LABEL_SUFFIXES[0] + "_Page", _ADDVIZELEM_LABEL_SUFFIXES[1] + "_page"]
_ADDVIZELEM_SCANNED_SUFFIXES = [_ADDVIZELEM_LABEL_SUFFIXES[0] + "_Scanned", _ADDVIZELEM_LABEL_SUFFIXES[1] + "_scanned"]
_ADDVIZELEM_TEXT_SUFFIXES = [_ADDVIZELEM_LABEL_SUFFIXES[0] + "_Text", _ADDVIZELEM_LABEL_SUFFIXES[1] + "_text"]
_ADDVIZELEM_TYPE_SUFFIXES = [_ADDVIZELEM_LABEL_SUFFIXES[0] + "_Type", _ADDVIZELEM_LABEL_SUFFIXES[1] + "_type"]

_ANNOT_APPROVSIGNC1_PREFIX = _ANNOT_PREFIX + ANNOT_APPROVSIGNC1_META_CLASS_NAME + "/"
_ANNOT_APPROVSIGNC1_VALID = _ANNOT_APPROVSIGNC1_PREFIX + "annotated"
_ANNOT_APPROVAL_PREFIX = _ANNOT_APPROVSIGNC1_PREFIX + "Approval"
_ANNOT_SIGNATURE_PREFIX = _ANNOT_APPROVSIGNC1_PREFIX + "Signature"
_ANNOT_C1_PREFIX = _ANNOT_APPROVSIGNC1_PREFIX + "Criteria_1"
_ANNOT_C2C3C4C5C6_PREFIX = _ANNOT_PREFIX + ANNOT_C2C3C4C5C6_META_CLASS_NAME + "/"
_ANNOT_C2C3C4C5C6_VALID = _ANNOT_C2C3C4C5C6_PREFIX + "annotated"
_ANNOT_C2_STRUCT_PREFIX = _ANNOT_C2C3C4C5C6_PREFIX + "c2_structure"
_ANNOT_C2_STRUCT_PREFIX_MINI = _ANNOT_C2C3C4C5C6_PREFIX + "c2_s"
_ANNOT_C2_OPS_PREFIX = _ANNOT_C2C3C4C5C6_PREFIX + "c2_operations"
_ANNOT_C2_OPS_PREFIX_MINI = _ANNOT_C2C3C4C5C6_PREFIX + "c2_op"
_ANNOT_C2_SUPPCH_PREFIX = _ANNOT_C2C3C4C5C6_PREFIX + "c2_supply_chain"
_ANNOT_C2_SUPPCH_PREFIX_MINI = _ANNOT_C2C3C4C5C6_PREFIX + "c2_sc"
_ANNOT_C3_RISK_PREFIX = _ANNOT_C2C3C4C5C6_PREFIX + "c3_risk_description"
_ANNOT_C3_RISK_PREFIX_MINI = _ANNOT_C2C3C4C5C6_PREFIX + "c3"
_ANNOT_C4_MITIG_PREFIX = _ANNOT_C2C3C4C5C6_PREFIX + "c4_risk_mitigation"
_ANNOT_C4_REMED_PREFIX = _ANNOT_C2C3C4C5C6_PREFIX + "c4_ms_remediation"
_ANNOT_C5_EFFECT_PREFIX = _ANNOT_C2C3C4C5C6_PREFIX + "c5_assess_effectiveness"
_ANNOT_C5_EFFECT_PREFIX_MINI = _ANNOT_C2C3C4C5C6_PREFIX + "c5"
_ANNOT_C6_CONSULT_PREFIX = _ANNOT_C2C3C4C5C6_PREFIX + "c6_consultation"
_ANNOT_C6_CONSULT_PREFIX_MINI = _ANNOT_C2C3C4C5C6_PREFIX + "c6"

YES_LABELS = ["yes", "Yes", "YES", "y", "Y", "True", "true", True, 1]
NO_LABELS = ["no", "No", "NO", "n", "N", "False", "false", False, 0]
UNCLEAR_LABELS = ["unclear", "Unclear", "UNCLEAR"]
_OPT_LABEL_PREFIXES = ["Option ", "Option", "OPTION ", "OPTION", "Opt", "OPT"]
OPT1_LABELS = [p + "1" for p in _OPT_LABEL_PREFIXES] + ["1", 1]
OPT2_LABELS = [p + "2" for p in _OPT_LABEL_PREFIXES] + ["2", 2]
OPT3_LABELS = [p + "3" for p in _OPT_LABEL_PREFIXES] + ["3", 3]

LABEL_SUFFIXES_TO_CONVERT_TO_BOOL = [
    _HAS_IMAGE_LABEL_SUFFIX,
    _HAS_DATE_SUFFIX,
    _DATE_TEXT_SCANNED_SUFFIX,
    *_NO_SUPPORT_INFO_SUFFIXES,
    *_VIZELEM_LABEL_SUFFIXES,
    *_VIZELEM_SCANNED_SUFFIXES,
    *_ADDVIZELEM_LABEL_SUFFIXES,
    *_ADDVIZELEM_SCANNED_SUFFIXES,
    # below are other custom labels used as fixes for some statement(s)
    "_no_supporting_info",
    "_add_element",
    "_visual_element_scan",
    "_add_element_scan",
    "_add_visual_element",
]


class AnnotationKeysBase:
    """Helper class to hold annotation keys generated for each type of annotation.

    This helper class is not intended to be used outside of this module; if you are looking for
    the actual annotation objects that hold the data provided by annotators, refer to the
    `AnnotationsBase` class and its derivations.
    """

    name: "CriteriaNameType"
    """The name of the annotation class that possesses the keys (defined in derived classes).

    This name is tied to the CRITERION associated with the parent annotation object, and is used to
    determine whether the annotation is relevant for different classification task setups.
    """

    def __init__(
        self,
        class_prefix: str,
        with_scanned_label: bool = False,
        with_date: bool = False,
        with_viz_elem: bool = True,
        with_add_viz_elem: bool = False,
        extra_fields: typing.Optional[typing.Dict[str, str]] = None,
        suffix_version: int = 0,
    ):
        """Prepares full annotation keys for the specified class + suffixes."""
        self._suffix_version = suffix_version
        v = suffix_version
        self.label = class_prefix
        self.has_no_support_info = class_prefix + _NO_SUPPORT_INFO_SUFFIXES[v]
        self.yes_text = class_prefix + _YES_LABEL_TEXT_SUFFIXES[v]
        self.no_text = class_prefix + _NO_LABEL_TEXT_SUFFIXES[v]
        self.unclear_text = class_prefix + _UNCLEAR_LABEL_TEXT_SUFFIXES[v]
        self._last_update = class_prefix + _LAST_UPDATE_SUFFIX  # private, as it won't always be defined
        if with_scanned_label:
            self.label_scanned = class_prefix + _SCANNED_LABEL_SUFFIXES[v]
        if with_date:
            self.has_date = class_prefix + _HAS_DATE_SUFFIX
            self.date_text = class_prefix + _DATE_TEXT_SUFFIX
            self.date_text_is_scanned = class_prefix + _DATE_TEXT_SCANNED_SUFFIX
        if with_viz_elem:
            self.viz_elem_label = class_prefix + _VIZELEM_LABEL_SUFFIXES[v]
            self.viz_elem_page = class_prefix + _VIZELEM_PAGE_SUFFIXES[v]
            self.viz_elem_scanned = class_prefix + _VIZELEM_SCANNED_SUFFIXES[v]
            self.viz_elem_text = class_prefix + _VIZELEM_TEXT_SUFFIXES[v]
            self.viz_elem_type = class_prefix + _VIZELEM_TYPE_SUFFIXES[v]
        if with_add_viz_elem:
            self.add_viz_elem_label = class_prefix + _ADDVIZELEM_LABEL_SUFFIXES[v]
            self.add_viz_elem_page = class_prefix + _ADDVIZELEM_PAGE_SUFFIXES[v]
            self.add_viz_elem_scanned = class_prefix + _ADDVIZELEM_SCANNED_SUFFIXES[v]
            self.add_viz_elem_text = class_prefix + _ADDVIZELEM_TEXT_SUFFIXES[v]
            self.add_viz_elem_type = class_prefix + _ADDVIZELEM_TYPE_SUFFIXES[v]
        if extra_fields is None:
            extra_fields = dict()
        for field_name, field_key in extra_fields.items():
            setattr(self, field_name, field_key)

    def keys(self) -> typing.List[str]:
        """Returns all keys that should be found for this annotation class in the dataset."""
        return [key for key in self.__dict__ if not key.startswith("_")]


class _ApprovalKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to the APPROVAL criterion."""

    name = ANNOT_APPROVAL_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the approval-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_APPROVAL_PREFIX,
            with_scanned_label=False,
            with_date=not basic_keys_only,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=False,
            extra_fields=dict(
                joint_opt_label=_ANNOT_APPROVAL_PREFIX + _JOINT_OPT_LABEL_SUFFIX,
                opt_3_explanation=_ANNOT_APPROVSIGNC1_PREFIX + "Option_3_explanation",
                extra_notes=_ANNOT_APPROVSIGNC1_PREFIX + "Notes",
            )
            if not basic_keys_only
            else dict(),
            suffix_version=0,
        )
        # no custom fix needed for approval keys!


class _SignatureKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to the SIGNATURE criterion."""

    name = ANNOT_SIGNATURE_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the signature-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_SIGNATURE_PREFIX,
            with_scanned_label=False,
            with_date=not basic_keys_only,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=False,
            extra_fields=dict(
                has_image=_ANNOT_SIGNATURE_PREFIX + _HAS_IMAGE_LABEL_SUFFIX,
            )
            if not basic_keys_only
            else dict(),
            suffix_version=0,
        )
        # no custom fix needed for signature keys!


class _Criterion1ReportEntKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to CRITERION 1 (C1)."""

    name = ANNOT_C1_REPENT_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the C1-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_C1_PREFIX,
            with_scanned_label=False,
            with_date=False,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=not basic_keys_only,
            extra_fields=dict(),
            suffix_version=0,
        )
        # fix: for some reason, the "yes_text" is under "Text" directly
        self.yes_text = _ANNOT_C1_PREFIX + "_Text"


class _Criterion2StructureKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to CRITERION 2 (C2), structure."""

    name = ANNOT_C2_STRUCT_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the C2-struct-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_C2_STRUCT_PREFIX_MINI,  # init with mini prefix. and fixes below
            with_scanned_label=not basic_keys_only,
            with_date=False,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=not basic_keys_only,
            extra_fields=dict(),
            suffix_version=1,
        )
        # we now update keys that use the 'full' prefix instead of the 'mini' prefix
        self.label = _ANNOT_C2_STRUCT_PREFIX
        self.yes_text = _ANNOT_C2_STRUCT_PREFIX + _YES_LABEL_TEXT_SUFFIXES[self._suffix_version]
        if not basic_keys_only:
            self.label_scanned = _ANNOT_C2_STRUCT_PREFIX + _SCANNED_LABEL_SUFFIXES[self._suffix_version]
            # also, the 'scanned' suffix for the visual elements is modified
            self.viz_elem_scanned = _ANNOT_C2_STRUCT_PREFIX_MINI + "_visual_element_scan"
            self.add_viz_elem_scanned = _ANNOT_C2_STRUCT_PREFIX_MINI + "_add_element_scan"
            # and there's an unexpected name for this one
            self.add_viz_elem_label = _ANNOT_C2_STRUCT_PREFIX_MINI + "_add_visual_element"


class _Criterion2OperationsKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to CRITERION 2 (C2), operations."""

    name = ANNOT_C2_OPS_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the C2-ops-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_C2_OPS_PREFIX_MINI,  # init with mini prefix. and fixes below
            with_scanned_label=not basic_keys_only,
            with_date=False,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=not basic_keys_only,
            extra_fields=dict(),
            suffix_version=1,
        )
        # we now update keys that use the 'full' prefix instead of the 'mini' prefix
        self.label = _ANNOT_C2_OPS_PREFIX
        self.yes_text = _ANNOT_C2_OPS_PREFIX + _YES_LABEL_TEXT_SUFFIXES[self._suffix_version]
        if not basic_keys_only:
            self.label_scanned = _ANNOT_C2_OPS_PREFIX + _SCANNED_LABEL_SUFFIXES[self._suffix_version]
            # also, the 'scanned' suffix for the visual elements is modified
            self.viz_elem_scanned = _ANNOT_C2_OPS_PREFIX_MINI + "_visual_element_scan"
            self.add_viz_elem_scanned = _ANNOT_C2_OPS_PREFIX_MINI + "_add_element_scan"


class _Criterion2SupplyChainsKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to CRITERION 2 (C2), supply chains."""

    name = ANNOT_C2_SUPPCH_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the C2-suppch-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_C2_SUPPCH_PREFIX_MINI,  # init with mini prefix. and fixes below
            with_scanned_label=not basic_keys_only,
            with_date=False,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=not basic_keys_only,
            extra_fields=dict(),
            suffix_version=1,
        )
        # we now update keys that use the 'full' prefix instead of the 'mini' prefix
        self.label = _ANNOT_C2_SUPPCH_PREFIX
        self.yes_text = _ANNOT_C2_SUPPCH_PREFIX + _YES_LABEL_TEXT_SUFFIXES[self._suffix_version]
        if not basic_keys_only:
            # also, the 'scanned' suffix for the visual elements is modified
            self.viz_elem_scanned = _ANNOT_C2_SUPPCH_PREFIX_MINI + "_visual_element_scan"
            self.add_viz_elem_scanned = _ANNOT_C2_SUPPCH_PREFIX_MINI + "_add_element_scan"
            # particular typo cases
            self.add_viz_elem_page = _ANNOT_C2_SUPPCH_PREFIX_MINI + "_ad_element_page"
            self.add_viz_elem_label = _ANNOT_C2_SUPPCH_PREFIX_MINI + "_add_visual_element"


class _Criterion3RiskDescKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to CRITERION 3 (C3)."""

    name = ANNOT_C3_RISK_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the C3-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_C3_RISK_PREFIX_MINI,  # init with mini prefix. and fixes below
            with_scanned_label=not basic_keys_only,
            with_date=False,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=not basic_keys_only,
            extra_fields=dict(),
            suffix_version=1,
        )
        # we now update keys that use the 'full' prefix instead of the 'mini' prefix
        self.label = _ANNOT_C3_RISK_PREFIX
        self.yes_text = _ANNOT_C3_RISK_PREFIX + _YES_LABEL_TEXT_SUFFIXES[self._suffix_version]
        if not basic_keys_only:
            self.label_scanned = _ANNOT_C3_RISK_PREFIX + _SCANNED_LABEL_SUFFIXES[self._suffix_version]
            # particular typo case
            self.add_viz_elem_label = _ANNOT_C3_RISK_PREFIX_MINI + "_add_visual_element"


class _Criterion4MitigationKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to CRITERION 4 (C4), mitigation."""

    name = ANNOT_C4_MITIG_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the C4-mit-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_C4_MITIG_PREFIX,
            with_scanned_label=not basic_keys_only,
            with_date=False,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=not basic_keys_only,
            extra_fields=dict(),
            suffix_version=1,
        )
        if not basic_keys_only:
            # we need to fix the "add_visual_element" key that's just "add_element" instead
            self.add_viz_elem_label = _ANNOT_C4_MITIG_PREFIX + "_add_element"


class _Criterion4RemediationKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to CRITERION 4 (C4), remediation."""

    name = ANNOT_C4_REMED_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the C4-rem-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_C4_REMED_PREFIX,
            with_scanned_label=not basic_keys_only,
            with_date=False,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=not basic_keys_only,
            extra_fields=dict(),
            suffix_version=1,
        )
        if not basic_keys_only:
            # we need to fix the "add_visual_element" key that's just "add_element" instead
            self.add_viz_elem_label = _ANNOT_C4_REMED_PREFIX + "_add_element"


class _Criterion5EffectKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to CRITERION 5 (C5)."""

    name = ANNOT_C5_EFFECT_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the C3-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_C5_EFFECT_PREFIX_MINI,  # init with mini prefix. and fixes below
            with_scanned_label=not basic_keys_only,
            with_date=False,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=not basic_keys_only,
            extra_fields=dict(),
            suffix_version=1,
        )
        # we now update keys that use the 'full' prefix instead of the 'mini' prefix
        self.label = _ANNOT_C5_EFFECT_PREFIX
        self.yes_text = _ANNOT_C5_EFFECT_PREFIX + _YES_LABEL_TEXT_SUFFIXES[self._suffix_version]
        if not basic_keys_only:
            # particular typo case
            self.add_viz_elem_label = _ANNOT_C5_EFFECT_PREFIX_MINI + "_add_visual_element"


class _Criterion6ConsultKeys(AnnotationKeysBase):
    """Helper class to hold annotation keys related to CRITERION 6 (C6)."""

    name = ANNOT_C6_CONSULT_CLASS_NAME
    """The name of the annotation class that possesses these keys."""

    def __init__(self, basic_keys_only: bool = False):
        """Prepares full annotation keys for the C3-related annotations."""
        super().__init__(
            class_prefix=_ANNOT_C6_CONSULT_PREFIX_MINI,  # init with mini prefix. and fixes below
            with_scanned_label=not basic_keys_only,
            with_date=False,
            with_viz_elem=not basic_keys_only,
            with_add_viz_elem=not basic_keys_only,
            extra_fields=dict(),
            suffix_version=1,
        )
        # we now update keys that use the 'full' prefix instead of the 'mini' prefix
        self.label = _ANNOT_C6_CONSULT_PREFIX
        self.yes_text = _ANNOT_C6_CONSULT_PREFIX + _YES_LABEL_TEXT_SUFFIXES[self._suffix_version]
        if not basic_keys_only:
            self.label_scanned = _ANNOT_C6_CONSULT_PREFIX + _SCANNED_LABEL_SUFFIXES[self._suffix_version]
            # all additional viz elems have a non-standard name here
            self.add_viz_elem_label = _ANNOT_C6_CONSULT_PREFIX_MINI + "_add_visual_element"
            self.add_viz_elem_page = _ANNOT_C6_CONSULT_PREFIX_MINI + "_add_visual_page"
            self.add_viz_elem_scanned = _ANNOT_C6_CONSULT_PREFIX_MINI + "_add_visual_scanned"
            self.add_viz_elem_text = _ANNOT_C6_CONSULT_PREFIX_MINI + "_add_visual_element_text"
            self.add_viz_elem_type = _ANNOT_C6_CONSULT_PREFIX_MINI + "_add_visual_type"
        # finally, the no_supporting_information key needs a fix
        self.has_no_support_info = _ANNOT_C6_CONSULT_PREFIX_MINI + "_no_supporting_info"


_ANNOT_CLASS_NAME_TO_KEYS_CONTAINER_TYPE_MAP = {
    ANNOT_APPROVAL_CLASS_NAME: _ApprovalKeys,
    ANNOT_SIGNATURE_CLASS_NAME: _SignatureKeys,
    ANNOT_C1_REPENT_CLASS_NAME: _Criterion1ReportEntKeys,
    ANNOT_C2_STRUCT_CLASS_NAME: _Criterion2StructureKeys,
    ANNOT_C2_OPS_CLASS_NAME: _Criterion2OperationsKeys,
    ANNOT_C2_SUPPCH_CLASS_NAME: _Criterion2SupplyChainsKeys,
    ANNOT_C3_RISK_CLASS_NAME: _Criterion3RiskDescKeys,
    ANNOT_C4_MITIG_CLASS_NAME: _Criterion4MitigationKeys,
    ANNOT_C4_REMED_CLASS_NAME: _Criterion4RemediationKeys,
    ANNOT_C5_EFFECT_CLASS_NAME: _Criterion5EffectKeys,
    ANNOT_C6_CONSULT_CLASS_NAME: _Criterion6ConsultKeys,
}


def create_annotation_keys_container(
    annotation_name: "CriteriaNameType",
    basic_keys_only: bool = False,  # basic keys = the ones that can be easily validated across all criteria
) -> AnnotationKeysBase:
    """Returns a tensor key container object that possesses all the keys for a criterion."""
    assert (
        annotation_name in _ANNOT_CLASS_NAME_TO_KEYS_CONTAINER_TYPE_MAP
    ), f"unrecognized annotation (criterion) name: {annotation_name}"
    container_type = _ANNOT_CLASS_NAME_TO_KEYS_CONTAINER_TYPE_MAP[annotation_name]
    keys_container = container_type(basic_keys_only=basic_keys_only)
    return keys_container


def get_annotation_last_update_key(
    annotation_name: "CriteriaNameType",
) -> str:
    """Returns the tensor name key associated with an annotation's last update timestamp."""
    return create_annotation_keys_container(annotation_name, True)._last_update  # noqa


def get_potentially_annotated_flag_key(
    annotation_or_meta_name: typing.Union["CriteriaNameType", str],
) -> str:
    """Returns the tensor name key associated with the binary flag for an annotation (or group)."""
    if annotation_or_meta_name == ANNOT_APPROVSIGNC1_META_CLASS_NAME:
        return _ANNOT_APPROVSIGNC1_VALID
    elif annotation_or_meta_name == ANNOT_C2C3C4C5C6_META_CLASS_NAME:
        return _ANNOT_C2C3C4C5C6_VALID
    assert annotation_or_meta_name in ANNOT_CLASS_NAMES
    meta_class_name = ANNOT_CLASS_NAME_TO_META_CLASS_NAME[annotation_or_meta_name]
    return get_potentially_annotated_flag_key(meta_class_name)
