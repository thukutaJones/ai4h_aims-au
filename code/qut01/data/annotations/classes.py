"""Contains defines, dataclasses, and utility functions to parse and preprocess annotations.

For more information on the annotations specifications, refer to the master document:
https://docs.google.com/document/d/1e1tuyrGmKz27xwqshc5aMuI0YJZqEntQ

TODO: add description snippets used for LLM experiments with each criterion here
"""
import collections
import datetime
import enum
import pathlib
import pickle
import re
import typing
import warnings

import numpy as np

from qut01.data.annotations.keys import (
    LABEL_SUFFIXES_TO_CONVERT_TO_BOOL,
    NO_LABELS,
    OPT1_LABELS,
    OPT2_LABELS,
    OPT3_LABELS,
    UNCLEAR_LABELS,
    YES_LABELS,
    create_annotation_keys_container,
    get_annotation_last_update_key,
    get_potentially_annotated_flag_key,
)
from qut01.data.classif_utils import (
    ANNOT_APPROVAL_CLASS_NAME,
    ANNOT_C1_REPENT_CLASS_NAME,
    ANNOT_C2_OPS_CLASS_NAME,
    ANNOT_C2_STRUCT_CLASS_NAME,
    ANNOT_C2_SUPPCH_CLASS_NAME,
    ANNOT_C3_RISK_CLASS_NAME,
    ANNOT_C4_MITIG_CLASS_NAME,
    ANNOT_C4_REMED_CLASS_NAME,
    ANNOT_C5_EFFECT_CLASS_NAME,
    ANNOT_C6_CONSULT_CLASS_NAME,
    ANNOT_CLASS_NAME_TO_META_CLASS_NAME,
    ANNOT_SIGNATURE_CLASS_NAME,
)
from qut01.data.preprocess_utils import (
    annotator_separator_pattern,
    annotator_separator_token,
)
from qut01.utils.logging import get_logger

if typing.TYPE_CHECKING:
    from qut01.data.annotations.chunks import AnnotationTextChunk
    from qut01.data.annotations.keys import AnnotationKeysBase
    from qut01.data.classif_utils import CriteriaNameType
    from qut01.data.dataset_parser import DataParser
    from qut01.data.statement_utils import StatementProcessedData

logger = get_logger(__name__)


empty_space_cleaner_pattern = re.compile(r"^\s+$")


class AnnotationLabel(enum.Enum):
    """Definitions to use for (post-processed) yes/no/unclear labels."""

    YES = enum.auto()  # 'yes', the statement contains the required information
    NO = enum.auto()  # 'no', the statement does not contain the required information
    UNCLEAR = enum.auto()  # 'unclear' (with a mandatory note justifying why it is unclear)

    @staticmethod
    def convert_from_str(val: str) -> typing.Optional["AnnotationLabel"]:
        """Converts a string-based (raw) annotation label to its corresponding enum."""
        if val in YES_LABELS:
            return AnnotationLabel.YES
        if val in NO_LABELS:
            return AnnotationLabel.NO
        if val in UNCLEAR_LABELS:
            return AnnotationLabel.UNCLEAR
        return None


class JointOptionLabel(enum.Enum):
    """Definitions to use for (post-processed) joint option labels."""

    OPT1 = enum.auto()  # the reporting entity(ies) all approved the statement
    OPT2 = enum.auto()  # a higher entity that owns the reporting entity approved the statement
    OPT3 = enum.auto()  # at least one reporting entity approved the statement (+ mandatory explanation)

    @staticmethod
    def convert_from_str(val: str) -> typing.Optional["JointOptionLabel"]:
        """Converts a string-based (raw) annotation label to its corresponding enum."""
        if val in OPT1_LABELS:
            return JointOptionLabel.OPT1
        if val in OPT2_LABELS:
            return JointOptionLabel.OPT2
        if val in OPT3_LABELS:
            return JointOptionLabel.OPT3
        return None


class AnnotationsBase:
    """Base class used for the processing and storage of all raw annotation data.

    Note that this class will contain many attributes that are dynamically assigned based on the
    "keys" that are provided at construction by the derived class. Those keys point to the names
    of relevant columns and values in the original CSV dataset.

    All annotations have some things in common: supporting text, visual elements (maybe), and an
    assigned evidence label, which can be "YES", "NO", or "UNCLEAR".
    """

    name: "CriteriaNameType"
    """The name of the annotation class (defined in derived classes).

    This name is tied to the CRITERION associated with this annotation object, and is used to
    determine whether the annotation is relevant for different classification task setups.
    """

    _keys: "AnnotationKeysBase"
    """The tensor keys associated with this annotation object, under which to find labels.

    Should be unique for each derived class (i.e. for each criterion type), as specific criterion
    might have different labels beyond the yes/no/unclear ones.
    """

    def __init__(
        self,
        tensor_data: typing.Dict[str, any],  # statement data, loaded from repackaged dataset
        annotation_index: int = 0,  # load values from the n-th annotated elements (if multi-annot)
        allow_missing_labels: bool = False,  # if false, will throw if an expected label is missing
    ):
        """Stores the keys used to access the annotated data, and preprocessed that data."""
        assert self._keys.name == self.name  # noqa (should both be set in derived class)
        self.statement_id: int = None  # noqa (will be set in the `_set_annot_values` func below)
        self.label: AnnotationLabel = None  # noqa (will be set in the `_set_annot_values` func below)
        self._yes_text: typing.Optional[str] = None  # supporting text for the 'yes' evidence label
        self._no_text: typing.Optional[str] = None  # supporting text for the 'no' evidence label
        self._unclear_text: typing.Optional[str] = None  # supporting/justification text for unclear label
        self.annotation_index = annotation_index
        self.allow_missing_labels = allow_missing_labels
        # note: the 'chunks' list below is assigned later when the full statement data is available
        # (see the `StatementProcessedData.create()` function for more info)
        self.chunks: typing.Optional[typing.List["AnnotationTextChunk"]] = None
        self.last_update: typing.Optional[datetime.datetime] = None  # none = not validated yet
        self._set_annot_values(tensor_data)
        assert isinstance(self.statement_id, int), f"unexpected statement id format: {self.statement_id}"

    @property
    def meta_name(self) -> str:
        """Returns the 'meta' annotation class name for this type of annotation.

        This 'meta' identifier is used to determine the phase of annotation in which the data in
        this object was generated. This can help determine what other annotations may have been
        created at the same time, and which ones may be missing.
        """
        return ANNOT_CLASS_NAME_TO_META_CLASS_NAME[self.name]

    @staticmethod
    def _get_item(
        val: typing.Union[str, np.ndarray],
        idx: int,
    ) -> str:
        """Returns the annotated value from the given value object, converting it as necessary."""
        if isinstance(val, np.ndarray):
            val = val[idx].item()
        else:
            assert idx == 0, "unexpected non-zero index provided for a scalar annotation value?"
        assert isinstance(val, str), f"unexpected value type: {type(val)}"
        return val

    def _set_annot_values(
        self,
        tensor_data: typing.Dict[str, any],
    ) -> None:
        """Sets internal annotation values, converting them as needed from their raw format."""
        self._orig_label = self._get_item(
            tensor_data[self._keys.label],
            self.annotation_index,
        )  # to re-access it later
        self.statement_id = tensor_data["statement_id"]
        assert isinstance(self.statement_id, int)
        for key in self.keys():
            dataset_key = getattr(self._keys, key)
            if not self.allow_missing_labels:
                assert dataset_key in tensor_data, f"missing '{dataset_key}' from loaded statement data dictionary"
            elif dataset_key not in tensor_data:
                continue  # no tensor available with the specified key, skip this label
            val = self._get_item(tensor_data[dataset_key], self.annotation_index)
            if key == "label":
                val = AnnotationLabel.convert_from_str(val)
            else:
                assert isinstance(val, str), f"unexpected type for '{dataset_key}':\n\t{type(val)=}"
                must_convert_to_bool = any([dataset_key.endswith(s) for s in LABEL_SUFFIXES_TO_CONVERT_TO_BOOL])
                if must_convert_to_bool:
                    val = val.strip()
                    if val in YES_LABELS:
                        val = True
                    elif val in NO_LABELS or val == "":
                        val = False
                    else:
                        raise ValueError(f"unexpected non-bool value for '{dataset_key}':\n\t{val=}")
                elif isinstance(val, str):
                    # if there are strings with only whitespace, consider them empty
                    val = empty_space_cleaner_pattern.sub("", val)
            if key in ["yes_text", "no_text", "unclear_text"]:
                key = f"_{key}"
            setattr(self, key, val)
        last_update_key = get_annotation_last_update_key(self.name)
        self.last_update = None
        if last_update_key in tensor_data:
            last_update_str = self._get_item(tensor_data[last_update_key], idx=0)
            if last_update_str:
                self.last_update = datetime.datetime.fromisoformat(last_update_str)

    def keys(self) -> typing.List[str]:
        """Returns all keys that should be found for this annotation in the loaded statement."""
        return self._keys.keys()

    def items(self) -> typing.List[typing.Tuple[str, any]]:
        """Returns a map of all annotations for this annotation in the loaded statement data."""
        return [(key, getattr(self, key)) for key in self.keys()]

    def get_error_msg(self) -> typing.Optional[str]:
        """If this object is invalid, returns a message specifying why; otherwise, returns None."""
        label = self.label
        if label is None:
            return f"annotation label '{self._orig_label}' could not be converted to yes/no/unclear"
        yes_text, no_text, unclear_text = self._yes_text, self._no_text, self._unclear_text  # noqa
        supporting_text_count = sum([bool(t) for t in [yes_text, no_text, unclear_text]])
        has_no_support_info = self.has_no_support_info  # noqa
        if label == AnnotationLabel.YES and has_no_support_info:
            return "provided label as 'yes' with contradictory flag for no supporting info"
        if has_no_support_info:
            if label != AnnotationLabel.NO:
                return f"provided flag for no supporting info with {label.name} label"
        if supporting_text_count > 0 and not self.supporting_text:
            return (
                "got supporting text that does not match the specified yes/no/unclear label:"
                f"\n\tlabel={label.name}"
                f"\n\t{yes_text=}"
                f"\n\t{no_text=}"
                f"\n\t{unclear_text=}"
            )
        if supporting_text_count > 1:
            return (
                "got more than one supporting text specified for the yes/no/unclear label:"
                f"\n\tlabel={label.name}"
                f"\n\t{yes_text=}"
                f"\n\t{no_text=}"
                f"\n\t{unclear_text=}"
            )
        if hasattr(self, "has_date") and self.has_date:
            if not self.date_text and not self.date_text_is_scanned:  # noqa
                return "found a date without providing it as text or flagging it as scanned"
        else:
            has_text = hasattr(self, "date_text") and self.date_text
            has_scanned_flag = hasattr(self, "date_text_is_scanned") and self.date_text_is_scanned
            if has_text or has_scanned_flag:
                return "specified date text or scanned flag without specifying that a date was found"
        if hasattr(self, "viz_elem_label") and self.viz_elem_label:
            if not self.viz_elem_text and not self.viz_elem_scanned:  # noqa
                return "found a visual element whose content was not scanned nor extracted"
            if not self.viz_elem_page:  # noqa
                return "found a visual element whose location was not specified"
            if not self.viz_elem_type:  # noqa
                return "found a visual element whose type was not specified"
        else:
            has_page = hasattr(self, "viz_elem_page") and self.viz_elem_page
            has_type = hasattr(self, "viz_elem_type") and self.viz_elem_type
            has_text = hasattr(self, "viz_elem_text") and self.viz_elem_text
            has_scanned_flag = hasattr(self, "viz_elem_scanned") and self.viz_elem_scanned
            if has_page or has_type or has_text or has_scanned_flag:
                return "specified viz elem info without specifying that a viz elem was found"
        if hasattr(self, "add_viz_elem_label") and self.add_viz_elem_label:
            if not self.add_viz_elem_text and not self.add_viz_elem_scanned:  # noqa
                return "found an additional visual element whose content was not scanned nor extracted"
            if not self.add_viz_elem_page:  # noqa
                return "found an additional visual element whose location was not specified"
            if not self.add_viz_elem_type:  # noqa
                return "found an additional visual element whose type was not specified"
        else:
            has_page = hasattr(self, "add_viz_elem_page") and self.add_viz_elem_page
            has_type = hasattr(self, "add_viz_elem_type") and self.add_viz_elem_type
            has_text = hasattr(self, "add_viz_elem_text") and self.add_viz_elem_text
            has_scanned_flag = hasattr(self, "add_viz_elem_scanned") and self.add_viz_elem_scanned
            if has_page or has_type or has_text or has_scanned_flag:
                return "specified additional viz elem info without specifying that a viz elem was found"
        return None

    @property
    def supporting_text(self) -> str:
        """Returns the supporting text for the corresponding label in the loaded statement data.

        Note: this does NOT include the text extracted from visual elements (if any).
        """
        if self.label == AnnotationLabel.YES:
            return self._yes_text
        if self.label == AnnotationLabel.NO:
            return self._no_text
        if self.label == AnnotationLabel.UNCLEAR:
            return self._unclear_text
        raise ValueError(f"invalid label value: {self._orig_label}")

    @property
    def viz_elem_supporting_text(self) -> str:
        """Returns the supporting text for the corresponding label extracted from visual elements.

        Note: this does NOT include the text extracted from the statement's main body. If no
        supporting text was extracted from any visual element, an empty string will be returned.
        Otherwise, the supporting text will also be split (based on the separator token) and
        each text chunk will be prefixed with `fig.pX.` and `tab.pX.` strings.
        """
        text_chunks_with_prefix = []
        if hasattr(self, "viz_elem_text") and self.viz_elem_text:
            text_chunks = re.split(annotator_separator_pattern, self.viz_elem_text)
            assert hasattr(self, "viz_elem_type") and self.viz_elem_type
            # note: we assume that the 'type' is uniform for all text, but it might not be...
            viz_elem_type = "tab" if "table" in self.viz_elem_type.lower() else "fig"  # noqa
            assert hasattr(self, "viz_elem_page") and self.viz_elem_page
            # note: the page might be badly formatted; if so, we substitute with "0" for now
            viz_elem_page = f"{int(self.viz_elem_page)}" if self.viz_elem_page.isdigit() else "0"  # noqa
            for text_chunk in text_chunks:
                text_chunks_with_prefix.append(f"{viz_elem_type}.p{viz_elem_page}.{text_chunk}")
        # do the same as above, but with the optional 'additional' visual element attribs
        if hasattr(self, "add_viz_elem_text") and self.add_viz_elem_text:
            text_chunks = re.split(annotator_separator_pattern, self.add_viz_elem_text)
            assert hasattr(self, "add_viz_elem_type") and self.add_viz_elem_type
            # note: we assume that the 'type' is uniform for all text, but it might not be...
            viz_elem_type = "tab" if "table" in self.add_viz_elem_type.lower() else "fig"  # noqa
            assert hasattr(self, "add_viz_elem_page") and self.add_viz_elem_page
            # note: the page might be badly formatted; if so, we substitute with "0" for now
            viz_elem_page = f"{int(self.viz_elem_page)}" if self.viz_elem_page.isdigit() else "0"  # noqa
            for text_chunk in text_chunks:
                text_chunks_with_prefix.append(f"{viz_elem_type}.p{viz_elem_page}.{text_chunk}")
        # finally, reassemble all the chunks with the separator in between
        return annotator_separator_token.join([c for c in text_chunks_with_prefix])

    @property
    def is_validated(self) -> bool:
        """Returns whether this annotation was manually validated or not."""
        assert self.last_update is None or isinstance(self.last_update, datetime.datetime)
        return self.last_update is not None

    def create_tensor_data(self) -> typing.Dict[str, any]:
        """Returns a tensor data dictionary for the annotation object data.

        Useful for example when storing updated annotations back into a tensor dataset. Note that
        this function does not manage multi-annotation arrays at all, and simply returns one string
        per annotated key.
        """
        tensor_data = dict()
        for key in self.keys():
            dataset_key = getattr(self._keys, key)
            if key in ["yes_text", "no_text", "unclear_text"]:
                if (
                    (key == "yes_text" and self.label == AnnotationLabel.YES)
                    or (key == "no_text" and self.label == AnnotationLabel.NO)
                    or (key == "unclear_text" and self.label == AnnotationLabel.UNCLEAR)
                ):
                    tensor_data[dataset_key] = self.supporting_text
                else:
                    tensor_data[dataset_key] = ""
            else:
                if not self.allow_missing_labels:
                    assert hasattr(self, key), f"missing label value for key '{key}'"
                elif not hasattr(self, key):
                    continue
                tensor_val = getattr(self, key)
                if isinstance(tensor_val, enum.Enum):
                    tensor_val = tensor_val.name
                tensor_data[dataset_key] = str(tensor_val)
        last_update = ""
        if self.last_update is not None:
            last_update = self.last_update.isoformat()
        last_update_key = get_annotation_last_update_key(self.name)
        tensor_data[last_update_key] = last_update
        # in case we want to write these tensors to the dataset, we'll also update the annotated flags
        annot_flag_key = get_potentially_annotated_flag_key(self.name)
        tensor_data[annot_flag_key] = True
        return tensor_data

    @classmethod
    def contains_annotation_data(
        cls,
        tensor_data: typing.Dict[str, any],  # statement data, loaded from repackaged dataset
    ) -> bool:
        """Returns whether the provided tensor data contains relevant annotation data."""
        # note: for simplicity, we only check that the tensor contains at least one valid label
        annot_keys = create_annotation_keys_container(cls.name)
        if annot_keys.label not in tensor_data:
            return False
        label_val = tensor_data[annot_keys.label]
        if isinstance(label_val, np.ndarray):
            label_val = label_val[0].item()
        assert isinstance(label_val, str)
        return len(label_val) > 0  # need to have anything but an empty string here


class Approval(AnnotationsBase):
    """Helper class to extract approval annotations from a single statement data batch."""

    name = ANNOT_APPROVAL_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_APPROVAL_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""

    def __init__(
        self,
        tensor_data: typing.Dict[str, any],  # statement tensor data, loaded from repackaged dataset
        annotation_index: int = 0,  # load values from the n-th annotated elements (if multi-annot)
        allow_missing_labels: bool = False,  # if false, will throw if an expected label is missing
    ):
        super().__init__(
            tensor_data=tensor_data,
            annotation_index=annotation_index,
            allow_missing_labels=allow_missing_labels,
        )
        self._set_joint_opt_label()

    def _set_joint_opt_label(self) -> None:
        """Sets the joint approval option label to its corresponding enum value."""
        if self.allow_missing_labels and not hasattr(self, "joint_opt_label"):
            return
        joint_opt_label = self.joint_opt_label  # noqa
        self._orig_joint_opt_label = joint_opt_label
        self.joint_opt_label = JointOptionLabel.convert_from_str(joint_opt_label)

    def get_error_msg(self) -> typing.Optional[str]:
        """If this object is invalid, returns a message specifying why; otherwise, returns None."""
        base_err_msg = super().get_error_msg()
        if base_err_msg is not None:
            return base_err_msg
        if hasattr(self, "joint_opt_label"):
            if self.joint_opt_label != JointOptionLabel.OPT3 and self.opt_3_explanation:  # noqa
                return "extracted explanation for opt 3 when the specified label is not opt 3"
        return None


class Signature(AnnotationsBase):
    """Helper class to extract signature annotations from a single statement data batch."""

    name = ANNOT_SIGNATURE_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_SIGNATURE_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


class Criterion1ReportEnt(AnnotationsBase):
    """Helper class to extract c1 annotations from a single statement data batch."""

    name = ANNOT_C1_REPENT_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_C1_REPENT_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


class Criterion2Structure(AnnotationsBase):
    """Helper class to extract c2-structure annotations from a single statement data batch."""

    name = ANNOT_C2_STRUCT_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_C2_STRUCT_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


class Criterion2Operations(AnnotationsBase):
    """Helper class to extract c2-operations annotations from a single statement data batch."""

    name = ANNOT_C2_OPS_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_C2_OPS_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


class Criterion2SupplyChains(AnnotationsBase):
    """Helper class to extract c2-supplychains annotations from a single statement data batch."""

    name = ANNOT_C2_SUPPCH_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_C2_SUPPCH_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


class Criterion3RiskDesc(AnnotationsBase):
    """Helper class to extract c3-risk annotations from a single statement data batch."""

    name = ANNOT_C3_RISK_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_C3_RISK_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


class Criterion4Mitigation(AnnotationsBase):
    """Helper class to extract c4-mitigation annotations from a single statement data batch."""

    name = ANNOT_C4_MITIG_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_C4_MITIG_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


class Criterion4Remediation(AnnotationsBase):
    """Helper class to extract c4-remediation annotations from a single statement data batch."""

    name = ANNOT_C4_REMED_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_C4_REMED_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


class Criterion5Effect(AnnotationsBase):
    """Helper class to extract c5-effectiveness annotations from a single statement data batch."""

    name = ANNOT_C5_EFFECT_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_C5_EFFECT_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


class Criterion6Consult(AnnotationsBase):
    """Helper class to extract c5-effectiveness annotations from a single statement data batch."""

    name = ANNOT_C6_CONSULT_CLASS_NAME
    """The name of this annotation class (same as the corresponding criterion name)."""

    _keys = create_annotation_keys_container(ANNOT_C6_CONSULT_CLASS_NAME)
    """The tensor keys associated with this annotation object, under which to find labels."""


ANNOT_CLASS_NAME_TO_TYPE_MAP = {
    ANNOT_APPROVAL_CLASS_NAME: Approval,
    ANNOT_SIGNATURE_CLASS_NAME: Signature,
    ANNOT_C1_REPENT_CLASS_NAME: Criterion1ReportEnt,
    ANNOT_C2_STRUCT_CLASS_NAME: Criterion2Structure,
    ANNOT_C2_OPS_CLASS_NAME: Criterion2Operations,
    ANNOT_C2_SUPPCH_CLASS_NAME: Criterion2SupplyChains,
    ANNOT_C3_RISK_CLASS_NAME: Criterion3RiskDesc,
    ANNOT_C4_MITIG_CLASS_NAME: Criterion4Mitigation,
    ANNOT_C4_REMED_CLASS_NAME: Criterion4Remediation,
    ANNOT_C5_EFFECT_CLASS_NAME: Criterion5Effect,
    ANNOT_C6_CONSULT_CLASS_NAME: Criterion6Consult,
}

default_validated_data_dir_name = "validated_data"
"""Default name of the directory where validated statement data will be stored."""


def get_annotation_dump_dir_path(
    statement_id: int,
    pickle_dir_path: typing.Optional[pathlib.Path] = None,  # none = default framework path
) -> pathlib.Path:
    """Returns the directory where validated annotation dumps may be located for a statement."""
    import qut01.utils.config

    if pickle_dir_path is None:
        data_root_dir = qut01.utils.config.get_data_root_dir()
        pickle_dir_path = data_root_dir / default_validated_data_dir_name
    statement_pickle_dir_path = pickle_dir_path / str(statement_id)
    return statement_pickle_dir_path


def get_annotations_for_statement(
    statement_tensor_data: typing.Dict[str, any],
    target_annot_types: typing.Optional[typing.List[typing.Callable]] = None,  # none = all
    return_bad_annot_list: bool = False,
    pickle_dir_path: typing.Optional[pathlib.Path] = None,  # none = default framework path
    dump_found_validated_annots_as_pickles: bool = False,  # dump to pickles as backup, if needed
    load_validated_annots_from_pickles: bool = False,  # load from pickles (backups), if needed
) -> typing.Union[list, typing.Tuple[list, list]]:
    """Returns a list of annotation objects that can be used with the given statement data.

    Note that this function may return multiple annotation objects of the same type if the
    statement is annotated by multiple annotators. If `return_bad_annot_list`, a list of
    annotation names + reasons for failure will be returned.
    """
    assert isinstance(statement_tensor_data, dict)  # should be raw tensor data from dataset parser
    if not target_annot_types:
        target_annot_types = [
            Approval,
            Signature,
            Criterion1ReportEnt,
            Criterion2Structure,
            Criterion2Operations,
            Criterion2SupplyChains,
            Criterion3RiskDesc,
            Criterion4Mitigation,
            Criterion4Remediation,
            Criterion5Effect,
            Criterion6Consult,
        ]
    assert len(target_annot_types), "no target annotation types?"
    statement_id = statement_tensor_data["statement_id"]
    assert isinstance(statement_id, int), f"unexpected statement id format: {statement_id}"
    # convert annotation names into actual types, if needed
    for annot_idx, annot_type in enumerate(target_annot_types):
        if annot_type in ANNOT_CLASS_NAME_TO_TYPE_MAP:
            target_annot_types[annot_idx] = ANNOT_CLASS_NAME_TO_TYPE_MAP[annot_type]
    good_annotations, bad_annotations = [], []
    annot_dump_dir_path = get_annotation_dump_dir_path(
        statement_id=statement_id,
        pickle_dir_path=pickle_dir_path,
    )
    for annot_type in target_annot_types:
        annot_meta_name = ANNOT_CLASS_NAME_TO_META_CLASS_NAME[annot_type.name]
        potential_annot_count = statement_tensor_data["potential_annotation_count"][annot_meta_name]
        good_so_far_annotations, found_bad_annotation = [], False
        expected_valid_annot_dump_path = annot_dump_dir_path / (annot_type.name + ".pkl")
        try:
            if load_validated_annots_from_pickles and expected_valid_annot_dump_path.exists():
                # skip the original annotator data, keep the validated data only
                with open(expected_valid_annot_dump_path, "rb") as fd:
                    annot = pickle.load(fd)
                assert annot.name == annot_type.name, "unexpected bad annotation type?"
                assert annot.last_update is not None, "is this not validated data?"
                err_msg = annot.get_error_msg()
                assert err_msg is None, "unexpected invalid 'validated' annotation?"
                good_so_far_annotations.append(annot)
            else:
                if not annot_type.contains_annotation_data(tensor_data=statement_tensor_data):
                    continue
                for annot_idx in range(potential_annot_count):
                    annot = annot_type(tensor_data=statement_tensor_data, annotation_index=annot_idx)
                    if not annot.is_validated:
                        err_msg = annot.get_error_msg()
                        if err_msg is not None:  # annotation is invalid
                            found_bad_annotation = f"invalid data, {err_msg}"
                            bad_annotations.append((annot_type.name, found_bad_annotation))
                            continue
                    good_so_far_annotations.append(annot)
                    if annot.last_update is not None:
                        # this is a VALIDATED ANNOTATION; it should be the only one!
                        # (we currently do not handle having more than one validated annotation)
                        assert len(good_so_far_annotations) == 1
                        if dump_found_validated_annots_as_pickles:
                            annot_dump_dir_path.mkdir(parents=True, exist_ok=True)
                            with open(expected_valid_annot_dump_path, "wb") as fd:
                                pickle.dump(annot, fd)
                        # note: we need to "break" out of the loop here to discard any validated
                        #       annotation copies (see the `update_tensors` function in the
                        #       dataset parser implementation for more details)
                        break
            # we drop all annots of the same type for a statement if a single one is bad
            if not found_bad_annotation:
                good_annotations.extend(good_so_far_annotations)
        except (ValueError, AssertionError) as e:
            bad_annot_reason = f"caught exception during creation, {e}"
            bad_annotations.append((annot_type.name, bad_annot_reason))
    if return_bad_annot_list:
        return good_annotations, bad_annotations
    return good_annotations


def get_annotations(
    dataset: "DataParser",
    target_statement_ids: typing.Optional[typing.Iterable[int]] = None,  # none = all
    target_annot_types: typing.Optional[typing.List[typing.Callable]] = None,  # none = all
    return_bad_annot_list: bool = False,
    pickle_dir_path: typing.Optional[pathlib.Path] = None,  # none = default framework path
) -> typing.Union[list, typing.Tuple[list, list]]:
    """Returns a list of annotation objects that can be used with the statement dataset.

    If no target statement identifiers are given, the entire dataset will be parsed for annotations.

    If `return_bad_annot_list`, a list of annotation sids + names + reasons for failure will
    be returned.
    """
    good_annotations = []  # list of annotation objects which contain the annot type + statement id
    bad_annotations = []  # tuples of (statement_id, annotation type, reason for failure)
    tot_annotation_count = collections.defaultdict(int)  # total of good+bad by annot type
    if not target_statement_ids:
        potentially_annotated_statement_ids = dataset.get_potentially_annotated_statement_ids()
        target_statement_ids = {sid for sids in potentially_annotated_statement_ids.values() for sid in sids}
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="Indexing by integer in a for loop, like*",
            category=UserWarning,
        )  # disables annoying deeplake warning when indexing over the metadata
        potential_ids = dataset.info["failed_statement_ids"] + dataset.statement_ids
        for statement_id in target_statement_ids:
            assert statement_id in potential_ids, f"invalid statement id: {statement_id} (not found in dataset)"
            if statement_id in dataset.info["failed_statement_ids"]:
                logger.warning(f"skipping annotations for statement id: {statement_id} (data export failed)")
                continue
            statement_idx = dataset.statement_ids.index(statement_id)
            statement_tensor_data = dataset.get_tensor_data(statement_idx, meta_only=True)
            good_annots, bad_annots = get_annotations_for_statement(
                statement_tensor_data=statement_tensor_data,
                target_annot_types=target_annot_types,
                return_bad_annot_list=True,
                pickle_dir_path=pickle_dir_path,
            )
            for good_annot in good_annots:
                tot_annotation_count[good_annot.name] += 1
                good_annotations.append(good_annot)
            for bad_annot_name, bad_annot_reason in bad_annots:
                tot_annotation_count[bad_annot_name] += 1
                logger.warning(
                    f"\nrejecting {bad_annot_name} annotation"
                    f" for {statement_tensor_data['batch_id']}"
                    f" at {statement_tensor_data['metadata/Link']}"
                    f"\nreason: {bad_annot_reason}\n"
                )
                bad_annotations.append((statement_id, bad_annot_name, bad_annot_reason))
    logger.info(f"parsed {len(good_annotations)} valid annotations for {len(target_statement_ids)} statements")
    for annot_name in tot_annotation_count.keys():
        valid_count = sum([a.name == annot_name for a in good_annotations])
        logger.info(
            f"valid {annot_name} annotations: {valid_count} "
            f"({valid_count / tot_annotation_count[annot_name]:.1%} of total, "
            f"{valid_count / len(target_statement_ids):.2} per statement)"
        )
        invalid_count = sum([a[1] == annot_name for a in bad_annotations])
        logger.info(
            f"invalid {annot_name} annotations: {invalid_count} "
            f"({invalid_count / tot_annotation_count[annot_name]:.1%} of total, "
            f"{invalid_count / len(target_statement_ids):.2} per statement)"
        )
    if return_bad_annot_list:
        return good_annotations, bad_annotations
    return good_annotations


class ValidatedAnnotation(AnnotationsBase):
    """Helper class to extract validated annotation labels across all possible criteria.

    Will initialize the "basic" attribs (label + supporting text) with merged prior annotations, if
    any exist. Note that this implementation totally ignores criteria-specific labels (e.g. the
    option label for approval) as well as all labels related to dates, scanned content, or visual
    elements.

    Merging prior annotations is done by setting the label as "YES" if any annotator previously
    specified it as such for the targeted statement; then, all supporting text chunks from prior
    annotations with the correct label are merged into the supporting text.

    TODO: update this class if we need to validate annotations beyond label and supporting text.
          Currently, there is no way to tell whether all the supporting information for a criteria
          is contained only in "embedded" visual elements, meaning that a "YES" label will be
          considered invalid as it does not have supporting text in these validated annotations...
          (we avoid this issue by just not checking for errors in validated annotations)
    """

    def __init__(
        self,
        annotation_name: "CriteriaNameType",
        statement_data: "StatementProcessedData",
    ):
        self.name = annotation_name
        self._keys = create_annotation_keys_container(
            annotation_name=annotation_name,
            basic_keys_only=True,
        )
        self.chunks_to_be_added: typing.List[str] = []  # during validation, this is where we'll add new chunks
        super().__init__(
            tensor_data=statement_data,  # noqa (we cheat and use the processed data directly below)
            allow_missing_labels=True,  # the generic tensor dict above cannot handle all annot types
        )
        # if chunks exist (inherited from prior annotations), we reassign their old object references
        for chunk in self.chunks:
            chunk.annotation = self
            chunk.statement_data = statement_data

    def _set_annot_values(
        self,
        tensor_data: "StatementProcessedData",
    ) -> None:
        """Sets internal annotation values, merging from existing values if possible."""
        self.statement_id = tensor_data.id
        assert isinstance(self.statement_id, int)
        self.chunks = []
        chunk_texts = []
        relevant_prior_annots = [a for a in tensor_data.annotations if a.name == self.name]
        if not relevant_prior_annots:
            self.label = AnnotationLabel.UNCLEAR  # default in case we have not seen any annotations at all yet
        else:
            any_labeled_as_yes = any([a.label == AnnotationLabel.YES for a in relevant_prior_annots])
            any_labeled_as_no = any([a.label == AnnotationLabel.NO for a in relevant_prior_annots])
            if any_labeled_as_yes:
                self.label = AnnotationLabel.YES
            elif any_labeled_as_no:
                self.label = AnnotationLabel.NO
            else:
                self.label = AnnotationLabel.UNCLEAR
            for sentence_idx, sentence_annot_chunks in enumerate(tensor_data.sentence_annotation_chunks):
                for chunk in sentence_annot_chunks:
                    if chunk.annotation.name == self.name and chunk.annotation.label == self.label:
                        if chunk.chunk_text not in chunk_texts:
                            self.chunks.append(chunk)
                            chunk_texts.append(chunk.chunk_text)

    @property
    def supporting_text(self) -> str:
        """Merges all the supporting text chunks together as a contiguous string.

        Note: this is the attribute that is used to generate actual `AnnotationTextChunk` objects
        later; this means that all new/modified/validated chunks have to be assembled here.
        """
        all_chunks = [*[c.chunk_text for c in self.chunks], *self.chunks_to_be_added]
        return annotator_separator_token.join([c for c in all_chunks if c])

    @property
    def has_no_support_info(self) -> bool:
        """Returns whether this annotation possesses any supporting info (text) for its label."""
        return not self.supporting_text

    def get_error_msg(self) -> typing.Optional[str]:
        """Note: we cannot really check for errors for partial (and manually-typed) labels."""
        # TODO: should we implement this? (might help warn user if they generate bad 'valid' data)
        return None
