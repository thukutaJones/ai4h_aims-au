"""Annotation validation tool.

NOTE: as of April 2024, this implementation does NOT allow you to annotate figures, tables,
or any other extra labels beyond the basic ones (yes/no/unclear) with supporting text. Instead,
to get supporting text to be considered as originating from a figure or table, you should
prefix your copy-pastes with `fig.pX.` or `tab.pX.`, where `pX` would be replaced by the (real)
page number where you found the figure/table. There is no way to specify that the information
was found in a figure or table that is entirely scanned/bitmap however.

For more information on the actions that can be performed using this tool, launch it, and then
use the "help" command ('?', 'h', or 'help') to display action information across different menus.
"""

import datetime
import itertools
import pathlib
import pickle
import tempfile
import textwrap
import typing

import numpy as np
import tqdm

import qut01

if typing.TYPE_CHECKING:
    from qut01.data.annotations.chunks import AnnotationTextChunk
    from qut01.data.annotations.classes import AnnotationsBase, ValidatedAnnotation
    from qut01.data.classif_utils import CriteriaNameType
    from qut01.data.statement_utils import StatementProcessedData

np.random.seed(0)
default_max_line_length: int = 140
default_max_chunk_lines_per_annot: int = 8
default_sentence_join_token = ". "
default_sentence_end_token = "."

shortcuts_help = ["h", "help", "?"]
shortcuts_quit = ["q", "quit", "exit"]
shortcuts_info = ["i", "info"]
shortcuts_all = ["a", "any", "all", "*"]
shortcuts_gold_set = ["g", "gs", "gold", "gold-set"]
shortcuts_next = ["n", "next", "ENTER"]
shortcuts_previous = ["p", "previous"]
shortcuts_discard = ["d", "drop", "discard"]
shortcuts_add = ["a", "add", "add-chunk"]
shortcuts_select = ["s", "select", "select-sentence"]
shortcuts_write_and_next = ["w", "write-and-next"]
shortcuts_skip_and_next = ["s", "skip-and-next"]
shortcuts_assign_label = ["l", "label", "assign-label"]
shortcuts_check_chunks = ["c", "check", "check-chunks"]
shortcuts_add_chunks = ["a", "add", "add-chunk"]
shortcuts_all_valid = ["all-ok", "all-valid"]

possible_validation_actions = [  # (shortcuts, description)
    (shortcuts_help, "Display information about potential validation actions."),
    (shortcuts_quit, "Quit the validation app (will ask whether to save changes, if needed)."),
    (shortcuts_info, "Display information about the statement undergoing validation."),
    (shortcuts_write_and_next, "Write validated annotation data and continue to the next statement."),
    (shortcuts_skip_and_next, "Skip validation for this statement and continue to the next statement."),
    (shortcuts_assign_label, "Assign a new label for a specific criterion for this statement."),
    (shortcuts_check_chunks, "Review chunks that support a criterion for this statement."),
    (shortcuts_add_chunks, "Add a new chunk that supports a criterion for this statement."),
    (shortcuts_all_valid, "Sets ALL EXISTING ANNOTATIONS as validated."),
]

possible_chunk_review_actions = [  # (shortcuts, description)
    (shortcuts_help, "Display information about chunk review actions."),
    (shortcuts_quit, "Quit the chunk review for the current annotation (finalizing discards)."),
    (shortcuts_discard, "Toggles whether we should discard the currently displayed chunk."),
    (shortcuts_next, "Skip the current chunk and display the next one."),
    (shortcuts_previous, "Skip the current chunk and display the previous one."),
    ("[1, 2, 3, ...]", "Skip the current chunk and display the one specified by its 1-based index."),
    (shortcuts_all, "Sets ALL chunks in the current statement to be discarded."),
]

possible_chunk_add_actions = [  # (shortcuts, description)
    (shortcuts_help, "Display information about chunk creation actions."),
    (shortcuts_quit, "Quit the chunk review for the current annotation (finalizing discards)."),
    (shortcuts_add, "Save a manually provided text string as a new chunk."),
    (shortcuts_select, "Save one or more contiguous sentences from the statement as a new chunk."),
]

annot_class_codes = {  # used to represent class names in very dense settings (e.g. status bar)
    qut01.data.classif_utils.ANNOT_APPROVAL_CLASS_NAME: "approval",
    qut01.data.classif_utils.ANNOT_SIGNATURE_CLASS_NAME: "signature",
    qut01.data.classif_utils.ANNOT_C1_REPENT_CLASS_NAME: "c1-report",
    qut01.data.classif_utils.ANNOT_C2_STRUCT_CLASS_NAME: "c2-struct",
    qut01.data.classif_utils.ANNOT_C2_OPS_CLASS_NAME: "c2-ops",
    qut01.data.classif_utils.ANNOT_C2_SUPPCH_CLASS_NAME: "c2-supply",
    qut01.data.classif_utils.ANNOT_C3_RISK_CLASS_NAME: "c3-risk",
    qut01.data.classif_utils.ANNOT_C4_MITIG_CLASS_NAME: "c4-mitig",
    qut01.data.classif_utils.ANNOT_C4_REMED_CLASS_NAME: "c4-remed",
    qut01.data.classif_utils.ANNOT_C5_EFFECT_CLASS_NAME: "c5-effect",
    qut01.data.classif_utils.ANNOT_C6_CONSULT_CLASS_NAME: "c6-consult",
}
annot_code_classes = {v: k for k, v in annot_class_codes.items()}  # reverse map of above


def _get_sentence_offset_lines(
    sentence: str,
    max_line_length: int,
) -> typing.List[str]:
    """Returns the given sentence rewrapped to a new line length and with subseq lines offset.

    Used to display sentences inside a fixed-width terminal using proper line wrapping.
    """
    sentence_lines = textwrap.wrap(
        sentence.replace("\n", " ").replace("  ", " "),  # remove newlines/spaces
        max_line_length - 4,  # wrap sentences and add quad-space after 1st line
    )
    for idx in range(1, len(sentence_lines)):
        sentence_lines[idx] = "    " + sentence_lines[idx]
    return sentence_lines


def _get_matched_sentences_display(
    chunk: "AnnotationTextChunk",
    max_line_length: int,
) -> str:
    """Returns the side-by-side view of matched sentences (with scores)."""
    output_lines = []
    sentence_count = len(chunk.sentences)
    half_line_length = max_line_length // 2 - 2  # -2 to keep some middle buffer empty
    for sidx, (chunk_sentence, match_orig_idx) in enumerate(zip(chunk.sentences, chunk.matched_sentences_orig_idxs)):
        chunk_lines = _get_sentence_offset_lines(chunk_sentence, max_line_length=half_line_length)
        matched_sentence = chunk.statement_sentences[match_orig_idx]
        matched_lines = _get_sentence_offset_lines(matched_sentence, max_line_length=half_line_length)
        match_score = chunk.matched_sentences_scores[sidx]
        sentence_header = f"sentence {sidx + 1}/{sentence_count}, {match_score:.1%} match score"
        sentence_header = sentence_header.center(max_line_length, " ")
        output_lines.append("-" * max_line_length)
        output_lines.append(sentence_header)
        for cline, mline in itertools.zip_longest(chunk_lines, matched_lines, fillvalue=""):
            cline = cline.ljust(half_line_length, " ")
            output_lines.append(cline + "    " + mline)
    output_lines.append("-" * max_line_length)
    output_display = "\n".join(output_lines)
    return output_display


def _get_data_parser_for_validation(
    dataset_path: typing.Optional[pathlib.Path],  # none = default framework path
    restart_from_raw_annotations: bool,  # specifies whether to start validation from scratch
    pickle_dir_path: typing.Optional[pathlib.Path],  # none = default framework path
    load_validated_annots_from_pickles: bool = False,  # load from pickles (backups), if needed
) -> qut01.data.dataset_parser.DataParser:
    """Returns a data parser ready for statement annotation validation purposes.

    The data parser will open the dataset without using a read-only mode --- this will crash if the
    dataset is already opened elsewhere. Make sure you only run one validation script at a time!
    """
    dataset = qut01.data.dataset_parser.get_deeplake_dataset(
        dataset_path=dataset_path,
        checkout_branch=qut01.data.dataset_parser.dataset_annotated_branch_name,
        read_only=False,
    )
    qut01.data.dataset_parser.prepare_dataset_for_validation(
        dataset=dataset,
        restart_from_raw_annotations=restart_from_raw_annotations,
        bypass_user_confirmation=False,
    )
    data_parser = qut01.data.dataset_parser.DataParser(
        dataset,
        pickle_dir_path=pickle_dir_path,
        dump_found_validated_annots_as_pickles=True,
        load_validated_annots_from_pickles=load_validated_annots_from_pickles,
        use_processed_data_cache=False,
    )
    return data_parser


def _get_annot_names_based_on_choice() -> typing.List["CriteriaNameType"]:
    """Returns a list of annotation names that should be validated based on the user's choice."""
    annot_class_names = qut01.data.classif_utils.ANNOT_CLASS_NAMES
    potential_choices = {
        **{annot_class_codes[potential_choice]: [potential_choice] for potential_choice in annot_class_names},
        **qut01.data.classif_utils.ANNOT_META_CLASS_NAME_TO_CLASS_NAMES_MAP,  # meta annot groups
        "c2-c6": qut01.data.classif_utils.ANNOT_C2C3C4C5C6_CLASS_NAMES,  # shortcut for 2nd meta group
        **{k: annot_class_names for k in shortcuts_all},  # to select "all" annot types
    }
    while True:
        choice = None
        while not choice:
            choice = input("\nSpecify which ANNOTATION(s) to validate (comma-separated), or '?' for help:\n")
            choice = choice.strip().lower()
            if choice in shortcuts_help:
                print("\n\tTarget criteria can be specified via any of the following (alone or separated by commas):")
                for potential_choice in potential_choices:
                    print(f"\t\t{potential_choice}")
                choice = None  # reset choice, go back to asking for actual input
        if choice in shortcuts_quit:
            print("Goodbye!")
            exit(0)  # exit app directly (nothing in progress that needs to be saved)
        elif choice in potential_choices:
            target_annot_names = potential_choices[choice]
            break
        bad_annots, target_annot_names = [], []
        for picked_annot in choice.split(","):
            picked_annot = picked_annot.strip()
            if picked_annot in annot_code_classes:
                target_annot_names.extend(annot_code_classes[picked_annot])
            elif any([annot_name.startswith(picked_annot) for annot_name in annot_class_names]):
                matched_annots = [s for s in annot_class_names if s.startswith(picked_annot)]
                target_annot_names.extend(matched_annots)
            else:
                bad_annots.append(f"'{picked_annot}'  (invalid/unknown annotation name or index)")
        if bad_annots:
            invalid_str = "\n\t\t".join(bad_annots)
            print(f"\n\tSpecified {len(bad_annots)} invalid annotation(s):\n\t\t{invalid_str}")
        if bad_annots and target_annot_names:
            choice = input("\nContinue while ignoring invalid annotations? (y/N):\n")
            confirmed = choice.strip().lower() in ["y", "yes"]
            if not confirmed:
                continue
        if target_annot_names:
            break
        print("\tNo valid annotation specified, try again!")
    target_info_str = "".join([f"\n\t{annot_name}" for annot_name in target_annot_names])
    print(f"\nSpecified {len(target_annot_names)} target annotation(s):{target_info_str}")
    return target_annot_names


def _get_statement_ids_based_on_choice(
    data_parser: qut01.data.dataset_parser.DataParser,
    target_annot_names: typing.List["CriteriaNameType"],
    skip_already_validated_statements: bool,
) -> typing.List[int]:
    """Returns a list of statement ids that should be validated based on the user's choice."""
    dataset_sids = data_parser.statement_ids
    dataset_failed_sids = data_parser.dataset.info.failed_statement_ids
    while True:
        choice = None
        while not choice:
            choice = input("\nSpecify which STATEMENT ID(s) to validate (comma-separated), or '?' for help:\n")
            choice = choice.strip().lower()
            if choice in shortcuts_help:
                print(
                    "\n\tYou must specify the identifier (ID) for the statement(s) you wish to validate.\n"
                    "\n\tIf more than one statement is to be validated, separate their IDs by commas.\n"
                    "\tThe IDs are the ones used on the modern slavery register; for example, in\n"
                    "\t\thttps://modernslaveryregister.gov.au/statements/XXXX/\n"
                    "\t...the statement identifier would be XXXX, which should be an integer.\n"
                    "\n\tYou may also specify 'gold' to go through statements reserved for the gold set."
                )
                choice = None  # reset choice, go back to asking for actual input
        if choice in shortcuts_quit:
            print("Goodbye!")
            exit(0)  # exit app directly (nothing in progress that needs to be saved)
        elif choice in shortcuts_all:
            target_statement_ids = dataset_sids
        elif choice in shortcuts_gold_set:
            print("\nFetching reserved gold set statement identifiers...")
            reserved_gold_sid_clusters = qut01.data.split_utils.get_reserved_gold_id_clusters(data_parser)
            # note: we sort the sids so that we see one statement per cluster until they are all done
            reserved_gold_sid_clusters = [list(c) for c in reserved_gold_sid_clusters]
            target_statement_ids = []
            while any([cluster for cluster in reserved_gold_sid_clusters]):
                for cluster in reserved_gold_sid_clusters:
                    if cluster:
                        target_statement_ids.append(cluster.pop(0))
            print("\nNote: statements will be sorted so that we validate one at a time per cluster")
            if not skip_already_validated_statements:
                choice = input("\nSkip statements that are already fully validated? (y/N):\n")
                confirmed = choice.strip().lower() in ["y", "yes"]
                if confirmed:
                    skip_already_validated_statements = True
        else:
            bad_statements, target_statement_ids = [], []
            for picked_id in choice.split(","):
                picked_id = picked_id.strip()
                if picked_id.isdigit() and int(picked_id) in dataset_sids:
                    target_statement_ids.append(int(picked_id))
                elif picked_id.isdigit() and int(picked_id) in dataset_failed_sids:
                    bad_statements.append(f"'{picked_id}'  (PDF text export failed)")
                else:
                    bad_statements.append(f"'{picked_id}'  (invalid/unknown statement id)")
            if bad_statements:
                invalid_str = "\n\t\t".join(bad_statements)
                print(f"\n\tSpecified {len(bad_statements)} invalid statement(s):\n\t\t{invalid_str}")
            if bad_statements and target_statement_ids:
                choice = input("\nContinue while ignoring invalid statements? (y/N):\n")
                confirmed = choice.strip().lower() in ["y", "yes"]
                if not confirmed:
                    continue
        if not target_statement_ids:
            print("\tNo valid statement identifier specified, try again!")
            continue
        print(flush=True)
        target_info_str, output_statement_ids = "", []
        for sid in tqdm.tqdm(target_statement_ids, desc="Parsing target statements"):
            # note: this will also load pre-existing validated annotations, if any exist
            data = data_parser.get_processed_data(dataset_sids.index(sid))
            if not data.sentences:
                target_info_str += f"\n\t\t'{sid}' (ERROR: no sentences found, text extraction failed)"
                continue  # skip this statement, we could not parse any text from it
            annots = [a for a in data.annotations if a.name in target_annot_names]
            prior_valid_annots = [a for a in annots if a.is_validated]
            is_statement_fully_validated = (len(prior_valid_annots) == len(annots)) and {
                a.name for a in prior_valid_annots
            } == set(target_annot_names)
            skip_statement = skip_already_validated_statements and is_statement_fully_validated
            found_annot_names = {a.name for a in annots}
            target_info_str += (
                f"\n\t\t'{sid}' ({len(found_annot_names)}/{len(target_annot_names)} annotated target criteria, "
                f"{len(prior_valid_annots)} already validated, "
                f"{'skipped' if skip_statement else 'todo'})"
            )
            if skip_statement:
                continue  # skip this statement, it is already totally validated
            output_statement_ids.append(sid)
        print(f"\n\tWill process {len(output_statement_ids)} target statement(s):{target_info_str}", flush=True)
        if not output_statement_ids:
            print(
                "\tNo statement to be validated!"
                "\n\t\t(try `skip_already_validated_statements=False`, or pick other statements to validate)"
            )
            continue
        print()
        return output_statement_ids


def _save_validated_annotations(
    annotation: "ValidatedAnnotation",
    data_parser: qut01.data.dataset_parser.DataParser,
    pickle_dir_path: typing.Optional[pathlib.Path],  # none = default framework path
) -> bool:  # returns whether anything was saved
    """Saves the given statement annotations in the pickle dir path + dataset (if needed)."""
    if not annotation.is_validated:
        return False  # not yet validated, keep going (no need to save it)
    statement_annot_dump_dir_path = qut01.data.annotations.classes.get_annotation_dump_dir_path(
        statement_id=annotation.statement_id,  # noqa
        pickle_dir_path=pickle_dir_path,
    )
    statement_annot_dump_dir_path.mkdir(parents=True, exist_ok=True)
    pkl_path = statement_annot_dump_dir_path / f"{annotation.name}.pkl"
    if pkl_path.exists():
        # to make this function robust to interruptions, we'll rename the old file before writing over it
        # (we keep the "old" pickle around as a backup, in case we want to go back on an edit)
        tmp_pickle_path = pkl_path.with_suffix(".oldpkl")
        pkl_path.rename(tmp_pickle_path)
    print(f"\t\tSaving updated {annotation.name} data to: {pkl_path}")
    with open(pkl_path, "wb") as fd:
        pickle.dump(annotation, fd)
    print(f"\t\tSaving updated {annotation.name} data to '{data_parser.dataset.branch}' dataset branch")
    tensor_data = annotation.create_tensor_data()
    data_parser.update_tensors(annotation.statement_id, tensor_data)  # noqa
    return True


def _get_status_bar_str(
    statement_data: "StatementProcessedData",
    annotations: typing.Dict["CriteriaNameType", "ValidatedAnnotation"],
    max_line_length: int,  # for display purposes only
    annotations_updated: bool,  # specifies whether any of the annotations have been updated yet
) -> str:
    """Returns a status bar that displays the annotation metadata for the current statement.

    For each criterion, we will display its shortened (code) name with its label (YES/NO/UNC)
    underneath it, followed by its chunk count (where `Nc` = N chunks). This is meant to give a
    quick overview of the statement/annotation data without displaying all of it. For a more
    exhaustive view of the statement/annotation data, see the `_get_statement_info_str` function.
    """
    max_class_code_len = max([len(k) for k in annot_code_classes])
    needed_per_criterion_len = max_class_code_len + 2  # with spaces before/after
    assert needed_per_criterion_len * len(annotations) <= max_line_length, (
        f"max terminal line length {max_line_length} is too small to create annotation label bar "
        f"(increase it to at least {needed_per_criterion_len * len(annotations)})"
    )
    max_per_criterion_len = max_line_length // len(annotations)
    total_chunks = 0
    annot_codes_str, annot_labels_str = "", ""
    for annot_name, annot in annotations.items():
        annot_code = annot_class_codes[annot_name]
        annot_codes_str += annot_code.center(max_per_criterion_len)[:max_per_criterion_len]
        annot_label = f"{annot.label.name[:3]}; {len(annot.chunks)}c"
        annot_labels_str += annot_label.center(max_per_criterion_len)[:max_per_criterion_len]
        total_chunks += len(annot.chunks)
    assert len(annot_codes_str) <= max_line_length and len(annot_labels_str) <= max_line_length
    annot_codes_str = annot_codes_str.center(max_line_length)
    annot_labels_str = annot_labels_str.center(max_line_length)
    statement_info_str = (
        f"Processing statement {statement_data.id} "
        f"with {len(statement_data.sentences)} sentences and {total_chunks} supporting chunks "
        f"(updated: {'**YES**' if annotations_updated else 'no'})"
    )
    if len(statement_info_str) > max_line_length:
        # use the short statement info str instead
        statement_info_str = f"statement {statement_data.id} (updated={annotations_updated})"
    assert len(statement_info_str) <= max_line_length, (
        f"max terminal line length {max_line_length} is too small to create statement info bar "
        f"(increase it to at least {len(statement_info_str)})"
    )
    statement_info_str = statement_info_str.center(max_line_length)
    max_line_bar = "".join(["-"] * max_line_length)
    max_line_double_bar = "".join(["="] * max_line_length)
    status_bar_str = (
        "\n"
        f"{max_line_bar}\n"
        f"{statement_info_str}\n"
        f"{annot_codes_str}\n"
        f"{annot_labels_str}\n"
        f"{max_line_double_bar}"
    )
    return status_bar_str


def _get_chunk_lines_for_annot(
    annotation: "AnnotationsBase",
    tab_count: int,
    max_line_length: int,  # for display purposes only
    max_chunk_lines_per_annot: typing.Optional[int] = default_max_chunk_lines_per_annot,  # for display purposes only
    chunk_idx: typing.Optional[int] = None,  # if none, will display all chunks together using raw chunk text
) -> str:
    """Returns a string containing wrapped and tabbed lines for chunk text inside an annotation."""
    if chunk_idx is None:
        # generate lines from all chunks using the 'raw' (unprocessed) text
        chunk_lines = [
            line
            for chunk in annotation.chunks
            for line in _get_sentence_offset_lines(
                sentence=chunk.chunk_text,
                max_line_length=max_line_length - (8 * tab_count),  # we're offset by N tabs with 8 chars each
            )
        ]
    else:
        assert 0 <= chunk_idx < len(annotation.chunks)
        # generate lines from the processed chunk sentences
        chunk_lines = [
            line
            for sentence in annotation.chunks[chunk_idx].sentences
            for line in _get_sentence_offset_lines(
                sentence=sentence,
                max_line_length=max_line_length - (8 * tab_count),  # we're offset by N tabs with 8 chars each
            )
        ]
    chunks_str = ""
    tabs = "\t" * tab_count
    tot_chunk_lines = 0
    for line in chunk_lines:
        if max_chunk_lines_per_annot is not None and tot_chunk_lines >= max_chunk_lines_per_annot:
            chunks_str += f"\n{tabs}    ..."
            break
        chunks_str += f"\n{tabs}{line}"
        tot_chunk_lines += 1
    return chunks_str


def _get_statement_info_str(
    statement_data: "StatementProcessedData",
    target_annot_names: typing.List["CriteriaNameType"],
    max_line_length: int,  # for display purposes only
) -> str:
    """Returns a full printout of the statement+annotation data."""
    statement_info_str = "\tStatement information:"
    statement_info_str += f"\n\t\tid: {statement_data.id}"
    statement_info_str += f"\n\t\turl: {statement_data.url}"
    max_sentence_len = max([len(s) for s in statement_data.sentences])
    statement_info_str += (
        f"\n\t\tsentence count: {len(statement_data.sentences)} " f"(max sentence char count: {max_sentence_len})"
    )
    statement_info_str += f"\n\t\tpage count: {statement_data.page_count}"
    prior_annots = [a for a in statement_data.annotations if a.name in target_annot_names]
    if prior_annots:
        statement_info_str += "\n\t\tprior annotations:"
        for annot in prior_annots:
            statement_info_str += (
                f"\n\t\t\t{annot.name}: label={annot.label.name}, "
                f"{len(annot.chunks)} chunk(s), validated={annot.is_validated}"
            )
            chunks_str = _get_chunk_lines_for_annot(annot, tab_count=4, max_line_length=max_line_length)
            statement_info_str += chunks_str
    statement_info_str = f"\n{statement_info_str}"
    return statement_info_str


def _get_target_annot_for_update(
    annotations: typing.Dict["CriteriaNameType", "ValidatedAnnotation"],  # NEVER updated in this function
    max_line_length: int,  # for display purposes only
    action_name: str,  # for display purposes only
) -> typing.Optional[qut01.data.annotations.classes.ValidatedAnnotation]:
    """Returns a specific annotation to be updated (picked by the user), or None (to exit)."""
    annotations_info = ""
    annot_idx_map, annot_code_map = {}, {}
    for annot_idx, annot_name in enumerate(annotations):
        annot = annotations[annot_name]
        annot_idx_map[annot_idx + 1] = annot
        annot_code = annot_class_codes[annot_name]
        annot_code_map[annot_code] = annot
        annotations_info += (
            f"\n\t\t[{annot_idx + 1}]  {annot_code}: " f"label={annot.label.name}, {len(annot.chunks)} chunk(s)"
        )
        chunks_str = _get_chunk_lines_for_annot(annot, tab_count=3, max_line_length=max_line_length)
        annotations_info += chunks_str
    print(f"\n\nExisting annotation(s):\n{annotations_info}")
    if len(annotations) == 1:
        # there's no actual choice to make, just return the only possibility
        return next(iter(annotations.values()))
    while True:
        choice = input(f"\nSelect an annotation (by name/index) to {action_name}, or 'q' to leave:\n")
        choice = choice.strip().lower()
        if choice in shortcuts_help:
            print(
                "\n\tYou must specify the name of the annotation (or its index) to update it:"
                f"\n{annotations_info}\n"
                "\n\tTo abort and go back to the statement action selection, use 'q' or 'quit'."
            )
            continue
        elif choice in shortcuts_quit:
            return None  # break the annotation selection loop, go back to main menu
        elif choice in annotations:
            target_annot = annotations[choice]
        elif choice in annot_code_map:
            target_annot = annot_code_map[choice]
        elif choice.isdigit() and int(choice) in annot_idx_map:
            target_annot = annot_idx_map[int(choice)]
        else:
            if choice:
                print(f"\n\tInvalid annotation choice! ('{choice}')")
            continue
        return target_annot


def _assign_label(
    statement_data: "StatementProcessedData",
    annotations: typing.Dict["CriteriaNameType", "ValidatedAnnotation"],  # might have some labels updated
    max_line_length: int,  # for display purposes only
) -> bool:
    """Allows the user to review and potentially update labels from one or more annotations.

    If a label is updated inside an annotation object, that object's `last_update` attribute will
    also be modified. If any annotation is modified, this function will return `True`.
    """
    statement_id = statement_data.id
    updated_annots = set()
    yes_label_options = ["y", "ye", "yes", "1"]
    no_label_options = ["n", "no", "0"]
    unclear_label_options = ["u", "unc", "unclear"]
    supported_label_choices = [*yes_label_options, *no_label_options, *unclear_label_options]
    while True:  # outer loop to keep selecting annotations for which to update a label
        target_annot = _get_target_annot_for_update(
            annotations=annotations,
            max_line_length=max_line_length,
            action_name="UPDATE ITS LABEL",
        )
        if target_annot is None:
            break  # break the annotation selection loop, go back to main menu
        while True:  # inner loop to select the new label for the target annotation
            # TODO: if we want to support validation of all labels beyond basic ones, refactor here
            choice = None
            while not choice:
                choice = input(
                    f"\nCurrent label for {target_annot.name} is {target_annot.label.name}; select new label:\n"
                )
                choice = choice.strip().lower()
                if choice in shortcuts_help:
                    target_chunks_str = _get_chunk_lines_for_annot(
                        target_annot,
                        tab_count=3,
                        max_line_length=max_line_length,
                    )
                    label_options = [lbl for lbl in ["YES", "NO", "UNCLEAR"] if lbl != target_annot.label.name]
                    print(
                        "\n\tYou must specify the new label value for the targeted annotation."
                        f"\n\t\tStatement ID: {target_annot.statement_id}"
                        f"\n\t\tCurrent label: {target_annot.label.name}"
                        f"\n\t\tChunks: {target_chunks_str if target_chunks_str else 'none'}"
                        f"\n\t\tNew label options: {label_options}"
                        "\n\t(select 'q' to cancel the label update)\n"
                    )
                    choice = None  # reset choice, go back to asking for actual input
            if choice in shortcuts_quit:
                print(f"\n\tCancelled label selection for {target_annot.name}.")
                break  # break the label selection loop, go back to annotation selection loop above
            elif choice in supported_label_choices:
                if choice in yes_label_options:
                    new_label = qut01.data.annotations.classes.AnnotationLabel.YES
                elif choice in no_label_options:
                    new_label = qut01.data.annotations.classes.AnnotationLabel.NO
                else:
                    assert choice in unclear_label_options
                    new_label = qut01.data.annotations.classes.AnnotationLabel.UNCLEAR
                if target_annot.label != new_label:
                    target_annot.label = new_label
                    updated_annots.add(target_annot.name)
                    target_annot.last_update = datetime.datetime.now()
                    print(f"\n\tUpdated label for {target_annot.name} to {new_label.name}.")
                    if target_annot.chunks:
                        choice = input("\nDiscard all existing chunks for this annotation? (y/N):\n")
                        confirmed = choice.strip().lower() in ["y", "yes"]
                        if confirmed:
                            target_annot.chunks = []
                            print(f"\n\tDiscarded all chunks for {target_annot.name}.")
                        else:
                            print(f"\n\tKeeping existing chunks for {target_annot.name} with new label.")
                else:
                    print(f"\n\tKeeping same label for {target_annot.name} ({new_label.name}).")
                break
            else:
                print(f"\n\tInvalid label choice! ('{choice}')")
                continue

        if not target_annot.chunks and target_annot.label.name == "YES":
            print(
                f"\n\tWARNING: No supporting chunks in {target_annot.name} with a 'YES' label!"
                "\n\t(you need to specify at least one chunk, otherwise it will be invalid)"
            )

        print(f"\nLabel selection completed for {target_annot.name}.")

        if len(annotations) == 1:
            break  # break the annotation selection loop, go back to main menu (no more annots to check)

    print(f"\nUpdated {len(updated_annots)} annotation(s) in statement {statement_id}.")
    return len(updated_annots) > 0


def _validate_chunks(
    statement_data: "StatementProcessedData",
    annotations: typing.Dict["CriteriaNameType", "ValidatedAnnotation"],  # might have some chunks removed
    max_line_length: int,  # for display purposes only
) -> bool:
    """Allows the user to review and potentially remove chunks from the targeted annotations.

    If a chunk is removed from an annotation object, that object's `last_update` attribute will
    also be modified. If any annotation is modified, this function will return `True`.
    """
    statement_id = statement_data.id
    updated_annots = set()
    while True:  # outer loop to keep selecting annotations for which to review the chunks
        target_annot = _get_target_annot_for_update(
            annotations=annotations,
            max_line_length=max_line_length,
            action_name="VALIDATE ITS CHUNKS",
        )
        if target_annot is None:
            break  # break the annotation selection loop, go back to main menu
        # first, we need to regenerate the chunks and make sure their matches are up-to-date
        qut01.data.statement_utils.StatementProcessedData.create_and_assign_chunks(
            statement_data=statement_data,
            annotation=target_annot,
        )
        # now, we're ready to display chunks with individual sentences matched across the statement
        curr_chunk_idx = 0
        discard_chunks = [False] * len(target_annot.chunks)
        valid_chunk_ids = [str(idx + 1) for idx in range(len(target_annot.chunks))]  # offset by 1
        while target_annot.chunks:  # inner loop to display + validate the chunks for the target annotation
            max_len_bar = "".join(["="] * max_line_length)
            matched_display = _get_matched_sentences_display(
                target_annot.chunks[curr_chunk_idx],
                max_line_length=max_line_length,
            )
            chunk_id_str = f"Reviewing chunk {valid_chunk_ids[curr_chunk_idx]} of {len(valid_chunk_ids)}"
            chunk_flag_str = f"Will be discarded: {'**YES**' if discard_chunks[curr_chunk_idx] else 'no'}"
            chunk_str = (
                f"\n{max_len_bar}\n{chunk_id_str.center(max_line_length)}"
                f"\n{matched_display}"
                f"\n{chunk_flag_str.center(max_line_length)}"
                f"\n{max_len_bar}"
            )
            print(chunk_str)
            # the chunk is now displayed, jump into the validation action loop for it
            chunk_selection_actions = ["", *shortcuts_next, *shortcuts_previous, *valid_chunk_ids]
            while True:
                choice = input(
                    "\nToggle chunk discard flag ('d'), "
                    "keep going (ENTER), "
                    "return to annotation selection ('q'), "
                    "or '?' for help:\n"
                )
                choice = choice.strip().lower()
                if choice in [*shortcuts_help, *shortcuts_info]:
                    help_action_format = "[shortcut1, shortcut2, ...]: Description of action."
                    help_action_bar = "".join(["-"] * len(help_action_format))
                    help_str = (
                        f"{chunk_str}"
                        f"\n\n\tYou must decide whether to discard (remove) the above chunk for "
                        f"{target_annot.name} in statement ID: {statement_id}"
                        f"\n\n\tPossible action shortcuts:\n"
                        f"\n\t\t{help_action_format}"
                        f"\n\t\t{help_action_bar}"
                    )
                    for action_shortcuts, action_desc in possible_chunk_review_actions:
                        help_str += f"\n\t\t{action_shortcuts}: {action_desc}"
                    print(f"{help_str}")
                elif choice in shortcuts_discard:
                    old_discard_flag = discard_chunks[curr_chunk_idx]
                    new_discard_flag = not old_discard_flag
                    print(
                        f"\n\tUpdating chunk {valid_chunk_ids[curr_chunk_idx]} discard flag"
                        f"\n\t\t{old_discard_flag} => {new_discard_flag}"
                    )
                    discard_chunks[curr_chunk_idx] = not old_discard_flag
                    chunk_flag_str = f"Will be discarded: {'**YES**' if discard_chunks[curr_chunk_idx] else 'no'}"
                    chunk_str = (
                        f"\n{max_len_bar}\n{chunk_id_str.center(max_line_length)}"
                        f"\n{matched_display}"
                        f"\n{chunk_flag_str.center(max_line_length)}"
                        f"\n{max_len_bar}"
                    )
                elif choice in shortcuts_all:
                    print(f"\n\tWill DISCARD ALL {len(target_annot.chunks)} CHUNKS in the statement.")
                    discard_chunks = [True] * len(target_annot.chunks)
                    break  # break both chunk review loops, go back to annotation selection loop above
                elif choice in shortcuts_quit:
                    break  # break both chunk review loops, go back to annotation selection loop above
                elif choice in chunk_selection_actions:
                    if not choice or choice in shortcuts_next:
                        next_chunk_idx = min(curr_chunk_idx + 1, len(target_annot.chunks) - 1)
                        if curr_chunk_idx == next_chunk_idx:
                            print("\n\tReached the end of the chunk list!")
                            continue
                    elif choice in shortcuts_previous:
                        next_chunk_idx = max(curr_chunk_idx - 1, 0)
                        if curr_chunk_idx == next_chunk_idx:
                            print("\n\tReached the beginning of the chunk list!")
                            continue
                    else:
                        assert choice in valid_chunk_ids
                        next_chunk_idx = valid_chunk_ids.index(choice)
                    # move the index, break off the input loop, go display new chunk
                    curr_chunk_idx = next_chunk_idx
                    break
                else:
                    print(f"\n\tInvalid action choice! ('{choice}')")

            if choice in shortcuts_quit:
                break  # break the chunk review loop, go back to annotation selection loop above

        if any(discard_chunks):
            assert len(target_annot.chunks) == len(discard_chunks)
            target_annot.chunks = [c for c, flag in zip(target_annot.chunks, discard_chunks) if not flag]
            updated_annots.add(target_annot.name)
            target_annot.last_update = datetime.datetime.now()

        print(
            f"\nReview completed for {target_annot.name} in statement {statement_id}:"
            f"\n\t{len(target_annot.chunks)} supporting chunk(s) kept;"
            f"\n\t{sum(discard_chunks)} supporting chunk(s) discarded."
        )

        if not target_annot.chunks and target_annot.label.name == "YES":
            print(
                f"\n\tWARNING: No supporting chunks in {target_annot.name} with a 'YES' label!"
                "\n\t(you need to specify at least one chunk, otherwise it will be invalid)"
            )

        if len(annotations) == 1:
            break  # break the annotation selection loop, go back to main menu (no more annots to check)

    print(f"\nUpdated {len(updated_annots)} annotation(s) in statement {statement_id}.")
    return len(updated_annots) > 0


def _add_new_chunk(
    statement_data: "StatementProcessedData",
    annotations: typing.Dict["CriteriaNameType", "ValidatedAnnotation"],  # to be updated with the new chunk(s)
    max_line_length: int,  # for display purposes only
) -> bool:
    """Allows the user to add a new chunk to support one or more annotations.

    If a chunk is added/edited in an annotation object, that object's `last_update` attribute will
    also be modified. If any annotation is modified, this function will return `True`.
    """
    statement_id = statement_data.id
    updated_annots = set()
    while True:  # outer loop to keep selecting annotations for which to add/edit chunks
        target_annot = _get_target_annot_for_update(
            annotations=annotations,
            max_line_length=max_line_length,
            action_name="ADD NEW CHUNKS",
        )
        if target_annot is None:
            break  # break the annotation selection loop, go back to main menu
        max_len_bar = "".join(["="] * max_line_length)
        intro_str = f"Adding new chunk for {target_annot.name} in statement {statement_id}"
        intro_bar_str = f"\n{max_len_bar}\n{intro_str.center(max_line_length)}\n{max_len_bar}"
        print(intro_bar_str)
        # for each targeted annotation, update the statement data to reflect its specific coverage
        assert len(target_annot.chunks_to_be_added) == 0  # there should not be any (yet)
        statement_data.refresh_annotations_data([target_annot])

        while True:  # inner loop to select the chunk creation action
            choice = input(
                "\nDo you want to add a chunk manually ('a'), browse existing sentences ('s'), or quit ('q')?\n"
            )
            choice = choice.strip().lower()
            if choice in [*shortcuts_help, *shortcuts_info]:
                help_action_format = "[shortcut1, shortcut2, ...]: Description of action."
                help_action_bar = "".join(["-"] * len(help_action_format))
                help_str = (
                    f"\n\n\tYou must decide whether to add a new chunk manually or from an existing sentence for "
                    f"{target_annot.name} in statement ID: {statement_id}"
                    f"\n\n\tPossible action shortcuts:\n"
                    f"\n\t\t{help_action_format}"
                    f"\n\t\t{help_action_bar}"
                )
                for action_shortcuts, action_desc in possible_chunk_add_actions:
                    help_str += f"\n\t\t{action_shortcuts}: {action_desc}"
                help_str += "\n\n\tNOTE: if you wish to EDIT an existing chunk, drop it and recreate it instead."
                print(f"{help_str}")
            elif choice in [*shortcuts_add, *shortcuts_select]:
                if choice in shortcuts_add:
                    print(f"\nPaste the text for {target_annot.name} (and then press enter repeatedly to finalize):\n")
                    consecutive_enter_counts = 0
                    new_lines = []
                    while True:
                        added_line = input()
                        if not added_line:
                            consecutive_enter_counts += 1
                            if consecutive_enter_counts >= 2:
                                break  # done, break off and finalized added text
                        else:
                            consecutive_enter_counts = 0
                        new_lines.append(added_line)
                    new_chunk = "\n".join(new_lines).strip()
                else:
                    select_title = (
                        f"Adding new chunk for {target_annot.name} in statement "
                        f"{statement_id} from existing sentence(s)"
                    )
                    select_str = f"\n{max_len_bar}\n{select_title.center(max_line_length)}\n{max_len_bar}"
                    sentence_id_map = {}  # str index to sentence text
                    for sentence_idx, sentence in enumerate(statement_data.sentences):
                        sentence_id = str(sentence_idx + 1)  # 1-based
                        sentence_id_map[sentence_id] = sentence
                        sentence_lines = _get_sentence_offset_lines(
                            sentence=sentence,
                            max_line_length=max_line_length - 8,  # we use one full tab (8-space) as prefix
                        )
                        assert len(sentence_lines) > 0
                        sentence_id_prefix = f"[{sentence_id}]".rjust(7)
                        select_str += f"\n{sentence_id_prefix} {sentence_lines[0]}"
                        for sentence_line_idx in range(1, len(sentence_lines)):
                            select_str += f"\n\t{sentence_lines[sentence_line_idx]}"
                    select_str += f"\n{max_len_bar}"
                    print(select_str)
                    while True:
                        new_chunk = None
                        selected_ids = input(
                            f"\nSentence IDs to combine into a chunk for {target_annot.name} "
                            f"in statement {statement_id} (must be contiguous, comma-separated):\n"
                        )
                        selected_ids = selected_ids.strip().lower()
                        if not selected_ids or selected_ids in shortcuts_quit:
                            break
                        selected_ids = [s.strip() for s in selected_ids.split(",")]
                        if any([sid not in sentence_id_map for sid in selected_ids]):
                            print(f"\n\tInvalid sentence ID(s): {selected_ids}")
                            continue
                        selected_idxs = sorted([int(sid) for sid in selected_ids])
                        if len(selected_idxs) > 1 and not np.all(np.diff(selected_idxs) == 1):
                            print(f"\n\tInvalid sentence IDs (not contiguous): {selected_ids}")
                            continue
                        selected_sentences = [sentence_id_map[sid] for sid in selected_ids]
                        new_chunk = default_sentence_join_token.join(selected_sentences) + default_sentence_end_token
                        break
                if not new_chunk:
                    print("\n\tChunk creation cancelled.")
                    continue
                prior_chunk_count = len(target_annot.chunks)
                target_annot.chunks_to_be_added = [new_chunk]
                # refresh the statement data to update all chunks and reflect the new annotation coverage
                statement_data.refresh_annotations_data([target_annot])
                # the validated annotation should also have been updated with the added chunk
                if len(target_annot.chunks) != prior_chunk_count + 1:
                    print(f"{prior_chunk_count=}")
                    print(f"{len(target_annot.chunks)=}")
                    assert len(target_annot.chunks) == prior_chunk_count
                    print(f"\n\tChunk creation FAILED for {target_annot.name}! (make sure it contains text...)")
                else:
                    # update succeeded (we have one more chunk)
                    print(f"\n\tChunk with {len(new_chunk)} characters created for {target_annot.name}")
                    target_annot.last_update = datetime.datetime.now()
                    updated_annots.add(target_annot.name)
                    if target_annot.label != qut01.data.annotations.classes.AnnotationLabel.YES:
                        choice = input(
                            f"\nKeep original label ({target_annot.label.name}) instead of switching to YES? (y/N)"
                        )
                        confirmed = choice.strip().lower() in ["y", "yes"]
                        if confirmed:
                            print(f"\n\tWill keep original {target_annot.label} label for {target_annot.name}.")
                        else:
                            print(f"\n\tSwitching label for {target_annot.name} to YES.")
                            target_annot.label = qut01.data.annotations.classes.AnnotationLabel.YES
                target_annot.chunks_to_be_added = []  # reset the list of added chunks
            elif choice in shortcuts_quit:
                break  # break chunk creation loop, go back to annotation selection loop above
            else:
                print(f"\n\tInvalid action choice! ('{choice}')")

        if len(annotations) == 1:
            break  # break the annotation selection loop, go back to main menu (no more annots to check)

    print(f"\nUpdated {len(updated_annots)} annotation(s) in statement {statement_id}.")
    return len(updated_annots) > 0


def _validate_statement_annotations(
    data_parser: qut01.data.dataset_parser.DataParser,
    statement_id: int,
    target_annot_names: typing.List["CriteriaNameType"],
    pickle_dir_path: typing.Optional[pathlib.Path],  # none = default framework path
    statement_dump_path: typing.Optional[pathlib.Path],  # none = do not dump PDF data
    max_line_length: int,
) -> None:
    """Validates the annotations for the given statement and saves the result.

    List of possible statement-level validation actions:
        1) Set a new label for a specific criterion (YES/NO/UNCLEAR). This will also optionally
           reset all stored chunks for that criterion.
        2) Display all the supporting chunks (one by one) across all criteria that justify their
           respective labels. Optionally displays the raw sentences that were matched to each
           chunk. While displayed, a chunk can be invalidated as supporting a criterion. If after
           being invalidated, a chunk no longer supports ANY criterion, it will be removed from the
           statement's chunk list.
        3) Display all the 'raw' sentences from the statement (one by one). Select any sentence to
           be used as a new chunk that supports one or more criterion. The selected sentence can be
           edited before being transformed into a supporting chunk.
        4) Write the validated annotation(s) for the statement, and go to the next statement.
        5) Skip the statement, and go to the next statement.
        6) Quit without saving any work-in-progress.

    If prior annotations exist for a target annotation type, their labels will be merged and used
    as an initial state ready for validation. Chunks will be considered as 'supporting evidence'
    for each criterion if any annotator flagged that sentence as such.
    """
    print(f"\nInitializing annotation validation loop for statement {statement_id}...")
    statement_data = data_parser.get_processed_data(data_parser.statement_ids.index(statement_id))
    if statement_dump_path is not None:
        with open(statement_dump_path, "wb") as fd:
            fd.write(statement_data.statement_data["pdf_data"])
        print(f"\tPDF successfully saved to: {statement_dump_path}")
    # create the actual annotation objects that will hold the validated (updated) annotation labels
    validated_annots = {
        annot_name: qut01.data.annotations.classes.ValidatedAnnotation(
            annotation_name=annot_name,
            statement_data=statement_data,
        )
        for annot_name in target_annot_names
    }
    statement_info_str = _get_statement_info_str(
        statement_data=statement_data,
        target_annot_names=target_annot_names,
        max_line_length=max_line_length,
    )
    must_save, updated = False, False
    while True:  # main validation action loop for the current statement data
        # note: every time we iterate in this loop, we will print the status bar with basic info
        status_bar_str = _get_status_bar_str(
            statement_data=statement_data,
            annotations=validated_annots,
            max_line_length=max_line_length,
            annotations_updated=updated,
        )
        choice = None
        while not choice:  # action selection loop
            print(status_bar_str)
            choice = input(f"\nSpecify action for statement {statement_data.id} (or '?' for help):\n")
            choice = choice.strip().lower()
            if choice in [*shortcuts_help, *shortcuts_info]:
                if choice in shortcuts_help:
                    help_action_format = "[shortcut1, shortcut2, ...]: Description of action."
                    help_action_bar = "".join(["-"] * len(help_action_format))
                    print(f"\n\tPossible actions:\n\t\t{help_action_format}\n\t\t{help_action_bar}")
                    for action_shortcuts, action_desc in possible_validation_actions:
                        print(f"\t\t{action_shortcuts}: {action_desc}")
                elif choice in shortcuts_info:
                    print(statement_info_str)
                choice = None  # reset choice, go back to asking for actual input
        # if we get here, the user actually picked an action; execute it
        if choice in shortcuts_quit:
            if updated:
                choice = input("\nSave current statement before exiting? (y/N):\n")
                confirmed = choice.strip().lower() in ["y", "yes"]
                if confirmed:
                    must_save = True
            break
        elif choice in shortcuts_write_and_next:
            must_save = True  # note: will not actually save anything if there were no changes at all
            break
        elif choice in shortcuts_skip_and_next:
            if updated:
                choice = input("\nAre you sure you want to skip and discard any annotation? (y/N)\n")
                confirmed = choice.strip().lower() in ["y", "yes"]
            else:
                confirmed = True
            if confirmed:
                print(f"\nSkipping validation for statement {statement_data.id}.")
                break
        elif choice in shortcuts_assign_label:
            updated = (
                _assign_label(
                    statement_data=statement_data,
                    annotations=validated_annots,
                    max_line_length=max_line_length,
                )
                or updated
            )
        elif choice in shortcuts_check_chunks:
            updated = (
                _validate_chunks(
                    statement_data=statement_data,
                    annotations=validated_annots,
                    max_line_length=max_line_length,
                )
                or updated
            )
        elif choice in shortcuts_add_chunks:
            updated = (
                _add_new_chunk(
                    statement_data=statement_data,
                    annotations=validated_annots,
                    max_line_length=max_line_length,
                )
                or updated
            )
        elif choice in shortcuts_all_valid:
            print(f"\nSetting all existing annotations as VALIDATED for statement {statement_data.id}.")
            for annot_name in target_annot_names:
                if not validated_annots[annot_name].is_validated:
                    validated_annots[annot_name].last_update = datetime.datetime.now()
                    updated = True
        else:
            print(f"\n\tInvalid validation action choice! ('{choice}')")

    if must_save and updated:
        print(f"\n\tSaving updated annotation(s) for statement {statement_data.id}...")
        for annot_name in target_annot_names:
            validated_annot = validated_annots[annot_name]
            _save_validated_annotations(
                annotation=validated_annot,
                data_parser=data_parser,
                pickle_dir_path=pickle_dir_path,
            )
        runtime_tags = qut01.utils.config.get_runtime_tags()
        runtime_tags_str = "".join([f"\n\t{k}: {v}" for k, v in runtime_tags.items()])
        commit_msg = f"validated annotations updated\b{runtime_tags_str}"
        data_parser.dataset.commit(commit_msg)
    elif not must_save and updated:
        print(f"\n\tDiscarding updated annotation(s) for statement {statement_data.id}.")
    else:
        print(f"\n\tNo updated annotation needs to be saved for statement {statement_data.id}.")


def run_statement_validator(
    dataset_path: typing.Optional[pathlib.Path] = None,  # none = default framework path
    pickle_dir_path: typing.Optional[pathlib.Path] = None,  # none = default framework path
    temp_dir_path: typing.Optional[pathlib.Path] = None,  # none = use a platform default tmpdir
    restart_from_raw_annotations: bool = False,  # specifies whether to start validation from scratch
    load_validated_annots_from_pickles: bool = True,  # load from pickles (backups), if needed
    skip_already_validated_statements: bool = False,
    max_line_length: int = default_max_line_length,
) -> None:
    """Initializes based on choices and executes the main loop for statement validation.

    Note: the validated annotations are dumped every iteration to a pickle file.
    """
    if dataset_path is None:
        dataset_path = qut01.data.dataset_parser.get_default_deeplake_dataset_path()
    if temp_dir_path is None:
        temp_dir_path = pathlib.Path(tempfile.gettempdir())
    statement_dump_path = temp_dir_path / "statement_to_validate.pdf"

    print(f"\n\nInitializing statement validator from dataset: {dataset_path}\n")

    # STEP 1: prepare the dataset to parse/store the validated annotations
    data_parser = _get_data_parser_for_validation(
        dataset_path=dataset_path,
        restart_from_raw_annotations=restart_from_raw_annotations,
        pickle_dir_path=pickle_dir_path,
        load_validated_annots_from_pickles=load_validated_annots_from_pickles,
    )

    # STEP 2: select the annotation names to validate
    target_annot_names = _get_annot_names_based_on_choice()

    # STEP 3: determine what statement(s) to validate
    target_statement_ids = _get_statement_ids_based_on_choice(
        data_parser=data_parser,
        target_annot_names=target_annot_names,
        skip_already_validated_statements=skip_already_validated_statements,
    )

    # STEP 4: proceed to the actual validation loop (for each statement, one at a time)
    for statement_id in target_statement_ids:
        # note: to keep things simple, we do NOT implement a statement selection step here
        # (all statements will only be processed once, and to re-process them, we have to re-launch)
        _validate_statement_annotations(
            data_parser=data_parser,
            statement_id=statement_id,
            target_annot_names=target_annot_names,
            pickle_dir_path=pickle_dir_path,
            statement_dump_path=statement_dump_path,
            max_line_length=max_line_length,
        )

    statement_dump_path.unlink(missing_ok=True)
    print("\nAll statements processed!")


if __name__ == "__main__":
    qut01.utils.logging.setup_logging_for_analysis_script()
    run_statement_validator()
