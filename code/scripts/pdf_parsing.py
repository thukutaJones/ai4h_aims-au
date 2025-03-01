import argparse
import datetime
import enum
import functools
import pathlib
import re
import typing

import fitz
import numpy as np

fitz_text_page_token = "\n----\n\n//\n\n----\n"  # should be unique enough..., we'll validate that this is correct below
fitz_text_page_token_regex = r"\s*\-\-\-\-\n\n\/\/\n\n\-\-\-\-\s*"
fitz_paragraph_separator = r"\s*\n\s*\n\s*"  # fitz leaves newlines in text blocks, so look for doubles

relevant_statement_years_that_are_not_section_numbers = list(
    range(2015, datetime.datetime.now().year + 2)  # range = 2015+ for UK MSA, and into the near future
)
statement_years_joint_regex = "|".join([rf"\b{y}\b" for y in relevant_statement_years_that_are_not_section_numbers])
section_numbers_regex = rf"^[\.\-\(]?(?!{statement_years_joint_regex})\d+([\.\-]\d*)*\)?"
pattern_for_section_numbers = re.compile(section_numbers_regex)

punctuation_to_remove = r"“”\"\.,;:!\?"
starting_or_ending_punctuation_regex = rf"^\s*[{punctuation_to_remove}]+\s*|\s*[{punctuation_to_remove}]+\s*$"
pattern_for_punctuation_to_remove = re.compile(starting_or_ending_punctuation_regex)

whitespace_cleanup_regex = r"\s+"
pattern_for_whitespace_cleanup = re.compile(whitespace_cleanup_regex)

non_empty_sentence_regex = r"\b\w{2,}\b"  # any more-than-two-letter word
pattern_for_non_empty_sentence = re.compile(non_empty_sentence_regex)

abbreviations_with_periods_that_never_end_sentences = [
    # if these are ever found, we will ALWAYS remove the period(s) and assume the sentence goes on
    "esp.",
    "No.",
    "Num.",
    "Sec.",
    "Fig.",
    "e.g.",
    "i.e.",
    "vs.",
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Hon.",
    "Prof.",
    "Rep.",
]
pattern_for_abbreviations_with_periods_that_never_end_sentences = re.compile(
    pattern="|".join(map(re.escape, abbreviations_with_periods_that_never_end_sentences)),
)

abbreviations_with_periods_that_may_end_sentences = [
    # if these are found before a capital letter, we will assume the sentence has ended
    # (this will be imperfect, but we will probably be OK with the result for high-level statistics)
    "Inc.",
    "Pty.",
    "Ltd.",
    "Pty.Ltd.",
    "Pty. Limited",
    "Co.",
    "Corp.",
    "ABN.",
    "AN.",
    "etc.",
    "Jan.",
    "Feb.",
    "Mar.",
    "Apr.",
    "Jun.",
    "Jul.",
    "Aug.",
    "Sep.",
    "Oct.",
    "Nov.",
    "Dec.",
    "Rev.",
    "Est.",
    "A.I.",
    "Bhd.",
    "Ges.m.b.H.",
    "G.m.b.H.",
    "L.L.C.",
    "LLC.",
    "U.S.",
    "Jr.",
    "Sr.",
    "esq.",
    "approx.",
    "H.R.",
    "PhD.",
    "Ph.D.",
    "Eng.",
    "Atty.",
    "Mgr.",
    "Dir.",
]
pattern_for_abbreviations_with_periods_that_may_end_sentences = re.compile(
    pattern="|".join(
        [
            p + r"[,;:]?\s[^A-Z]"  # could be followed by punctuation, parentheses, or words
            for p in map(re.escape, abbreviations_with_periods_that_may_end_sentences)
        ]
    ),
)

bullet_point_codes = [
    "\u2022",  # • BULLET
    "\u2043",  # ⁃ HYPHEN BULLET
    "\u25E6",  # ◦ WHITE BULLET
    "\u2619",  # ☙ REVERSED ROTATED FLORAL HEART BULLET
    "\u2765",  # ❥ ROTATED HEAVY BLACK HEART BULLET
    "\u29BE",  # ⦾ CIRCLED WHITE BULLET
    "\u29BF",  # ⦿ CIRCLED BULLET
    "\u2023",  # ‣ Triangular Bullet
    "\u2024",  #  One Dot Leader
    "\u204C",  # ⁌ Black Leftwards Bullet
    "\u204D",  # ⁍ Black Rightwards Bullet
    "\u2219",  # ∙ Bullet Operator
    "\u25D8",  # ◘ Inverse Bullet
    "\u25D9",  # ◙ Inverse White Circle
    "\u25AA",  # ▪ Black Small Square
    "\u25AB",  # ▫ White Small Square
    "\u25FD",  # ◽ White Medium Small Square
    "\u25FE",  # ◾ Black Medium Small Square
    "\u2794",  # ➔ Heavy Wide-Headed Rightwards Arrow
    "\u2798",  # ➘ Heavy South East Arrow
    "\u2799",  # ➙ Heavy Rightwards Arrow
    "\u279A",  # ➚ Heavy North East Arrow
    "\u279D",  # ➝ Triangle-Headed Rightwards Arrow
    "\u2B1B",  # ⬛ Black Large Square
    "\u2B1C",  # ⬜ White Large Square
    "\u2B25",  # ⬥ Black Medium Diamond
    "\u2B26",  # ⬦ White Medium Diamond
    "\u2B27",  # ⬧ Black Medium Lozenge
    "\u2B28",  # ⬨ White Medium Lozenge
    "\u2012",  # ‒ Figure Dash
    "\u2013",  # – En Dash
    "\u2014",  # — Em Dash
]

potential_sentence_separators = [
    r"\.\s+",  # after taking care of potential abbreviations, this should indicate new sentences
    rf"\n\s*[{''.join(bullet_point_codes)}]\s*",  # bullet points are considered individual sentences
    r"\n\s*\-\s+",  # dashed points are considered individual sentences
    r"\n\s*[A-Za-z0-9][\.\)]\s+",  # enumerations are considered individual sentences
]

pattern_for_potential_fitz_sentence_separators = re.compile(
    pattern="|".join(
        [
            fitz_text_page_token_regex,
            *potential_sentence_separators,
            fitz_paragraph_separator,
        ]
    )
)


def _validate_or_create_text_idx_map(
    text: str,
    text_idx_map: typing.Optional[typing.List[int]],
) -> typing.List[int]:
    """Creates or validates an array of indices that map to characters inside the given string."""
    # note: we will perform cleanups while updating a map of char locations in the original raw text
    # (this is used to backtrack from resulting sentences to the original input raw text later)
    assert isinstance(text, str)
    if text_idx_map is None:
        text_idx_map = list(range(len(text)))
    assert isinstance(text_idx_map, list) and len(text_idx_map) == len(text)
    assert all([isinstance(idx, int) and idx >= 0 for idx in text_idx_map])
    return text_idx_map


def fix_potential_abbreviations(
    text: str,
    text_idx_map: typing.Optional[typing.List[int]],
) -> typing.Tuple[str, typing.List[int]]:
    """Removes dots used as part of abbreviations to make sure they do not split sentences."""
    text_idx_map = _validate_or_create_text_idx_map(text, text_idx_map)
    text_idxs_to_remove = []

    def _remove_periods_from_abbreviations_that_are_not_sentence_ends(match: re.Match):
        match_start_idx, match_end_idx = match.span()
        for idx_offset, ch in reversed(list(enumerate(match.group()))):
            if ch == ".":
                text_idxs_to_remove.append(match_start_idx + idx_offset)
        return match.group().replace(".", "")

    # first, remove periods for abbreviations that should NEVER end a sentence
    text = pattern_for_abbreviations_with_periods_that_never_end_sentences.sub(
        repl=_remove_periods_from_abbreviations_that_are_not_sentence_ends,
        string=text,
    )
    for idx in reversed(sorted(text_idxs_to_remove)):
        del text_idx_map[idx]
    text_idxs_to_remove = []
    # next, remove periods for abbreviations that look like they did not end the sentence
    text = pattern_for_abbreviations_with_periods_that_may_end_sentences.sub(
        repl=_remove_periods_from_abbreviations_that_are_not_sentence_ends,
        string=text,
    )
    for idx in reversed(sorted(text_idxs_to_remove)):
        del text_idx_map[idx]
    return text, text_idx_map


def split_text_into_sentences(
    text: str,
    separator_pattern: re.Pattern,
    text_idx_map: typing.Optional[typing.List[int]],
) -> typing.Tuple[typing.List[str], typing.List[typing.List[int]]]:
    """Converts a raw text block into a list of sentences, each with its own char idx list."""
    # note: the char location idx lists map back to the original raw text (useful for debugging!)
    text_idx_map = _validate_or_create_text_idx_map(text, text_idx_map)
    separator_matches = list(separator_pattern.finditer(text))
    sentence_idxs = []  # list of (sentence_start_idx, sentence_end_idx) tuples
    last_separator_end_idx = 0
    for match in separator_matches:
        separator_start_idx, separator_end_idx = match.span()
        sentence_idxs.append((last_separator_end_idx, separator_start_idx))
        last_separator_end_idx = separator_end_idx
    if last_separator_end_idx != len(text):
        sentence_idxs.append((last_separator_end_idx, len(text)))
    sentences = [text[start_idx:end_idx] for start_idx, end_idx in sentence_idxs]
    sentence_idx_maps = [text_idx_map[start_idx:end_idx] for start_idx, end_idx in sentence_idxs]
    return sentences, sentence_idx_maps


def cleanup_sentence(
    text: str,
    text_idx_map: typing.Optional[typing.List[int]],
) -> typing.Tuple[str, typing.List[int]]:
    """Removes unnecessary whitespaces, special characters, and section numbers from the text."""
    text_idx_map = _validate_or_create_text_idx_map(text, text_idx_map)
    text_idxs_to_remove: typing.List[int] = None  # noqa

    def _execute_cleanup_fn_and_update_idx_map(func: typing.Callable) -> str:
        nonlocal text_idxs_to_remove
        text_idxs_to_remove = []
        res = func()
        for idx in reversed(sorted(text_idxs_to_remove)):
            del text_idx_map[idx]
        return res

    def _cleanup_regex_match_with_no_replacement(match: re.Match) -> str:
        match_start_idx, match_end_idx = match.span()
        for idx in range(match_start_idx, match_end_idx):
            text_idxs_to_remove.append(idx)
        return ""

    def _cleanup_regex_with_whitespace_replacement(match: re.Match) -> str:
        match_start_idx, match_end_idx = match.span()
        for idx in range(match_start_idx + 1, match_end_idx):
            text_idxs_to_remove.append(idx)
        return " "

    def _convert_text_to_ascii(text_: str) -> str:
        converted_text = ""
        for idx, char in enumerate(text_):
            try:
                char.encode("ascii")
                converted_text += char
            except UnicodeEncodeError:
                text_idxs_to_remove.append(idx)
        return converted_text

    # first, remove all section numbers that might be at the start of sentences
    text = _execute_cleanup_fn_and_update_idx_map(
        func=functools.partial(
            pattern_for_section_numbers.sub,
            repl=_cleanup_regex_match_with_no_replacement,
            string=text,
        ),
    )

    # next, convert all chars to ascii to drop uncommon symbols
    text = _execute_cleanup_fn_and_update_idx_map(
        func=functools.partial(
            _convert_text_to_ascii,
            text_=text,
        ),
    )

    # drop all punctuation at the start or end of the sentence
    text = _execute_cleanup_fn_and_update_idx_map(
        func=functools.partial(
            pattern_for_punctuation_to_remove.sub,
            repl=_cleanup_regex_match_with_no_replacement,
            string=text,
        ),
    )

    # cleanup all whitespaces by replacing them with duplicate-less single spaces
    text = _execute_cleanup_fn_and_update_idx_map(
        func=functools.partial(
            pattern_for_whitespace_cleanup.sub,
            repl=_cleanup_regex_with_whitespace_replacement,
            string=text,
        ),
    )

    # finally, remove any whitespaces still at the start or end of the sentence
    if len(text) and text[0] == " ":
        text = text[1:]
        text_idx_map = text_idx_map[1:]
    if len(text) and text[-1] == " ":
        text = text[:-1]
        text_idx_map = text_idx_map[:-1]

    return text, text_idx_map


def validate_extracted_sentences(
    statement_text,
    sentences,
    text_idx_maps,
) -> None:
    """Internal function used to validate that the char index maps are indeed correct."""
    assert len(sentences) == len(text_idx_maps)
    for sentence, text_idx_map in zip(sentences, text_idx_maps):
        assert all([0 <= idx < len(statement_text) for idx in text_idx_map])
        orig_sentence = "".join([statement_text[idx] for idx in text_idx_map])
        orig_sentence = re.sub(r"\s+", " ", orig_sentence)  # cleanup whitespace stuff
        assert orig_sentence == sentence


def get_preprocessed_sentences(raw_text: str) -> typing.Tuple[typing.List[str], typing.List[typing.List[int]]]:
    """Performs the full suite of preprocessing steps on a raw text block.

    This includes fixing abbreviations, splitting into sentences, and removing unnecessary
    whitespaces, special characters, and section numbers.

    The resulting cleaned up sentences will be returned along with a list of indices that map each
    character inside the sentence strings back to the characters inside the input text block.
    """
    # the idx map below is used to backtrack from the output sentences to the original input string
    text_idx_map = list(range(len(raw_text)))
    # first, look for abbreviations with periods and remove those before splitting sentences
    text, text_idx_map = fix_potential_abbreviations(raw_text, text_idx_map)
    sentence_separator_pattern = pattern_for_potential_fitz_sentence_separators

    # split the resulting text using sentence separators
    sentences, sentence_idx_maps = split_text_into_sentences(
        text=text,
        separator_pattern=sentence_separator_pattern,
        text_idx_map=text_idx_map,
    )
    # cleanup the resulting sentences (remove prefixes, whitespaces, drop empties, ...)
    output_sentences, output_sentence_idx_maps = [], []
    for sentence, sentence_idx_map in zip(sentences, sentence_idx_maps):
        sentence, sentence_idx_map = cleanup_sentence(sentence, sentence_idx_map)
        assert len(sentence) == len(sentence_idx_map)
        if pattern_for_non_empty_sentence.search(sentence):
            output_sentences.append(sentence)
            output_sentence_idx_maps.append(sentence_idx_map)
    return output_sentences, output_sentence_idx_maps


def create_sentence_contextualized_data(
    statement_data: typing.Dict,
    target_sentence_idxs: typing.List[int],
    target_sentences: typing.List[str],
    context_word_count: int = 0,
    left_context_boundary_token=None,
    right_context_boundary_token=None,
    sentence_join_token: str = ". ",
) -> str:
    """Creates and returns a sentence data object filled with necessary context."""
    assert len(target_sentences) == 1

    # combine the target sentences into a contiguous text block with separators tokens
    target_text = sentence_join_token.join(target_sentences)
    output_text = target_text
    # finally, add context (if needed) while keeping mask updated to only focus on target sentences
    if context_word_count:
        full_text_left = sentence_join_token.join(statement_data["sentences"][: target_sentence_idxs[0]])
        if full_text_left:
            full_text_left += sentence_join_token
        full_text_right = sentence_join_token.join(statement_data["sentences"][target_sentence_idxs[-1] + 1 :])
        full_text_right = sentence_join_token + full_text_right
        # left = full_text[0 : full_text.index(target_sentences[0])].split(" ")
        # right = full_text[full_text.index(target_sentences[-1]) + len(target_sentences[-1]) :].split(" ")
        left = full_text_left.split(" ")
        right = full_text_right.split(" ")
        to_add_left = left[-context_word_count // 2 :]
        to_add_right = right[: context_word_count // 2]
        output_text = (
            " ".join(to_add_left) + f" {left_context_boundary_token} {target_text} "
            f"{right_context_boundary_token}" + " ".join(to_add_right)
        ).strip()
    return output_text


def parse_statements(proc_file_path: str) -> typing.Dict:
    """Parse a statement provided via a local file path and return a Dict."""
    word_count, img_count = 0, 0
    page_count = 0
    pdf_data_bytes = None
    result_dict = {}

    with open(pathlib.Path(proc_file_path), mode="rb") as fd:
        pdf_data_bytes = fd.read()
    with fitz.open(proc_file_path) as pdf_reader:  # noqa
        page_count = pdf_reader.page_count
        statement_text = []
        for page_idx in range(pdf_reader.page_count):
            page = pdf_reader.load_page(page_idx)
            statement_text.append(page.get_text("text"))
            word_count += len(page.get_text("words"))
            images = page.get_images(full=True)
            img_count += len(images)

    result_dict["PageCount"] = page_count
    result_dict["WordCount"] = word_count
    result_dict["ImageCount"] = img_count
    result_dict["pdf_data"] = np.frombuffer(pdf_data_bytes, dtype=np.uint8)
    result_dict["fitz/text"] = fitz_text_page_token.join(statement_text)
    result_dict["tokens"] = {"fitz_text_page_token": fitz_text_page_token}

    statement_sentences, text_idx_maps = get_preprocessed_sentences(raw_text=result_dict["fitz/text"])

    validate_extracted_sentences(
        statement_text=result_dict["fitz/text"],
        sentences=statement_sentences,
        text_idx_maps=text_idx_maps,
    )

    # List of sentences (each a string) processed out of this statement.
    result_dict["sentences"] = statement_sentences

    # List of text indices maps corresponding to each processed sentence in the statement.
    #   Note that these indices correspond to individual characters inside the original (raw) text of
    #   the statement, i.e. an index of 10 inside a sentence's indices map corresponds to the 10th
    #   character inside the original text.
    result_dict["sentence_text_idx_maps"] = text_idx_maps

    return result_dict


def get_contextualized_sentences(
    statement_data: typing.Dict,
    context_word_count: int = 0,
    left_context_boundary_token=None,
    right_context_boundary_token=None,
    sentence_join_token: str = ". ",
) -> typing.List[str]:
    """Prepares and returns contextualized sentence data samples for all sentences in the statement."""
    assert context_word_count >= 0, f"invalid context word count: {context_word_count}"
    if context_word_count > 0:
        if left_context_boundary_token is None or right_context_boundary_token is None:
            raise ValueError(
                "context_word_count > 0 means we are using context. "
                "Please provide both left_context_boundary_token and "
                "right_context_boundary_token"
            )

    output_samples = []
    for sentence_idx, sentence in enumerate(statement_data["sentences"]):
        output_samples.append(
            create_sentence_contextualized_data(
                statement_data=statement_data,
                target_sentence_idxs=[sentence_idx],
                target_sentences=[sentence],
                context_word_count=context_word_count,
                left_context_boundary_token=left_context_boundary_token,
                right_context_boundary_token=right_context_boundary_token,
                sentence_join_token=sentence_join_token,
            )
        )
    statement_data["contextualized_sentences"] = output_samples
    return statement_data


if __name__ == "__main__":
    # pip install PyMuPDF
    # For Bert Model, use '[SEP]' for both left_sep, and right_sep
    # For Llama models, use '<|start_header_id|>' for left_sep and '<|end_header_id|>' for right_sep.

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="path to statement", required=True)
    parser.add_argument("--context", help="context length", type=int, default=0)
    parser.add_argument("--join_token", help="sentence join token", default=". ")
    parser.add_argument("--left_sep", help="left context separator", default="[SEP]")
    parser.add_argument("--right_sep", help="right context separator", default="[SEP]")
    args = parser.parse_args()

    result = parse_statements(args.file)
    print(f"Sentences: {result['sentences']}")

    print("\n\n\n")
    get_contextualized_sentences(result, args.context, args.left_sep, args.right_sep, args.join_token)
    print(f"Contextualized Sentences: {result['contextualized_sentences']}")
