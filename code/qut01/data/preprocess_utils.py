"""Contains utility functions to clean up raw text extracted by PDF readers or by annotators."""

import datetime
import enum
import functools
import re
import typing

annotator_separator_token = " // "
"""The token that should be used by annotators to separate text taken from different locations.

This should only be used between complete sentences, but it may also be found in the middle of a
sentence that runs across two pages. The idea of "complete sentences" is also a bit ambiguous, as
text found across different bullet points or text extracted across different cells of a table is
also considered as part of different sentences.
"""

# below are regex patterns and defines used in the cleanup functions defined further down
# (these are unlikely useful outside this file)

pattern_for_generic_capitalized_abbreviations_with_period = re.compile(pattern=r"\b([A-Z]+[\w\d]+\.)\s[a-z\d]")
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
common_abbreviations_with_periods = [
    *abbreviations_with_periods_that_never_end_sentences,
    *abbreviations_with_periods_that_may_end_sentences,
]

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
abbyy_paragraph_separator = r"\s*\n\s*"  # assume abbyy does not leave newlines in text blocks
pattern_for_potential_abbyy_sentence_separators = re.compile(
    pattern="|".join(
        [
            *potential_sentence_separators,
            abbyy_paragraph_separator,
        ]
    ),
)
fitz_text_page_token = "\n----\n\n//\n\n----\n"  # we'll validate that this is correct below
fitz_text_page_token_regex = r"\s*\-\-\-\-\n\n\/\/\n\n\-\-\-\-\s*"
fitz_paragraph_separator = r"\s*\n\s*\n\s*"  # fitz leaves newlines in text blocks, so look for doubles
pattern_for_potential_fitz_sentence_separators = re.compile(
    pattern="|".join(
        [
            fitz_text_page_token_regex,
            *potential_sentence_separators,
            fitz_paragraph_separator,
        ]
    )
)
annotator_paragraph_separator = r"\s*\n\s*\n\s*"  # assume extra newlines extracted = new paragraph
annotator_separator_pattern = (
    r"\.?\s*(?<!http:)(?<!https:)(?<!ftp:)\/\/\s*(?:[\.•\-]\s*)?"  # nosec  (slashslash token, not password)
)
pattern_for_potential_annotated_sentence_separators = re.compile(
    pattern="|".join(
        [
            *potential_sentence_separators,
            annotator_separator_pattern,
            annotator_paragraph_separator,
        ]
    ),
)

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


class TextSource(enum.Enum):
    """Source of the text to process (either an OCR/PDF processing engine, or annotators)."""

    ABBYY = enum.auto()  # text originates from ABBYY FineReader
    FITZ = enum.auto()  # text originates from fitz (PyMuPDF)
    ANNOTATORS = enum.auto()  # text originates from human annotators

    @staticmethod
    def get_enum_from_tensor_name(text_tensor_name: str):
        if text_tensor_name == "fitz/text":
            return TextSource.FITZ
        elif text_tensor_name == "abbyy/text":
            return TextSource.ABBYY
        raise ValueError(f"Unrecognized tensor name: {text_tensor_name}")


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


def get_preprocessed_sentences(
    raw_text: str,
    text_source: TextSource,
) -> typing.Tuple[typing.List[str], typing.List[typing.List[int]]]:
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
    if text_source == TextSource.ABBYY:
        sentence_separator_pattern = pattern_for_potential_abbyy_sentence_separators
    elif text_source == TextSource.FITZ:
        sentence_separator_pattern = pattern_for_potential_fitz_sentence_separators
    elif text_source == TextSource.ANNOTATORS:
        sentence_separator_pattern = pattern_for_potential_annotated_sentence_separators
    else:
        raise ValueError(f"unrecognized text source: {text_source}")
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
