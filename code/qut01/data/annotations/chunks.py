import dataclasses
import enum
import re
import typing

if typing.TYPE_CHECKING:
    from qut01.data.annotations.classes import AnnotationsBase
    from qut01.data.statement_utils import StatementProcessedData


chunk_figure_prefix_pattern = re.compile(pattern=r"^\s*fig\.p(\d+)\.\s*")
"""Prefix used to identify chunks taken from a figure, followed by an integer page number."""

chunk_table_prefix_pattern = re.compile(pattern=r"^\s*tab\.p(\d+)\.\s*")
"""Prefix used to identify chunks taken from a table, followed by an integer page number."""


class ChunkOrigin(enum.Enum):
    """Origin that corresponds to the type of content that a chunk of text was extracted from."""

    main_body = enum.auto()  # text originates from the statement's main text itself
    infographic = enum.auto()  # text originates from an infographic/figure
    table = enum.auto()  # text originates from a table


@dataclasses.dataclass
class AnnotationTextChunk:
    """Holds data related to a 'chunk' of supporting text extracted by an annotator.

    The 'supporting text' for a label given by an annotator can be processed into 'chunks', where
    each chunk is a block of contiguous sentences extracted without a break (//). A chunk therefore
    contains evidence in support of the label which should be also (roughly) contiguous in the
    original text.

    In the majority of cases, a chunk should be readable by itself and determined as relevant or not
    relevant for the label provided by the annotator. This may not always be the case however, as
    some chunks require contextual information to be understood as relevant, and this context may be
    located elsewhere in the statement.
    """

    annotation: "AnnotationsBase"
    """Parent annotation object that holds all the text and attributes provided by annotators."""

    annotation_chunk_count: int
    """Number of annotations 'chunks' (blocks split by //) in the parent annotation."""

    chunk_idx: int
    """Index of this particular chunk in the parent annotation chunks."""

    chunk_text: str
    """The string of contiguous text for this chunk."""

    chunk_origin: ChunkOrigin
    """The original type of text that the chunk was extracted from."""

    sentences: typing.List[str]
    """The list of sentences (each a string) processed out of this chunk."""

    matched_sentences_orig_idxs: typing.List[int]
    """The list of sentence indices corresponding to this chunk's sentences in the statement."""

    matched_sentences_text_idxs: typing.List[typing.List[int]]
    """The list of character indices corresponding to locations in the statement text."""

    matched_sentences_scores: typing.List[float]
    """The list of scores for how well the chunk sentences match with those in the statement."""

    statement_data: "StatementProcessedData"
    """Reference to the parent statement's processed data."""

    @property
    def statement_text(self) -> str:
        """Returns the full text (raw/unprocessed) extracted from the statement."""
        return self.statement_data.text

    @property
    def statement_sentences(self) -> typing.List[str]:
        """Returns the list of (processed) sentences extracted from the statement."""
        return self.statement_data.sentences

    @property
    def statement_sentences_text_idx_maps(self) -> typing.List[typing.List[int]]:
        """Returns the list of (per-char) text indices mapping back to the raw text."""
        return self.statement_data.sentence_text_idx_maps
