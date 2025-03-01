import dataclasses
import pathlib
import re
import typing

import Levenshtein
import numpy as np

from qut01.data.classif_utils import ANNOT_CLASS_NAMES

if typing.TYPE_CHECKING:
    from qut01.data.annotations.chunks import AnnotationTextChunk
    from qut01.data.annotations.classes import AnnotationsBase
    from qut01.data.batch_utils import BatchDictType


@dataclasses.dataclass
class StatementProcessedData:
    """Holds data related to the processed sentences extracted from a PDF statement.

    This class is used to store the processed sentences along with maps that identify which ones are
    tied to annotations and which are not. This may be useful for sampling sentences with specific
    characteristics later.
    """

    text: str
    """String of contiguous (unprocessed) text for this statement.

    Note that this is the string to which all "text index maps" refer to, i.e. an index of 10 in
    such a map refers to the 10th character inside this string.
    """

    sentences: typing.List[str]
    """List of sentences (each a string) processed out of this statement."""

    sentence_text_idx_maps: typing.List[typing.List[int]]
    """List of text indices maps corresponding to each processed sentence in the statement.

    Note that these indices correspond to individual characters inside the original (raw) text of
    the statement, i.e. an index of 10 inside a sentence's indices map corresponds to the 10th
    character inside the original text.
    """

    annotation_counts: typing.Dict[str, int]
    """Number of annotations (not chunks!) associated with this statement, per annot group."""

    annotations: typing.List["AnnotationsBase"]
    """List of full annotation objects (not chunks!) tied to this statement."""

    annotation_chunks: typing.List["AnnotationTextChunk"]
    """List of annotation chunks (across all sentences) tied to this statement."""

    sentence_annotation_chunks: typing.List[typing.List["AnnotationTextChunk"]]
    """List of references to annotation chunks that have been matched for each sentence.

    The top-level list is the same length as `sentences`, as each item corresponds to a list of
    annotation chunks for a particular processed sentence. Sentences that do not have any
    associated annotation have an empty corresponding list.
    """

    sentence_annotation_counts: typing.List[typing.Dict[str, int]]
    """List of annotation counts for each sentence in the statement.

    The counts are stored separately for each annotation (group) type.
    """

    annotated_sentence_idxs: typing.List[int]
    """List of sentence indices for sentences that possess at least one matched annotation."""

    unannotated_sentence_idxs: typing.List[int]
    """List of sentence indices for sentences that do not possess any annotation."""

    statement_data: "BatchDictType"
    """Reference to the original batch dictionary that contains all the data for the statement."""

    @property
    def id(self) -> int:
        """Returns the identifier (from the modern slavery register) for this statement."""
        sid = self.statement_data["statement_id"]
        if isinstance(sid, np.ndarray) and sid.size == 1:
            sid = sid.item()
        return int(sid)

    @property
    def url(self) -> str:
        """Returns the URL for the modern slavery register for this statement."""
        url = self.statement_data["metadata/Link"]
        if isinstance(url, np.ndarray) and url.size == 1:
            url = url.item()
        return str(url)

    @property
    def page_count(self) -> int:
        """Returns the PDF page count for this statement."""
        page_count = self.statement_data["metadata/PageCount"]
        if isinstance(page_count, np.ndarray) and page_count.size == 1:
            page_count = page_count.item()
        return int(page_count)

    @property
    def word_count(self) -> int:
        """Returns the PDF word count for this statement."""
        word_count = self.statement_data["metadata/WordCount"]
        if isinstance(word_count, np.ndarray) and word_count.size == 1:
            word_count = word_count.item()
        return int(word_count)

    @property
    def is_fully_validated(self) -> bool:
        """Returns whether the statement contains all (unique) and fully validated annotations.

        So-called 'validated' annotations are those that have been reviewed and potentially
        corrected by one or more experts. See the `split_utils` module for more information.
        """
        if not self.annotations:
            return False  # no annotations = not fully validated (all annot types need to exist)
        validated_annotations = [annot for annot in self.annotations if annot.is_validated]
        return all(
            [
                sum([a.name == annot_name for a in validated_annotations]) == 1  # should only have one per type
                for annot_name in ANNOT_CLASS_NAMES
            ]
        )

    @staticmethod
    def create(
        statement_tensor_data: "BatchDictType",
        source_text_tensor: str = "fitz/text",
        load_annotations: bool = True,  # whether to try to load any annotation data at all
        pickle_dir_path: typing.Optional[pathlib.Path] = None,  # none = default framework path
        dump_found_validated_annots_as_pickles: bool = False,  # dump to pickles as backup, if needed
        load_validated_annots_from_pickles: bool = False,  # load from pickles (backups), if needed
    ) -> "StatementProcessedData":
        """Creates an instance of `StatementProcessedData` from a raw data parser batch dict."""
        # imports below are here to avoid circular imports
        import qut01.data.preprocess_utils

        assert isinstance(statement_tensor_data, dict), f"bad batch dict type: {type(statement_tensor_data)}"
        assert source_text_tensor in statement_tensor_data, f"missing source text tensor: {source_text_tensor}"
        statement_text = statement_tensor_data[source_text_tensor]
        if isinstance(statement_text, np.ndarray):
            assert statement_text.size == 1
            statement_text = statement_text.item()
        statement_sentences, text_idx_maps = qut01.data.preprocess_utils.get_preprocessed_sentences(
            raw_text=statement_text,
            text_source=qut01.data.preprocess_utils.TextSource.get_enum_from_tensor_name(source_text_tensor),
        )
        qut01.data.preprocess_utils.validate_extracted_sentences(
            statement_text=statement_text,
            sentences=statement_sentences,
            text_idx_maps=text_idx_maps,
        )
        # note: we cannot fill in the annotation data before we create the processed data object
        # (we do it after, and assume that the annotation lists are not used in the meantime)
        statement_data = StatementProcessedData(
            text=statement_text,
            sentences=statement_sentences,
            sentence_text_idx_maps=text_idx_maps,
            annotation_counts=dict(),  # empty for now, will update below!
            annotations=[],  # empty for now, will update below!
            annotation_chunks=[],  # empty for now, will update below!
            sentence_annotation_chunks=[],  # empty for now, will update below!
            sentence_annotation_counts=[],  # empty for now, will update below!
            annotated_sentence_idxs=[],  # empty for now, will update below!
            unannotated_sentence_idxs=[],  # empty for now, will update below!
            statement_data=statement_tensor_data,
        )
        if load_annotations:
            annotations = qut01.data.annotations.classes.get_annotations_for_statement(
                statement_tensor_data,
                pickle_dir_path=pickle_dir_path,
                dump_found_validated_annots_as_pickles=dump_found_validated_annots_as_pickles,
                load_validated_annots_from_pickles=load_validated_annots_from_pickles,
            )
            statement_data.refresh_annotations_data(annotations)
        # all done, object is now fully created+filled
        return statement_data

    @staticmethod
    def create_and_assign_chunks(
        statement_data: "StatementProcessedData",
        annotation: "AnnotationsBase",
    ) -> None:  # note: the provided annotation object will contain the newly generated chunks
        """Creates and assigns the chunk objects for the supporting text of a given annotation.

        Note: this function is static and it does NOT modify the `statement_data` object contents
        directly. It however does modify the content of the provided annotation (that is where the
        chunks will be stored).

        Note: this is where we combine the main body supporting text with the visual element text
        (if any), and discard any optional prefixes used to identify text extracted from figures or
        tables (`fig.pX.<TEXT>` or `tab.pX.<TEXT>`, where `pX` is for page X).
        """
        import qut01.data.annotations.chunks
        import qut01.data.preprocess_utils  # here to avoid circular imports

        supporting_text = []
        if annotation.supporting_text:
            supporting_text.append(annotation.supporting_text)
        if annotation.viz_elem_supporting_text:
            supporting_text.append(annotation.viz_elem_supporting_text)
        supporting_text = qut01.data.preprocess_utils.annotator_separator_token.join(supporting_text)
        if len(supporting_text) == 0:
            # nothing to chunk, it's a no-supporting-info annotation
            annotation.chunks = []
            return
        # split the supporting text into contiguous chunks extracted in one shot (via token)
        text_chunks = re.split(
            qut01.data.preprocess_utils.annotator_separator_pattern,
            supporting_text,
        )
        output_chunks = []
        for chunk_idx, chunk in enumerate(text_chunks):
            # for each chunk, we first look for a prefix token indicating the origin of the text
            fig_match = qut01.data.annotations.chunks.chunk_figure_prefix_pattern.match(chunk)
            tab_match = qut01.data.annotations.chunks.chunk_table_prefix_pattern.match(chunk)
            if fig_match or tab_match:
                assert not fig_match or not tab_match  # can't have both
                chunk = chunk[(fig_match or tab_match).span()[1] :]
                if fig_match:
                    chunk_origin = qut01.data.annotations.chunks.ChunkOrigin.infographic
                else:
                    chunk_origin = qut01.data.annotations.chunks.ChunkOrigin.table
            else:
                # default (no prefix) = text comes from the main body directly
                chunk_origin = qut01.data.annotations.chunks.ChunkOrigin.main_body

            # now, the chunks should be clean sentences (except for those broken across pages)
            chunk_sentences, _ = qut01.data.preprocess_utils.get_preprocessed_sentences(
                raw_text=chunk,
                text_source=qut01.data.preprocess_utils.TextSource.ANNOTATORS,
            )

            chunk_sentences_orig_idxs: typing.List[int] = []
            chunk_sentences_text_idxs: typing.List[typing.List[int]] = []
            chunk_sentences_match_scores: typing.List[float] = []

            for chunk_sentence in chunk_sentences:
                (
                    matched_sentence_idx,
                    matched_sentence_text_idxs,
                    matched_score,
                ) = statement_data.get_match_inside_statement(chunk_sentence)
                chunk_sentences_orig_idxs.append(matched_sentence_idx)
                chunk_sentences_text_idxs.append(matched_sentence_text_idxs)
                chunk_sentences_match_scores.append(matched_score)

            output_chunks.append(
                qut01.data.annotations.chunks.AnnotationTextChunk(
                    # note: this will internally prepare a list of sentences
                    annotation=annotation,
                    annotation_chunk_count=len(text_chunks),
                    chunk_idx=chunk_idx,
                    chunk_text=chunk,
                    chunk_origin=chunk_origin,
                    sentences=chunk_sentences,
                    matched_sentences_orig_idxs=chunk_sentences_orig_idxs,
                    matched_sentences_text_idxs=chunk_sentences_text_idxs,
                    matched_sentences_scores=chunk_sentences_match_scores,
                    statement_data=statement_data,
                )
            )
        annotation.chunks = output_chunks

    def get_match_inside_statement(
        self,
        sentence_to_match: str,
    ) -> typing.Tuple[int, typing.List[int], float]:  # (sentence_idx, text_idx_map, match_score)
        """Returns the index + score of the statement sentence that best matches the given sentence.

        Note: we assume that the given sentence is PROCESSED, i.e. it contains no leftover periods
        or acronyms.

        Matching is done in three stages: first, we look for a perfect match; if one is found, the
        match score will be 1.0. If no perfect match is found, we try to find a sentence in the
        statement that CONTAINS the given sentence. This happens sometimes e.g. due to section
        titles sticking in front of actual sentences following text extraction. If such a match is
        found, the match will be set to an arbitrary high value (0.999). Otherwise, the match is
        found by minimizing the Levenshtein distance between the provided sentence and all
        sentences in the statement. The match score is then defined as `(1-normalized_distance)`.
        """
        if sentence_to_match in self.sentences:  # if a perfect match exists, keep its info
            matched_sentence_idx = self.sentences.index(sentence_to_match)
            matched_sentence_text_idxs = self.sentence_text_idx_maps[matched_sentence_idx]
            matched_score = 1.0  # perfect match
            return matched_sentence_idx, matched_sentence_text_idxs, matched_score

        # next, determine if the match can be found as a substring of any processed sentence
        for matched_sentence_idx, statement_sentence in enumerate(self.sentences):
            if sentence_to_match in statement_sentence:  # bingo
                # this happens e.g. when section titles are found at the start of paragraphs
                substring_start_idx = statement_sentence.index(sentence_to_match)
                substring_end_idx = substring_start_idx + len(sentence_to_match)
                matched_sentence_text_idxs = self.sentence_text_idx_maps[matched_sentence_idx]
                # we keep the sentence text indices only for the matched portion
                matched_sentence_text_idxs = matched_sentence_text_idxs[substring_start_idx:substring_end_idx]
                matched_score = 0.999  # arbitrary high score for almost-perfect match
                return matched_sentence_idx, matched_sentence_text_idxs, matched_score

        # if no partial match found, fall back to a levenshtein-based matching approach
        potential_match_scores = [
            1
            - (
                Levenshtein.distance(sentence_to_match, statement_sentence)
                / max(len(sentence_to_match), len(statement_sentence))
            )
            for statement_sentence in self.sentences
        ]
        matched_sentence_idx = int(np.argmax(potential_match_scores))
        # TODO: figure out if we want to edit this idx map with editops from levenshtein?
        matched_sentence_text_idxs = self.sentence_text_idx_maps[matched_sentence_idx]
        matched_score = potential_match_scores[matched_sentence_idx]
        return matched_sentence_idx, matched_sentence_text_idxs, matched_score

    def refresh_annotations_data(
        self,
        annotations: typing.List["AnnotationsBase"],
    ):
        """Updates the internal annotation-related data attributes with the given annotations."""
        self.annotations = annotations
        self.annotation_counts = {name: 0 for name in ANNOT_CLASS_NAMES}
        for annot in self.annotations:
            self.annotation_counts[annot.name] += 1
        self.annotation_chunks = []
        for annotation in self.annotations:
            # create the chunks (and match them), and then keep the results for the statement data
            StatementProcessedData.create_and_assign_chunks(
                statement_data=self,
                annotation=annotation,  # note: the chunks will be auto-stored in the annotation obj here
            )
            self.annotation_chunks.extend(annotation.chunks)
        self.sentence_annotation_chunks = [[] for _ in range(len(self.sentences))]
        sentence_annotations = [[] for _ in range(len(self.sentences))]
        for chunk in self.annotation_chunks:
            for matched_sentence_idx in chunk.matched_sentences_orig_idxs:
                if chunk not in self.sentence_annotation_chunks[matched_sentence_idx]:
                    self.sentence_annotation_chunks[matched_sentence_idx].append(chunk)
                if chunk.annotation not in sentence_annotations[matched_sentence_idx]:
                    sentence_annotations[matched_sentence_idx].append(chunk.annotation)
        self.sentence_annotation_counts = [
            {
                annot_name: sum([annot.name == annot_name for annot in sentence_annotations[sidx]])
                for annot_name in self.annotation_counts.keys()
            }
            for sidx in range(len(self.sentences))
        ]
        self.annotated_sentence_idxs = [
            sidx for sidx in range(len(self.sentences)) if len(self.sentence_annotation_chunks[sidx])
        ]
        self.unannotated_sentence_idxs = [
            sidx for sidx in range(len(self.sentences)) if not len(self.sentence_annotation_chunks[sidx])
        ]
