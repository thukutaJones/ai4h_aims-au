"""Contains static definitions related to the AIMS data split used for all experiments.

NOTE: the content of this file should NEVER be modified. This ensures that all past and future
experiments remain comparable, and that there are no leaks across the subsets.
"""
import collections
import typing

import numpy as np

import qut01.data.classif_utils
import qut01.utils.logging

if typing.TYPE_CHECKING:
    from qut01.data.classif_utils import ClassifSetupType
    from qut01.data.dataset_parser import DataParser


logger = qut01.utils.logging.get_logger(__name__)


supported_subset_types = tuple(["train", "valid", "test", "gold", "unused"])
"""Name of the subsets (or loaders) that are supported by the AIMS dataset.

The 'train' and 'valid' subsets can vary across experiments based on split settings provided to the
data modules. They should always primarily contain statements with noisy labels. The valid subset
may also contain fully validated (gold) statements from the reserved gold subset.

The 'test' subset should contain only statements from the reserved gold subset that possess fully
validated annotations. The exact number of statements may vary depending on the size of the
reserved gold subset itself and on the proportion of fully validated statements used in the 'valid'
subset. See the `gold_set_statement_ids_for_valid_subset` for more information.

The 'gold' subset is fixed and consists of predetermined statements that are reserved so that we
can manually validate or correct their annotations and make sure they are of high quality. The
exact size of the subset may vary as we validate (and correct) more annotations in the future. The
IDs in this subset will likely overlap with those in the valid and test subsets.

The 'unused' subset varies across experiments and contains the statement IDs that have NOT been
used in any other subset.
"""

SubsetType = typing.Literal[*supported_subset_types]  # noqa

variable_subset_types = tuple(["train", "valid"])
"""Name of the subsets (or loaders) that can vary across experiments with the AIMS dataset.

See the docstring of the `supported_subset_types` attribute for more information. The settings
that can be used to vary these subsets are 1) the split seed; 2) the split ratios; 3) the target
classification task; and 4) whether gold statements are added to the valid subset or not.
"""

VariableSubsetType = typing.Literal[*variable_subset_types]  # noqa

gold_set_seed: int = 0
"""The 'gold' set is a subset of the statements that are reserved for high-quality evaluations.

By evaluating models with the statements in this subset, we can provide a more robust idea of their
performance. Note however that this reserved 'gold' set is a very small portion of the entire
dataset, and only a fraction of it will be thoroughly validated and corrected as needed. The seed
used to identify the statements in this 'gold' subset should NEVER change.
"""

gold_set_reserved_size: int = 500
"""The reserved 'gold' subset may contain up to this many statements.

The real size of this subset will depend on the number of carefully re-annotated statements that
have been processed by experts so far. This size is unlikely to reach this upper bound, but
statements that may be re-annotated in the future need to be identified in advance.
"""

gold_set_expected_input_data_parser_size: int = 8629  # valid as of 2024-04-17 with v0.5.0
"""The size of the data parser expected by the gold subset split function."""

gold_set_expected_input_cluster_count: int = 3599  # valid as of 2024-04-17 with v0.5.0
"""The number of clusters expected by the gold subset split function."""

gold_set_banned_statements: typing.Sequence[int] = tuple([
    320,  # used in the annotation spec / cheat sheet
    1004,  # used in the annotation spec / cheat sheet
    1276,  # used in the annotation spec / cheat sheet
    1535,  # used in the annotation spec / cheat sheet
    1674,  # used in the annotation spec / cheat sheet
    1693,  # used in the annotation spec / cheat sheet
    2314,  # used in the annotation spec / cheat sheet
    2499,  # used in the annotation spec / cheat sheet
    2503,  # used in the annotation spec / cheat sheet
    4480,  # used in the annotation spec / cheat sheet
    5508,  # used in the annotation spec / cheat sheet
    5916,  # used in the annotation spec / cheat sheet
    6198,  # used in the annotation spec / cheat sheet
    7101,  # used in the annotation spec / cheat sheet
    7480,  # used in the annotation spec / cheat sheet
    7832,  # used in the annotation spec / cheat sheet
    7944,  # used in the annotation spec / cheat sheet
    7997,  # used in the annotation spec / cheat sheet
    8609,  # used in the annotation spec / cheat sheet
    8717,  # used in the annotation spec / cheat sheet
    8933,  # used in the annotation spec / cheat sheet
    9309,  # used in the annotation spec / cheat sheet
    10015,  # used in the annotation spec / cheat sheet
    10017,  # used in the annotation spec / cheat sheet
    10029,  # used in the annotation spec / cheat sheet
    10045,  # used in the annotation spec / cheat sheet
    10056,  # used in the annotation spec / cheat sheet
    10070,  # used in the annotation spec / cheat sheet
    10073,  # used in the annotation spec / cheat sheet
    10077,  # used in the annotation spec / cheat sheet
    10079,  # used in the annotation spec / cheat sheet
    10080,  # used in the annotation spec / cheat sheet
    10084,  # used in the annotation spec / cheat sheet
    10088,  # used in the annotation spec / cheat sheet
    10095,  # used in the annotation spec / cheat sheet
    10171,  # used in the annotation spec / cheat sheet
    10698,  # used in the annotation spec / cheat sheet
    10720,  # used in the annotation spec / cheat sheet
    10736,  # used in the annotation spec / cheat sheet
    11569,  # used in the annotation spec / cheat sheet
    11808,  # used in the annotation spec / cheat sheet
    11954,  # used in the annotation spec / cheat sheet
    265,  # used with ChatGPT online (might have been leaked)
    11403,  # used with ChatGPT online (might have been leaked)
    11511,  # used with ChatGPT online (might have been leaked)
    11562,  # used with ChatGPT online (might have been leaked)
    12009,  # used with ChatGPT online (might have been leaked)
    299,  # fully scanned (contains no sentences after preprocessing)
    554,  # fully scanned (contains no sentences after preprocessing)
    874,  # fully scanned (contains no sentences after preprocessing)
    971,  # fully scanned (contains no sentences after preprocessing)
    1469,  # fully scanned (contains no sentences after preprocessing)
    1705,  # fully scanned (contains no sentences after preprocessing)
    2171,  # fully scanned (contains no sentences after preprocessing)
    2676,  # fully scanned (contains no sentences after preprocessing)
    2724,  # fully scanned (contains no sentences after preprocessing)
    3007,  # fully scanned (contains no sentences after preprocessing)
    3063,  # fully scanned (contains no sentences after preprocessing)
    3155,  # fully scanned (contains no sentences after preprocessing)
    3543,  # fully scanned (contains no sentences after preprocessing)
    3726,  # fully scanned (contains no sentences after preprocessing)
    4572,  # fully scanned (contains no sentences after preprocessing)
    4846,  # fully scanned (contains no sentences after preprocessing)
    4902,  # fully scanned (contains no sentences after preprocessing)
    5540,  # fully scanned (contains no sentences after preprocessing)
    5744,  # fully scanned (contains no sentences after preprocessing)
    5787,  # fully scanned (contains no sentences after preprocessing)
    7541,  # fully scanned (contains no sentences after preprocessing)
    7750,  # fully scanned (contains no sentences after preprocessing)
    8102,  # fully scanned (contains no sentences after preprocessing)
    8151,  # fully scanned (contains no sentences after preprocessing)
    8258,  # fully scanned (contains no sentences after preprocessing)
    8322,  # fully scanned (contains no sentences after preprocessing)
    8732,  # fully scanned (contains no sentences after preprocessing)
    9077,  # fully scanned (contains no sentences after preprocessing)
    9807,  # fully scanned (contains no sentences after preprocessing)
    9817,  # fully scanned (contains no sentences after preprocessing)
    10213,  # fully scanned (contains no sentences after preprocessing)
    10695,  # fully scanned (contains no sentences after preprocessing)
    11365,  # fully scanned (contains no sentences after preprocessing)
    11560,  # fully scanned (contains no sentences after preprocessing)
    11764,  # fully scanned (contains no sentences after preprocessing)
    11788,  # fully scanned (contains no sentences after preprocessing)
    11896,  # fully scanned (contains no sentences after preprocessing)
    12548,  # fully scanned (contains no sentences after preprocessing)
    12816,  # fully scanned (contains no sentences after preprocessing)
    12866,  # fully scanned (contains no sentences after preprocessing)
    13071,  # fully scanned (contains no sentences after preprocessing)
    13093,  # fully scanned (contains no sentences after preprocessing)
    13183,  # fully scanned (contains no sentences after preprocessing)
    13307,  # fully scanned (contains no sentences after preprocessing)
    13604,  # fully scanned (contains no sentences after preprocessing)
    13719,  # fully scanned (contains no sentences after preprocessing)
    13809,  # fully scanned (contains no sentences after preprocessing)
    13944,  # fully scanned (contains no sentences after preprocessing)
    14076,  # fully scanned (contains no sentences after preprocessing)
    14369,  # fully scanned (contains no sentences after preprocessing)
    14422,  # fully scanned (contains no sentences after preprocessing)
    14600,  # fully scanned (contains no sentences after preprocessing)
    14862,  # fully scanned (contains no sentences after preprocessing)
    8583,  # contains two statements (one for a previous year), confusing
    # add more banned statements here, if needed (but keep at least 300 reserved statements)
])  # fmt: skip
"""List of statements that should be removed from the statements reserved for the gold set.

There are different reasons to exclude these statements, including:
- the statement is fully scanned and it contains no text that is easy to parse without OCR;
- the statement was used to provide examples in annotation specifications (or in cheat sheets)
  which may be used inside prompts to guide zero-shot model predictions;
- the statement was used in preliminary experiments with cloud-based LLM providers that may
  have kept the chatbot interactions about the statement to train future LLMs;
- the statement (or part of it) is unreadable in PDF format due to corruption;
- the statement is an extreme outlier in terms of number of pages or images.

Note that these 'banned' statements are only removed from being potentially reserved for the gold
set. It means that they can still be otherwise used for model training/validation purposes.
"""

gold_set_statement_ids_for_valid_subset: typing.Sequence[int] = tuple([  # last update: 2024-05-07
    # (if the split settings have not been touched, these should all already be in the gold set!)
    61, 96, 136, 241, 373, 462, 890, 949, 991, 1033, 1797, 1973, 1974, 2757, 2824, 3584, 3708,
    3752, 3786, 3847, 3961, 4128, 4471, 4511, 4680, 4744, 5015, 5137, 5169, 5284, 6063, 6272,
    6676, 7068, 7474, 7476, 7738, 7771, 8585, 9307, 9313, 10291, 10492, 11201, 11377, 11499,
    11660, 11820, 12153, 14984,
])  # fmt: skip
"""List of identifiers for gold set statements intended to be used for model validation.

Note that these statements have been fully re-annotated ("validated") by a single expert instead of
by multiple experts.
"""


gold_set_statement_ids_for_test_subset: typing.Sequence[int] = tuple([  # last update: 2024-07-31
    # (if the split settings have not been touched, these should all already be in the gold set!)
    248, 454, 666, 2405, 3547, 4305, 4490, 4651, 4992, 7276, 7707, 8560, 8960, 9049, 9285, 10129,
    10579, 10726, 10959, 11012, 11119, 11135, 11282, 11816, 11998, 12318, 12373, 12464, 12470,
    12532, 12563, 13294, 13357, 13428, 13459, 13523, 13552, 13656, 13833, 13845, 13908, 13978,
    14039, 14055, 14094, 14296, 14358, 14402, 14618, 14781,
])  # fmt: skip
"""List of identifiers for gold set statements intended to be used for (final) model testing.

Note that these statements have been fully re-annotated by an expert and validated by two other
experts in order to make sure they are of highest quality.
"""

SIDClusterType = typing.Tuple[int, ...]
"""Type of statement ID clusters.

The clusters should be FIXED, i.e. there is nothing that should make us want to remove or add an
item from a cluster (they are entirely based on entity names and trademarks).
"""

_gold_set_first_cluster: SIDClusterType = tuple([11499])  # valid as of 2024-04-17 with v0.5.0
"""Hard-coded value of the SIDs in the first cluster returned in the reserved gold set.

This is used for internal checks only, and should never change, unless the dataset grows or unless
the clustering algo changes.
"""


def get_split_statement_ids(
    data_parser: "DataParser",
    classif_setup: "ClassifSetupType",
    train_valid_split_ratios: typing.Dict[VariableSubsetType, float],
    train_valid_split_seed: int,
    use_gold_set: bool = True,
) -> typing.Dict[SubsetType, typing.List[int]]:
    """Returns the list of statement identifiers for all train/valid/test subsets of the dataset.

    This is done by first splitting the gold set statements that have already been fully validated
    into the correct subsets (valid or test). Then, all statements that were not reserved for the
    gold set are split across the train and valid subsets.
    """
    logger.info("Fetching all statement clusters to regenerate reserved gold set...")
    reserved_gold_sid_clusters = get_reserved_gold_id_clusters(data_parser)
    reserved_gold_sids = [sid for cluster in reserved_gold_sid_clusters for sid in cluster]
    logger.info("Fetching non-reserved + annotated statement clusters to generate train/valid split...")
    trainvalid_sid_clusters = _get_statement_clusters(
        data_parser=data_parser,
        classif_setup=classif_setup,
        ignore_sids=reserved_gold_sids,
    )
    trainvalid_sids = [sid for cluster in trainvalid_sid_clusters for sid in cluster]
    assert not (set(reserved_gold_sids) & set(trainvalid_sids))
    sid_clusters = {
        "train": [],
        "valid": [],
        "test": [],
    }

    validated_gold_sid_clusters, assigned_gold_count = [], {"valid": 0, "test": 0}
    if use_gold_set:
        # go fetch the reserved statements that are FULLY VALIDATED (i.e. that have been verified by experts)
        validated_gold_sid_clusters = get_validated_gold_id_clusters(
            data_parser=data_parser,
            reserved_gold_clusters=reserved_gold_sid_clusters,
        )
        # we will assign those to valid/test; the two sets of statement IDs should NOT intersect at all
        assert not (set(gold_set_statement_ids_for_valid_subset) & set(gold_set_statement_ids_for_test_subset))
        # for each set, go find the cluster that matches the specified ID, and assign it to that set
        for subset_name, expected_gold_sids in zip(
            ["valid", "test"],
            [gold_set_statement_ids_for_valid_subset, gold_set_statement_ids_for_test_subset],
        ):
            clusters_to_assign = []
            for expected_gold_sid in expected_gold_sids:
                # each gold sid we expect to have fully annotated should only match one cluster
                matched_clusters = [c for c in validated_gold_sid_clusters if expected_gold_sid in c]
                if len(matched_clusters) != 1:
                    raise AssertionError(
                        "unexpected issue found in gold split assignment: "
                        f"\tmissing statement {expected_gold_sid} that must go in the gold {subset_name} set"
                    )
                clusters_to_assign.append(matched_clusters[0])
            # the gold sids might match multiple times to the same cluster, get rid of duplicates
            clusters_to_assign = [c for c in set(clusters_to_assign)]
            sid_clusters[subset_name].extend(clusters_to_assign)
            assigned_gold_count[subset_name] += sum([len(c) for c in clusters_to_assign])

    # next, assign the remaining (non-reserved) sid clusters to the train/valid sets (randomly)
    validate_split_ratios(train_valid_split_ratios)
    trainvalid_cluster_idxs = [cidx for cidx in range(len(trainvalid_sid_clusters))]
    np.random.default_rng(train_valid_split_seed).shuffle(trainvalid_cluster_idxs)
    idx_offset = 0
    for subset_name in ["train", "valid"]:
        # note: assigning clusters to train set before valid here means we can omit a certain % of
        #       unreserved statements in the total (e.g. 80% train, 0% valid), and use the remains
        #       later in other analyses (e.g. to compare model perf on noisy vs gold valid sets)
        subset_clusters = sid_clusters[subset_name]
        split_ratio = train_valid_split_ratios.get(subset_name, 0.0)
        split_count = int(round(split_ratio * len(trainvalid_sid_clusters)))
        picked_cluster_idxs = trainvalid_cluster_idxs[idx_offset : idx_offset + split_count]
        subset_clusters.extend([trainvalid_sid_clusters[cidx] for cidx in picked_cluster_idxs])
        idx_offset += split_count

    # cluster assignment is done, just need to cleanup and make sure we did it correctly:
    sids = {
        subset_name: [sid for cluster in clusters for sid in cluster]  # flatten clusters into a list
        for subset_name, clusters in sid_clusters.items()
    }
    validate_split(sids)  # makes sure we did not leak a statement across two sets
    gold_sids = [sid for cluster in validated_gold_sid_clusters for sid in cluster]
    sids["gold"] = gold_sids  # we keep this around for debugging only
    sids["unused"] = [  # we keep this around for debugging or for subsequent OOD/split analyses
        # (note: this will be useful when parsing which statements were NOT used to train/valid)
        sid
        for sid in data_parser.statement_ids
        if not any([sid in subset for subset in sids.values()])
    ]
    ratios = {
        subset_name: len(sids[subset_name]) / (len(gold_sids) + len(trainvalid_sids))
        for subset_name, clusters in sid_clusters.items()
    }
    valid_reg_count = len(sids["valid"]) - assigned_gold_count["valid"]
    valid_count_str = f"{valid_reg_count} regular statements, {assigned_gold_count['valid']} gold statements"
    logger.info(
        "Split complete:"
        f"\n\t{len(sid_clusters['test'])} clusters ({len(sids['test'])} gold statements) assigned to test set"
        f"\n\t{len(sid_clusters['valid'])} clusters ({valid_count_str}) assigned to valid set"
        f"\n\t\t({ratios['valid']:.1%} of total statements, requested was {train_valid_split_ratios['valid']:.1%})"
        f"\n\t{len(sid_clusters['train'])} clusters ({len(sids['train'])} regular statements) assigned to train set"
        f"\n\t\t({ratios['train']:.1%} of total statements, requested was {train_valid_split_ratios['train']:.1%})"
        f"\n\t{len(sids['unused'])} statements unused"
    )
    return sids


def get_reserved_gold_id_clusters(data_parser: "DataParser") -> typing.List[SIDClusterType]:
    """Returns the list of statement identifier clusters reserved for the gold set.

    This relies on the corresponding subset seed and reserved statement count, and assumes that the
    provided data parser allows access to the ENTIRE dataset. This should therefore always return
    the SAME clusters, no matter who is calling this function, and on what machine.

    Note: we remove any "banned" cluster AFTER picking the initial clusters so that the gold set
    can only shrink if we ban more statements it contains (instead of reshuffling it each time).
    """
    assert data_parser is not None and len(data_parser) == gold_set_expected_input_data_parser_size, (
        "unexpected statement count from data parser!"
        f"\n\t(expected {gold_set_expected_input_data_parser_size}, got {len(data_parser)})"
    )
    all_clusters = _get_statement_clusters(data_parser, classif_setup=None, ignore_sids=None)
    assert len(all_clusters) == gold_set_expected_input_cluster_count, (
        "unexpected cluster count from data parser!"
        f"\n\t(expected {gold_set_expected_input_cluster_count}, got {len(all_clusters)})"
    )
    assert len(all_clusters) > gold_set_reserved_size
    picked_cluster_idxs, statement_count = [], 0
    rng = np.random.default_rng(gold_set_seed)
    while statement_count < gold_set_reserved_size:
        remaining_idxs = [idx for idx in range(len(all_clusters)) if idx not in picked_cluster_idxs]
        assert len(remaining_idxs) > 0
        picked_cluster_idx = rng.choice(remaining_idxs)
        picked_cluster_idxs.append(picked_cluster_idx)
        statement_count += len(all_clusters[picked_cluster_idx])
    # after doing the initial picking, drop the clusters that contain banned statements and re-count
    kept_clusters, statement_count = [], 0
    for picked_cluster_idx in picked_cluster_idxs:
        if any([sid in gold_set_banned_statements for sid in all_clusters[picked_cluster_idx]]):
            continue
        statement_count += len(all_clusters[picked_cluster_idx])
        kept_clusters.append(all_clusters[picked_cluster_idx])
    validate_clusters(kept_clusters)
    assert kept_clusters[0] == _gold_set_first_cluster, "unexpected first reserved cluster?"
    logger.info(f"gold set has {statement_count} reserved statements across {len(kept_clusters)} clusters")
    return kept_clusters


def get_validated_gold_id_clusters(
    data_parser: "DataParser",
    reserved_gold_clusters: typing.Optional[typing.List[SIDClusterType]] = None,  # none = go fetch it
) -> typing.List[SIDClusterType]:
    """Returns the list of VALIDATED statement identifier clusters from the gold set.

    This looks at the reserved gold clusters (see `get_reserved_gold_id_clusters`) and returns only
    clusters that contain statements with FULLY VALIDATED annotations. In other words, out of all
    the reserved clusters of statements, this will only return those with the highest quality
    annotations approved by at least one expert.
    """
    expected_branch = qut01.data.dataset_parser.dataset_validated_branch_name
    assert data_parser.dataset.branch == expected_branch, f"bad dataset branch (should be '{expected_branch}')"
    validated_mask = data_parser.dataset.fully_validated_annotations.numpy().flatten().tolist()
    assert len(validated_mask) == len(data_parser.statement_ids)
    validated_sids = [sid for sid, flag in zip(data_parser.statement_ids, validated_mask) if flag]
    if reserved_gold_clusters is None:
        reserved_gold_clusters = get_reserved_gold_id_clusters(data_parser)
    output_clusters = []
    for sid_cluster in reserved_gold_clusters:
        curr_validated_sids = tuple([sid for sid in sid_cluster if sid in validated_sids])
        if not curr_validated_sids:
            continue
        output_clusters.append(curr_validated_sids)
    validate_clusters(output_clusters)
    return output_clusters


def _get_statement_clusters(
    data_parser: "DataParser",
    classif_setup: typing.Optional["ClassifSetupType"],
    ignore_sids: typing.Optional[typing.List[int]],
) -> typing.List[SIDClusterType]:
    """Returns a list of statement clusters (i.e. lists of IDs for potentially similar statements).

    Each 'cluster' contains statements that share at least one trademarks or entity name, and
    that should NOT be split into different train/valid/test subsets, as they might correspond
    to statements made by related subsidiaries or by the same entity across multiple years.
    Those statements tend to have overlapping contents, and are kept together for the split to
    avoid potential leaks.

    The `classif_setup` argument is meant to dictate which statements should be at all
    considered for clustering. If it is not provided, then all statements are considered and
    returned in the clusters. If it is set as `any`, then only statements that potentially possess
    an annotation (of any type) will be used and returned. Otherwise, only statements that
    potentially possess at least one annotation of the specified type will be used and returned.

    The returned clusters are lists of identifiers for statements that should be similar.
    """
    supported_meta_classes = [
        *qut01.data.classif_utils.ANNOT_META_CLASS_NAMES,
        "any",  # this will grab any statement that has at least one annotation of any type
    ]
    if classif_setup is not None and classif_setup not in supported_meta_classes:
        # assume it's a group instead of a meta-group, and convert it using the lookup tables
        assert classif_setup in qut01.data.classif_utils.ANNOT_CLASS_NAMES
        classif_setup = qut01.data.classif_utils.ANNOT_CLASS_NAME_TO_META_CLASS_NAME[classif_setup]
    assert (
        classif_setup is None or classif_setup in supported_meta_classes
    ), f"invalid annotation meta group: {classif_setup}"
    trademarks_array = data_parser.dataset["metadata/Trademarks"].numpy()
    assert trademarks_array.shape == (len(data_parser), 1)
    trademarks_array = trademarks_array.flatten()
    entities_array = data_parser.dataset["metadata/Entities"].numpy()
    assert entities_array.shape == (len(data_parser), 1)
    entities_array = entities_array.flatten()
    if classif_setup is not None:
        potentially_annotated_statement_ids = data_parser.get_potentially_annotated_statement_ids()
        potentially_annotated_statement_idxs = {  # convert ids to indices for easier processing below
            annot_meta_name: [data_parser.statement_ids.index(sid) for sid in sids]
            for annot_meta_name, sids in potentially_annotated_statement_ids.items()
        }
        if classif_setup == "any":
            target_sidxs = {
                sidx  # keep the indices of all statements that might have at least one annotation
                for sidxs in potentially_annotated_statement_idxs.values()
                for sidx in sidxs
            }
        else:
            target_sidxs = set(potentially_annotated_statement_idxs[classif_setup])
    else:
        target_sidxs = range(len(data_parser))
    if ignore_sids is not None:
        target_sidxs = [sidx for sidx in target_sidxs if data_parser.statement_ids[sidx] not in ignore_sids]
    logger.info(
        f"Creating clusters for {len(target_sidxs)} statements "
        f"(target={classif_setup}, {len(ignore_sids) if ignore_sids else 0} ignored)"
    )
    trademark_sidx_tuples, entities_sidx_tuples = [], []
    unclusterable_sidxs = []  # they'll be bunched together
    for sidx in target_sidxs:
        trademarks = trademarks_array[sidx]
        if trademarks not in ["", "n/a"]:
            for trademark in trademarks.split(", "):
                trademark_sidx_tuples.append((trademark, sidx))
        entities = entities_array[sidx]
        if entities in ["", "n/a"]:
            unclusterable_sidxs.append(sidx)  # no entity name? really?
            # note: the (rare) statements that don't define a trademark or entity name will be clustered together
            # (this is not ideal, but there should not be that many...)
        for entity in entities.split(", "):
            entities_sidx_tuples.append((entity, sidx))
    if len(unclusterable_sidxs) > 1:
        unclusterable_sids = [data_parser.statement_ids[sidx] for sidx in unclusterable_sidxs]
        unclusterable_str = ", ".join([str(sid) for sid in unclusterable_sids])
        logger.info(
            f"Found {len(unclusterable_sidxs)} statements without trademarks/entities;"
            f"\n\tthese will be clustered together: {unclusterable_str}"
        )
    sidx_clusters = collections.defaultdict(list)
    for trademark, sidx in trademark_sidx_tuples:
        sidx_clusters[trademark].append(sidx)
    for entity, sidx in entities_sidx_tuples:
        if entity != "Inc.":
            sidx_clusters[entity].append(sidx)
    unmerged_clusters = [clusters for clusters in sidx_clusters.values()]
    output_clusters = _merge_clusters(unmerged_clusters)
    clustered_sidxs = {sidx for cluster in output_clusters for sidx in cluster}
    # finally, add the unclustered target sidxs manually
    for sidx in target_sidxs:
        if sidx not in clustered_sidxs:
            output_clusters.append([sidx])
    # convert the statement INDICES into statement IDENTIFIERS
    output_clusters = [[data_parser.statement_ids[sidx] for sidx in cluster] for cluster in output_clusters]
    # and sort the results (top and sublist level)
    output_clusters.sort()
    for cluster in output_clusters:
        cluster.sort()
    # the clusters should not longer change, so convert them into tuples
    output_clusters = [tuple(c) for c in output_clusters]
    # extra bit of final validation before we go...
    output_ids = [sid for sids in output_clusters for sid in sids]
    assert len(output_ids) <= len(data_parser.dataset), "too many statements?"
    assert len(output_ids) == len(set(output_ids)), "found duplicate statements?"
    logger.info(
        f"Created {len(output_clusters)} statement clusters;"
        f"\n\tbiggest cluster size: {max([len(cluster) for cluster in output_clusters])}"
        f"\n\tmedian cluster size: {int(np.median([len(cluster) for cluster in output_clusters]))}"
        f"\n\tsolo statements: {sum([len(cluster) == 1 for cluster in output_clusters])}"
    )
    return output_clusters


def _merge_clusters(
    clusters: typing.List[typing.List[int]],
) -> typing.List[typing.List[int]]:
    """Fuses a list-of-lists so that all the sublists that share an integer value are merged."""
    parent_map = {}

    def _find_parent(x):
        if x != parent_map[x]:
            parent_map[x] = _find_parent(parent_map[x])
        return parent_map[x]

    def _union(x, y):
        parent_map[_find_parent(x)] = _find_parent(y)

    for cluster in clusters:
        for item in cluster:
            if item not in parent_map:
                parent_map[item] = item
    for cluster in clusters:
        for idx in range(len(cluster) - 1):
            _union(cluster[idx], cluster[idx + 1])
    merged_clusters = collections.defaultdict(list)
    for item in parent_map:
        merged_clusters[_find_parent(item)].append(item)
    return list(merged_clusters.values())


def validate_clusters(clusters: typing.List[SIDClusterType]) -> None:
    """Validates that all the specified clusters are non-empty and all have unique elements."""
    for cluster in clusters:
        assert len(cluster) > 0 and all([isinstance(sid, int) and sid > 0 for sid in cluster])
    sids = [sid for cluster in clusters for sid in cluster]
    assert len(set(sids)) == len(sids)


def validate_split_ratios(split_ratios: typing.Dict[SubsetType, float]) -> None:
    """Validates that the specified split ratios are all OK and that they sum at most to 1.0."""
    for loader_type, split_ratio in split_ratios.items():
        assert loader_type != "unused", "cannot specify 'unused' split ratio!"
        assert loader_type in supported_subset_types, f"invalid data loader type: {loader_type}"
        assert 0.0 <= split_ratio <= 1.0, f"invalid {loader_type} split ratio: {split_ratio}"
    total_ratio = sum(split_ratios.values())
    assert total_ratio < 1.0 or np.isclose(total_ratio, 1.0), f"invalid total split fraction: {total_ratio}"


def validate_split(
    subset_ids: typing.Dict["SubsetType", typing.List[int]],
) -> None:
    """Validates that the statement IDs are OK and that they do not intersect at all."""
    found_ids = []
    for loader_type, ids in subset_ids.items():
        assert loader_type in supported_subset_types, f"invalid data loader type: {loader_type}"
        for idx in ids:
            assert idx not in found_ids, f"duplicate found for statement index: {idx}"
        found_ids.extend(ids)


if __name__ == "__main__":
    qut01.utils.logging.setup_logging_for_analysis_script()
    dataset_ = qut01.data.dataset_parser.get_deeplake_dataset()
    data_parser_ = qut01.data.dataset_parser.DataParser(dataset_)
    reserved_gold_sids_ = get_reserved_gold_id_clusters(data_parser_)
    print(f"{reserved_gold_sids_=}")
