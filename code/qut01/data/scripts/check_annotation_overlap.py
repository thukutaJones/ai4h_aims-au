"""Parses a handful of statements for double annotations, and computes the IAA scores for those."""

import qut01

qut01.utils.logging.setup_logging_for_analysis_script()
dataset = qut01.data.dataset_parser.DataParser(
    dataset_path_or_object=qut01.data.dataset_parser.get_default_deeplake_dataset_path(),
    dataset_branch=qut01.data.dataset_parser.dataset_annotated_branch_name,
)
found_page_token = dataset.info["tokens"]["fitz_text_page_token"]
assert (
    found_page_token == qut01.data.preprocess_utils.fitz_text_page_token
), f"unexpected fitz text page token: {found_page_token}"

target_statement_ids = []
target_annot_types = [
    qut01.data.annotations.classes.Approval,
    qut01.data.annotations.classes.Signature,
    qut01.data.annotations.classes.Criterion1ReportEnt,
    qut01.data.annotations.classes.Criterion2Structure,
    qut01.data.annotations.classes.Criterion2Operations,
    qut01.data.annotations.classes.Criterion2SupplyChains,
    qut01.data.annotations.classes.Criterion3RiskDesc,
    qut01.data.annotations.classes.Criterion4Mitigation,
    qut01.data.annotations.classes.Criterion4Remediation,
    qut01.data.annotations.classes.Criterion5Effect,
    qut01.data.annotations.classes.Criterion6Consult,
]
annotations = qut01.data.annotations.classes.get_annotations(
    dataset=dataset,
    target_statement_ids=target_statement_ids,
    target_annot_types=target_annot_types,
)

if not target_statement_ids:
    target_statement_ids = {a.statement_id for a in annotations}

iaa_scores = {atype: [] for atype in target_annot_types}  # annot_type-to-score-list
for target_sid in target_statement_ids:
    for target_annot_type in target_annot_types:
        target_annots = [a for a in annotations if a.statement_id == target_sid and type(a) is target_annot_type]
        if len(target_annots) < 2:
            continue  # annotations were filtered out due to errors, or not double-annotated
        assert len(target_annots) == 2  # we either single-annotate, or double-annotate, nothing else?
        iaa = qut01.metrics.iaa.compute_inter_annotator_agreement(*target_annots)
        iaa_scores[target_annot_type].append(iaa)

print()

for target_annot_type in target_annot_types:
    if iaa_scores[target_annot_type]:
        scores = iaa_scores[target_annot_type]
        iaa = (sum(scores) / len(scores)) if len(scores) else 0.0
        print(f"average IAA, {target_annot_type.name} ({len(scores)} statements): {iaa:.02f}")

print("all done")
