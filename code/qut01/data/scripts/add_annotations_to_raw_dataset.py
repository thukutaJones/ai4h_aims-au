"""Adds the provided annotations (in their original CSV format) into the dataset.

Will also optionally add validated annotations done by the annotation tool (and locally stored in
pickle format) to the dataset.
"""
import datetime

import numpy as np
import pandas as pd
import tqdm

import qut01

data_root_path = qut01.utils.config.get_data_root_dir()
valid_annot_root_path = qut01.utils.config.get_data_root_dir() / "validated_data"
statements_dataset_path = qut01.data.dataset_parser.get_default_deeplake_dataset_path()
annot_approvsignc1_meta_prefix = qut01.data.classif_utils.ANNOT_APPROVSIGNC1_META_CLASS_NAME
annot_c2c3c4c5c6_meta_prefix = qut01.data.classif_utils.ANNOT_C2C3C4C5C6_META_CLASS_NAME
annotations_map = {  # annotation type prefix to annotation csv path list
    annot_approvsignc1_meta_prefix: [
        data_root_path / "gt" / annot_approvsignc1_meta_prefix / "2024-01-15.csv",
    ],
    annot_c2c3c4c5c6_meta_prefix: [
        (data_root_path / "gt" / annot_c2c3c4c5c6_meta_prefix / "2024-02-29-600samples.csv"),
        (data_root_path / "gt" / annot_c2c3c4c5c6_meta_prefix / "2024-04-01-1447samples.csv"),
        (data_root_path / "gt" / annot_c2c3c4c5c6_meta_prefix / "2024-05-02-1447samples-fixed.csv"),
        (data_root_path / "gt" / annot_c2c3c4c5c6_meta_prefix / "2024-05-31-1164samples.csv"),
    ],
}
dataset = qut01.data.dataset_parser.get_deeplake_dataset(
    statements_dataset_path,
    checkout_branch="main",  # go back to the initial dataset state without annotations
    read_only=False,
)
dataset_sids = dataset.statement_id.numpy().flatten().tolist()
failed_sids = dataset.info.failed_statement_ids
assert len(set(dataset_sids)) == len(dataset_sids)
print(f"Will parse and store annotation data for at most {len(dataset_sids)} statements")

annot_branch = qut01.data.dataset_parser.dataset_annotated_branch_name
valid_annot_branch = qut01.data.dataset_parser.dataset_validated_branch_name
if annot_branch in dataset.branches:
    confirmed = input(f"Overwrite {annot_branch} branch? (Y/N): ").strip().lower() in ["y", "yes"]
    if not confirmed:
        print("Aborting!")
        exit(-1)
    if valid_annot_branch in dataset.branches:
        # we need to delete the derived branch before the base one (otherwise it will crash)
        confirmed = input(f"Overwrite {valid_annot_branch} branch? (Y/N): ").strip().lower() in ["y", "yes"]
        if not confirmed:
            print("Aborting!")
            exit(-1)
        dataset.delete_branch(valid_annot_branch)
    dataset.delete_branch(annot_branch)
dataset.checkout(annot_branch, create=True)
dataset.info["updated_on"] = datetime.datetime.now().isoformat()
dataset.info["repo_version"] = qut01.__version__
dataset.info["annotation_count"] = {annot_name: None for annot_name in annotations_map}

ignored_annot_col_names = ["statement_id", "PDF URL", "Statement score"]
stored_annotations_sids, stored_annotations_data = {}, {}
for annotations_type, annotations_csv_paths in annotations_map.items():
    stored_annotations_sids[annotations_type] = []  # to make sure we don't overlap statements across CSVs
    stored_annotations_data[annotations_type] = {}  # to store maps of annotation column names -> values
    annotation_count = None
    for annotations_csv_path in annotations_csv_paths:
        print(f"parsing {annotations_type} annotations from: {annotations_csv_path}")
        annotations = pd.read_csv(annotations_csv_path, dtype=str).fillna("")
        assert len(annotations) > 0, f"invalid annotation file: {annotations_csv_path}"
        print(f"annotations shape: {annotations.shape}")
        print("annotations columns:")
        for col_name, col_vals in annotations.items():
            unique_vals = col_vals.unique()
            unique_val_count = len(unique_vals)
            print(f"\t{col_name}: {unique_val_count} unique values")
            if unique_val_count < 6:
                print(f"\t\t{sorted([v for v in unique_vals])}")

        assert "statement_id" in annotations, "missing mandatory statement_id column in annotations csv"
        # note: statement ids should always be numbers, but they might contain pesky commas (e.g. "1,323")
        annotations["statement_id"] = annotations["statement_id"].str.replace(",", "").astype(int)
        annotated_sids, unique_counts = np.unique(annotations["statement_id"], return_counts=True)
        assert len(np.unique(unique_counts)) == 1, "statements not all annotated the same number of times?"
        if annotation_count is None:
            annotation_count = int(unique_counts[0])
        assert annotation_count == unique_counts[0], f"mismatched annot counts across {annotations_type} CSVs"
        annotated_sids = annotated_sids.tolist()
        assert all([sid not in stored_annotations_sids[annotations_type] for sid in annotated_sids])
        assert all([(sid in dataset_sids) or (sid in failed_sids) for sid in annotated_sids])
        kept_sids = [sid for sid in annotated_sids if sid in dataset_sids]
        print(f"keeping annotations for {len(kept_sids)} statements ({annotation_count} per statement)")
        stored_annotations_sids[annotations_type].extend(kept_sids)
        drop_sids = [sid for sid in annotated_sids if sid in failed_sids]
        print(f"dropping annotations for {len(drop_sids)} statements (PDF text export failed)")
        if dataset.info["annotation_count"][annotations_type] is None:
            dataset.info["annotation_count"][annotations_type] = annotation_count
        assert dataset.info["annotation_count"][annotations_type] == annotation_count
        annot_col_names = [cname for cname in annotations.columns if cname not in ignored_annot_col_names]
        if len(stored_annotations_data[annotations_type]) == 0:
            for annot_col_name in annot_col_names:
                stored_annotations_data[annotations_type][annot_col_name] = {}
        assert set(annot_col_names) == set(
            stored_annotations_data[annotations_type].keys()
        ), f"mismatch between annotation column names across {annotations_type} CSVs"
        for annot_col_name in annot_col_names:
            # add the annotated values directly (without cleanups) with an empty string as the fill value
            if annotation_count == 1:
                sid_to_val_map = annotations.set_index("statement_id")[annot_col_name].to_dict()
            else:
                relevant_sub_df = annotations[["statement_id", annot_col_name]]
                sid_to_val_map = relevant_sub_df.groupby("statement_id")[annot_col_name].apply(list).to_dict()
            for sid in kept_sids:
                assert (
                    sid not in stored_annotations_data[annotations_type][annot_col_name]
                ), f"annotation {annot_col_name} for sid {sid} was already provided by another CSV file?"
                stored_annotations_data[annotations_type][annot_col_name][sid] = sid_to_val_map[sid]
    pbar = tqdm.tqdm(stored_annotations_data[annotations_type].items(), desc=f"creating {annotations_type} tensors")
    for annot_col_name, annot_sid_to_val_map in pbar:
        tensor_name = f"annotations/{annotations_type}/{annot_col_name}"
        if annotation_count == 1:
            dataset.create_tensor(tensor_name, htype="text")
            tensor_values = [annot_sid_to_val_map.get(sid, "") for sid in dataset_sids]
        else:
            dataset.create_tensor(tensor_name, htype="sequence[text]")
            tensor_values = [annot_sid_to_val_map.get(sid, [""] * annotation_count) for sid in dataset_sids]
        dataset[tensor_name].extend(tensor_values)
    # finally, fill the binary annotation mask (true/false) for this annotation type
    annotated_mask_name = f"annotations/{annotations_type}/annotated"  # corresponds to _ANNOT_XXXX_VALID define
    dataset.create_tensor(annotated_mask_name, htype="binary_mask")  # will be updated at the end
    mask = [sid in stored_annotations_sids[annotations_type] for sid in dataset_sids]
    dataset[annotated_mask_name].extend(mask)

print("Committing changes to branch...")
added_annot_file_paths = [str(p) for paths in annotations_map.values() for p in paths]
annotations_str = "\n\t".join(added_annot_file_paths)
commit_msg = f"annotation CSVs added:\n\t{annotations_str}"
print(f"{commit_msg}\n\n")
dataset.commit(commit_msg)

potentially_validated_statement_ids = []
if valid_annot_root_path.exists():
    for child in valid_annot_root_path.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        potential_statement_id = int(child.name)
        if potential_statement_id not in dataset_sids:
            continue
        has_pickles = any([subchild.suffix == ".pkl" for subchild in child.iterdir()])
        if has_pickles:
            potentially_validated_statement_ids.append(potential_statement_id)

if potentially_validated_statement_ids:
    print(
        "Will parse and store validated annotation data "
        f"for at most {len(potentially_validated_statement_ids)} statements"
    )
    # now, prep the 'validated' annotation branch to store pickle files generated by the validator app
    qut01.data.dataset_parser.prepare_dataset_for_validation(
        dataset=dataset,
        restart_from_raw_annotations=True,
        bypass_user_confirmation=False,
    )
    data_parser = qut01.data.dataset_parser.DataParser(
        dataset,
        pickle_dir_path=None,  # will look for pickle files in the default directory
        dump_found_validated_annots_as_pickles=False,
        load_validated_annots_from_pickles=True,
        use_processed_data_cache=False,
    )
    fully_validated_statement_ids, processed_statement_ids = [], []
    for statement_id in tqdm.tqdm(potentially_validated_statement_ids, desc="Parsing validated annotations"):
        statement_data = data_parser.get_processed_data(data_parser.statement_ids.index(statement_id))
        if statement_data.is_fully_validated:
            fully_validated_statement_ids.append(statement_id)
        # re-export the validated annotations (and only those, the rest are already in the dataset)
        for annot in statement_data.annotations:
            if not annot.is_validated:
                continue
            data_parser.update_tensors(annot.statement_id, annot.create_tensor_data())
            if statement_id not in processed_statement_ids:
                processed_statement_ids.append(statement_id)
    print(f"Found {len(fully_validated_statement_ids)} statements with fully validated annotations.")
    dataset.create_tensor("fully_validated_annotations", htype="binary_mask")
    mask = [sid in fully_validated_statement_ids for sid in dataset_sids]
    dataset["fully_validated_annotations"].extend(mask)

    print("Committing changes to branch...")
    commit_str = "\n\t".join(
        [
            f"{sid} (fully validated)" if sid in fully_validated_statement_ids else f"{sid}"
            for sid in processed_statement_ids
        ]
    )
    commit_msg = f"Added validated annotation(s) for {len(processed_statement_ids)} statement(s):\n\t{commit_str}"
    print(f"{commit_msg}\n\n")
    dataset.commit(commit_msg)

print("All done.")
