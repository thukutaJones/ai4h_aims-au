"""Parses raw statements and processed data (from ABBYY FineReader), and repackages via deeplake.

By default, the repackaged data will NOT be compressed under the assumptions that 1) runtime
performance will not be bottlenecked by the disk I/O operations, 2) that not spending extra time
decompressing is beneficial for downstream performance; and 3) we do not care about using roughly
50% more disk space to store this datasets (a few GBs).
"""
import datetime
import io
import json
import pathlib
import typing

import deeplake
import fitz
import lxml.etree  # nosec
import numpy as np
import pandas as pd
import tqdm

import qut01

data_root_path = qut01.utils.config.get_data_root_dir()
raw_statements_root_path = data_root_path / "raw"  # where to find the yearly metadata files + PDFs
statements_csv_path = data_root_path / "statements.20231129.csv"  # global snapshot from the register
processed_data_path = data_root_path / "processed" / "ABBYY-FineReader14.0" / "full-20231127"
target_years = [2020, 2021, 2022, 2023]
expected_metadata_cols = [
    "PeriodStart",
    "PeriodEnd",
    "Type",
    "SubmittedAt",
    "PublishedAt",
    "Countries",
    "Trademarks",
    "AnnualRevenue",
    "Entities",
    "Link",
    "IndustrySectors",
    "OverseasObligations",
]
expected_processed_file_extensions = [
    ".json",
    ".txt",
    ".xml",
]
new_metadata_cols = [
    "LocalLink",
    "PageCount",
    "WordCount",
    "ImageCount",
    "RegisterYear",
]
compress_data = False
verbose_printing = False
fitz_text_page_token = "\n----\n\n//\n\n----\n"  # should be unique enough...

assert data_root_path.is_dir(), f"invalid data root dir: {data_root_path}"
print(f"Parsing global statements metadata from:\n\t{statements_csv_path}")
assert statements_csv_path.is_file(), f"invalid path: {statements_csv_path}"
assert processed_data_path.is_dir(), f"invalid path: {processed_data_path}"
statements_metadata = pd.read_csv(statements_csv_path)
assert all([c in statements_metadata.columns for c in expected_metadata_cols])
assert "StatementId" in statements_metadata.columns and "LocalLink" not in statements_metadata.columns
assert not statements_metadata["StatementId"].duplicated().any()
statements_metadata.set_index("StatementId")
num_statements, num_entities = len(statements_metadata), len(statements_metadata["Entities"].unique())
print(f"\tDone! Found {num_statements} statements for {num_entities} unique entities.")
print(f"\tLatest statement publication date: {statements_metadata['PublishedAt'].max()}")
statements_metadata["LocalLink"] = None  # initializes this new column with a default value
skipped_since_proc_files_missing, seen_ids = [], []

for curr_year in target_years:
    curr_statements_root_path = raw_statements_root_path / f"themodernslaveryregister_{curr_year}"
    curr_statements_csv_path = curr_statements_root_path / f"metadata{curr_year}.csv"
    print(f"Parsing {curr_year} statements metadata from:\n\t{curr_statements_csv_path}")
    curr_statements_metadata = pd.read_csv(curr_statements_csv_path)
    assert all([c in curr_statements_metadata.columns for c in expected_metadata_cols])
    assert "StatementId" not in curr_statements_metadata.columns
    num_statements, num_entities = len(curr_statements_metadata), len(curr_statements_metadata["Entities"].unique())
    expected_unique_cols = ["IDX", "Link"]
    assert all(
        [
            c in curr_statements_metadata.columns and len(curr_statements_metadata[c].unique()) == num_statements
            for c in expected_unique_cols
        ]
    )
    print(f"\tDone! Found {num_statements} statements for {num_entities} unique entities.")
    print(f"\tLatest statement publication date: {curr_statements_metadata['PublishedAt'].max()}")
    progbar = tqdm.tqdm(
        curr_statements_metadata.iterrows(),
        desc=f"Parsing {curr_year} statements",
        total=num_statements,
    )
    partial_matches = []
    print(f"Parsing {curr_year} statements...")
    for _, metadata in progbar:
        metadata_minus_idx = pd.Series(data={c: metadata[c] for c in expected_metadata_cols})
        pdf_name = metadata["IDX"] + ".pdf"
        orig_file_path = curr_statements_root_path / pdf_name
        assert orig_file_path.is_file(), "could not locate statement expected from metadata?"
        proc_file_path = processed_data_path / pdf_name
        if not proc_file_path.is_file():
            skipped_since_proc_files_missing.append(pdf_name)
            continue
        for file_ext in expected_processed_file_extensions:
            assert (
                processed_data_path / (metadata["IDX"] + file_ext)
            ).is_file(), f"missing processed file: {metadata['IDX'] + file_ext}"
        matched_statement_metadata = pd.merge(
            statements_metadata.reset_index().fillna("n/a"),
            metadata.to_frame().T.fillna("n/a"),
            on=expected_metadata_cols,
            how="inner",
        )
        assert len(matched_statement_metadata) <= 1
        if not len(matched_statement_metadata):
            # fallback (partial) matching strategy: look for the statement id as part of the link
            statement_link = metadata["Link"]
            expected_link_prefix = "https://modernslaveryregister.gov.au/statements/"
            assert statement_link.startswith(expected_link_prefix)
            statement_id = int(statement_link.strip("/").split(expected_link_prefix)[-1])
            matched_idxs = statements_metadata.index[statements_metadata["StatementId"] == statement_id]
            assert len(matched_idxs) == 1, f"missing statement {statement_id} from global csv file!"
            if verbose_printing:
                # let's actually print why the mismatch occurred...
                matched_statement = statements_metadata[statements_metadata["StatementId"] == statement_id]
                matched_metadata = pd.Series(
                    data={c: matched_statement[c].item() for c in expected_metadata_cols}
                ).fillna("n/a")
                expected_metadata = metadata_minus_idx.fillna("n/a")
                for c in expected_metadata_cols:
                    if matched_metadata[c] != expected_metadata[c]:
                        print(f"\t\t{pdf_name} metadata mismatch on {c}: {matched_metadata[c]} vs {metadata[c]}")
            partial_matches.append(pdf_name)
        else:
            statement_id = matched_statement_metadata["StatementId"].item()
        assert statement_id not in seen_ids, f"found duplicated statement id: {statement_id}"
        seen_ids.append(statement_id)
        statement_idx = statements_metadata[statements_metadata["StatementId"] == statement_id].index[0]
        statements_metadata.at[statement_idx, "LocalLink"] = proc_file_path
        word_count, img_count = 0, 0
        with fitz.open(proc_file_path) as pdf_reader:  # noqa
            statements_metadata.at[statement_idx, "PageCount"] = pdf_reader.page_count
            for page_idx in range(pdf_reader.page_count):
                page = pdf_reader.load_page(page_idx)
                word_count += len(page.get_text("words"))
                images = page.get_images(full=True)
                img_count += len(images)
        statements_metadata.at[statement_idx, "WordCount"] = word_count
        statements_metadata.at[statement_idx, "ImageCount"] = img_count
        statements_metadata.at[statement_idx, "RegisterYear"] = curr_year
    print(f"\tFound {len(partial_matches)} statement(s) from {curr_year} with partial metadata matches.")
    if len(partial_matches):
        print(f"\t\tfiles with partially matching metadata: {'; '.join(partial_matches)}")

if len(skipped_since_proc_files_missing) > 0:
    print(f"WARNING: Skipped {len(skipped_since_proc_files_missing)} statements!")
    print(f"\tpdfs skipped since processed data missing: {'; '.join(skipped_since_proc_files_missing)}")

print("Exporting statements metadata...")
output_pkl_path, output_csv_path = statements_csv_path.with_suffix(".pkl"), statements_csv_path.with_suffix(".out.csv")
print(f"\t{output_pkl_path}")
statements_metadata.to_pickle(output_pkl_path)
print(f"\t{output_csv_path}")
statements_metadata.to_csv(output_csv_path)

print("Exporting deeplake dataset...")
output_deeplake_path = statements_csv_path.with_suffix(".deeplake")
print(f"\t{output_deeplake_path}")
if output_deeplake_path.exists():
    confirmed = input("Overwrite? (Y/N): ").strip().lower() in ["y", "yes"]
    if not confirmed:
        print("Aborting!")
        exit(-1)


def convert_xml_to_dict(
    element: lxml.etree.ElementBase,
    remove_standard_prefix: bool = True,
) -> typing.Dict[str, typing.Any]:
    """Converts an lxml.etree element and its children into a dict, recursively if needed.

    Element attributes will be stored under the `@attributes` key of their element's dict. If the
    `remove_standard_prefix` is true, then the prefix that seems to be added to all keys across
    all files (`{http://www.loc.gov/standards/alto/ns-v3#}`) will be removed from key names.
    """
    assert isinstance(element, lxml.etree._Element), f"invalid element type: {type(element)}"  # noqa

    def _remove_prefix(s: str) -> str:
        if remove_standard_prefix:
            s = s.replace("{http://www.loc.gov/standards/alto/ns-v3#}", "")
        assert s != "@attributes", "oh hell naw"
        return s

    def _convert(e_: lxml.etree.ElementBase) -> typing.Any:
        if not e_.getchildren():
            return e_.text or ""
        children = {}
        for child in e_:
            key = _remove_prefix(child.tag)
            child_result = _convert(child)
            if key in children:
                if not isinstance(children[key], list):
                    children[key] = [children[key]]
                children[key].append(child_result)
            else:
                children[key] = child_result
        result = {**children, **e_.attrib}
        return result

    return {_remove_prefix(element.tag): _convert(element)}


assert len(seen_ids) == len(set(seen_ids))
assert len(statements_metadata["StatementId"].unique()) == len(statements_metadata)
assert set(seen_ids).issubset(set(statements_metadata["StatementId"]))

dataset = deeplake.empty(output_deeplake_path, overwrite=True)
saved_ids, failed_ids = [], []
with dataset:  # makes sure the export is cached/buffered correctly
    metadata_cols = [*expected_metadata_cols, *new_metadata_cols]
    sample_compression = "unspecified" if not compress_data else "lz4"
    dataset.create_tensor(name="pdf_data", htype="generic", sample_compression=sample_compression)
    dataset.create_tensor(name="fitz/text", htype="text", sample_compression=sample_compression)
    dataset.create_tensor(name="abbyy/json", htype="json", sample_compression=sample_compression)
    dataset.create_tensor(name="abbyy/xml", htype="generic", sample_compression=sample_compression)
    dataset.create_tensor(name="abbyy/xml_as_json", htype="json", sample_compression=sample_compression)
    dataset.create_tensor(name="abbyy/text", htype="text", sample_compression=sample_compression)
    for c in metadata_cols:
        if c.endswith("Count"):
            dataset.create_tensor(name=f"metadata/{c}", htype="generic")
        else:
            dataset.create_tensor(name=f"metadata/{c}", htype="text")
    progbar = tqdm.tqdm(
        sorted(seen_ids),
        desc="Exporting statements",
        total=len(seen_ids),
    )
    for statement_id in progbar:
        statement_metadata = pd.Series(
            data={
                c: statements_metadata[statements_metadata["StatementId"] == statement_id][c].item()
                for c in metadata_cols
            }
        ).fillna("n/a")
        statement_path = pathlib.Path(statement_metadata["LocalLink"])
        try:
            assert statement_path.is_file()
            with open(statement_path, mode="rb") as fd:
                pdf_data_bytes = fd.read()
            with fitz.open(statement_metadata["LocalLink"]) as pdf_reader:  # noqa
                statement_text = []
                for page_idx in range(pdf_reader.page_count):
                    page = pdf_reader.load_page(page_idx)
                    statement_text.append(page.get_text("text"))
            with open(statement_path.with_suffix(".json")) as fd:
                abbyy_json_data = json.load(fd)
            with open(statement_path.with_suffix(".txt")) as fd:
                abbyy_text_data = fd.read()
            with open(statement_path.with_suffix(".xml"), mode="rb") as fd:
                abbyy_xml_data_bytes = fd.read()
            abbyy_xml_data = lxml.etree.parse(io.BytesIO(abbyy_xml_data_bytes))  # nosec
            abbyy_xml_data = convert_xml_to_dict(abbyy_xml_data.getroot())
            output_data = {
                "pdf_data": np.frombuffer(pdf_data_bytes, dtype=np.uint8),
                "fitz/text": fitz_text_page_token.join(statement_text),
                "abbyy/json": abbyy_json_data,
                "abbyy/xml": np.frombuffer(abbyy_xml_data_bytes, dtype=np.uint8),
                "abbyy/xml_as_json": abbyy_xml_data,
                "abbyy/text": abbyy_text_data,
            }
            output_metadata = {
                f"metadata/{c}": str(statement_metadata[c]) if not c.endswith("Count") else int(statement_metadata[c])
                for c in metadata_cols
            }
            dataset.append({**output_data, **output_metadata})
            saved_ids.append(statement_id)
        except Exception as e:
            print(f"\tFAILED to process statement: {statement_id}\n\t\treason: {e}")
            failed_ids.append(statement_id)

    dataset.create_tensor(name="statement_id", htype="generic", dtype=np.int32)
    dataset.statement_id.extend(saved_ids)

    dataset.info.update(
        {
            "created_on": datetime.datetime.now().isoformat(),
            "register_csv_path": str(statements_csv_path),
            "processed_data_path": str(processed_data_path),
            "target_years": target_years,
            "failed_statement_ids": failed_ids,
            "repo_version": qut01.__version__,
            "tokens": {
                "fitz_text_page_token": fitz_text_page_token,
            },
        }
    )

print("Committing changes to dataset...")
dataset.commit("origin")  # will leave the dataset on branch 'main' with all changes committed
print("All done.")
