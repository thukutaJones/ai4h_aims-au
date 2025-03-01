import argparse
import datetime
import os
import pathlib
import shlex
import subprocess
import time
from pathlib import Path

import deeplake
import fitz
import numpy as np
import pandas as pd
import requests
import tqdm
from bs4 import BeautifulSoup

import qut01


def extract_download_link(link, session):
    """
    Extract the href item associated with the Download button from the link.
    Uses a requests.Session for better performance.
    """
    try:
        response = session.get(link, timeout=10)  # Timeout added for robustness
        soup = BeautifulSoup(response.text, "html.parser")
        download_button = soup.find("a", string="Download")
        if download_button and "href" in download_button.attrs:
            return "https://modernslaveryregister.gov.au" + download_button["href"]
    except Exception as e:
        # Log or handle specific errors if necessary
        return None
    return None


def main():
    """
    Script to generate a csv file for sentence annotation using UK/CA/AU data
    Downloads the pdf files and parses them
    Optionally creates a deeplake
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata_filename",
        type=str,
        help="Path to the CSV file containing the PDF file metadata",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save the generated CSV file and deeplake dataset",
    )
    parser.add_argument(
        "--statement_id_offset",
        type=int,
        default=0,
        help="If statement ids are not present in the metadata, ids are generated starting from this number",
    )
    parser.add_argument(
        "--create_deeplake",
        action="store_true",
        help="Set this flag to create and populate a deeplake dataset",
    )

    args = parser.parse_args()

    # Create a temporary folder to store pdf files
    temp_folder = Path(args.output_dir, "tmp/")
    os.makedirs(temp_folder, exist_ok=True)

    # Download all the pdf documents from the csv file shared by Adriana
    statements_csv_path = args.metadata_filename
    statements_metadata = pd.read_csv(statements_csv_path)
    print(statements_metadata.head())

    # If the metadata has Statement URL, copy it to the Download Link column
    if "Statement URL" in statements_metadata:
        statements_metadata.rename(columns={"Statement URL": "Download Link"}, inplace=True)

    if "Download Link" not in statements_metadata:
        if "Link" in statements_metadata:
            print("Extracting download links...")
            with requests.Session() as session:  # Reuse session for performance
                tqdm.tqdm.pandas(desc="Extracting download links")
                statements_metadata["Download Link"] = statements_metadata["Link"].progress_apply(
                    lambda link: extract_download_link(link, session)
                )
            print("Extracted download links")

    # Save the updated statements_metadata to a new metadata file in the output directory
    statements_metadata.to_csv(f"{args.output_dir}/metadata.csv", index=False)

    # If metadata has StatementId rename it to Statement ID
    if "StatementId" in statements_metadata:
        statements_metadata.rename(columns={"StatementId": "Statement ID"}, inplace=True)
        print("Renamed StatementId to Statement ID")
        print(statements_metadata.head())

    # Create statement IDs in metadata table if required
    if "Statement ID" not in statements_metadata:
        statements_metadata["Statement ID"] = statements_metadata.apply(
            lambda row: "SID_" + str(row.name + args.statement_id_offset), axis=1
        )
        statements_metadata.to_csv(args.metadata_filename)
    statements_metadata = statements_metadata.dropna(subset=["Statement ID"])

    # Remove statements with empty download links
    statements_metadata = statements_metadata.dropna(subset=["Download Link"])

    # Add a column to store the local path of the downloaded pdf files
    statements_metadata["LocalLink"] = ""

    # Download the pdf files based on the URLs in the csv file. Loop only through unique Statement IDs
    for i, statement_id in tqdm.tqdm(
        enumerate(statements_metadata["Statement ID"].unique()),
        desc="Downloading statements",
        total=len(statements_metadata["Statement ID"].unique()),
    ):
        # Get the URL of the pdf file for the current statement ID
        url = statements_metadata.loc[statements_metadata["Statement ID"] == statement_id, "Download Link"].values[0]
        pdf_file = temp_folder / Path(str(statement_id) + ".pdf")
        # Update pdf_file in the rows which match the statement ID
        statements_metadata.loc[statements_metadata["Statement ID"] == statement_id, "LocalLink"] = pdf_file

        # Define the command and the number of retries
        command = ["wget", "-O", pdf_file, url]
        max_retries = 3
        attempts = 0

        # Retry logic
        while attempts < max_retries:
            try:
                print(f"Attempt {attempts + 1} of {max_retries}")
                subprocess.run(command, check=True)
                # Add a timer to avoid getting blocked by the server
                time.sleep(2)
                print("Download successful!")
                break
            except subprocess.CalledProcessError as e:
                attempts += 1
                print(f"Command failed with exit code {e.returncode}. Attempt {attempts} failed.")
                if attempts < max_retries:
                    print("Retrying...")
                    time.sleep(2)  # Wait 2 seconds before retrying
                else:
                    print("All attempts failed. Exiting.")

    # Check if all the files have been downloaded by counting pdf files
    num_files = len(list(temp_folder.glob("*.pdf")))

    print(f"Downloaded {num_files} files to the download folder")

    # Count number of unique statement ids
    num_unique_statement_ids = len(statements_metadata["Statement ID"].unique())
    print(f"Number of unique valid statement ids: {num_unique_statement_ids}")

    # Extract all sentences and put in a csv file
    all_sentences = []
    statement_ids = []
    failed_statements = []
    successful_statements = []

    # This will take a long time to process with a very large csv file.
    df = extract_sentences_from_pdfs(
        statements_metadata, failed_statements, successful_statements, None, args.output_dir
    )

    print(
        "Could not extract sentences from the following statements:",
        set(failed_statements),
    )
    # Filter duplicate sentences within each statement
    df = df[~df[["sentence", "statement_id"]].duplicated(keep=False)]

    df["targets"] = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * len(df)

    print(df.head())
    df.to_csv(f"{args.output_dir}/processed_statement_data.csv", index=False)

    # Save failed statements to a csv file
    failed_statements_df = pd.DataFrame(failed_statements, columns=["Statement ID"])
    failed_statements_df.to_csv(f"{args.output_dir}/failed_statements.csv", index=False)

    if not args.create_deeplake:
        return

    target_years = [2024]

    output_deeplake_path = args.output_dir / Path(statements_csv_path).with_suffix(".deeplake")

    expected_metadata_cols = [
        "Company",
        "Metric",
        "Year",
        "Source Page",
        "Original Source",
        "Comments",
        "ISIN",
        "Download Link",
        "Answer Page",
    ]

    new_metadata_cols = [
        "LocalLink",
        "PageCount",
        "WordCount",
        "ImageCount",
    ]
    compress_data = False
    verbose_printing = False
    fitz_text_page_token = "\n----\n\n//\n\n----\n"
    metadata_cols = [*expected_metadata_cols, *new_metadata_cols]
    sample_compression = "unspecified" if not compress_data else "lz4"
    # Add add metadata_cols in statement_metadata if they are missing
    for c in metadata_cols:
        if c not in statements_metadata.columns:
            if c.endswith("Count"):
                statements_metadata[c] = 0  # default to 0
            else:
                statements_metadata[c] = np.nan

    dataset = deeplake.empty(output_deeplake_path, overwrite=True)

    populate_deeplake_dataset(
        statements_csv_path,
        statements_metadata,
        target_years,
        fitz_text_page_token,
        metadata_cols,
        sample_compression,
        dataset,
        successful_statements,
    )

    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    print(f"Deeplake dataset saved at: {output_deeplake_path}")

    print("Verifying the deeplake dataset...")
    # Verify the deeplake dataset by loading it and printing tensors
    verify_deeplake_dataset(output_deeplake_path)


def verify_deeplake_dataset(output_deeplake_path):
    """Load the deeplake object and print tensors."""
    dataset = deeplake.load(output_deeplake_path)

    # List all tensors in the dataset
    print("Available tensors in the dataset:")
    for tensor_name in dataset.tensors:
        print(f"Tensor Name: {tensor_name}")
        tensor = dataset.tensors[tensor_name]
        print(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}")
        print(f"Sample data from {tensor_name}: {str(tensor[:1].numpy(aslist=True))}")


def populate_deeplake_dataset(
    statements_csv_path,
    statements_metadata,
    target_years,
    fitz_text_page_token,
    metadata_cols,
    sample_compression,
    dataset,
    successful_statements,
):
    """Populate the deeplake dataset with the processed data from the pdf files.
    Args:
        statements_csv_path (str): Path to the CSV file containing the PDF file metadata
        statements_metadata (pd.DataFrame): DataFrame containing the metadata of the PDF files
        target_years (list): List of target years
        fitz_text_page_token (str): Token to separate text from different pages
        metadata_cols (list): List of metadata columns
        sample_compression (str): Compression type for the dataset
        dataset (deeplake.Dataset): Deeplake dataset object
        successful_statements (list): List of successful statement IDs
    Returns:
        None
    """
    # Remove failed_ids from saved_ids
    saved_ids = list(set(successful_statements))
    failed_ids = []

    try:
        with dataset:
            try:
                dataset.create_tensor(
                    name="pdf_data",
                    htype="generic",
                    sample_compression=sample_compression,
                    exist_ok=True,
                )
                dataset.create_tensor(
                    name="fitz/text",
                    htype="text",
                    sample_compression=sample_compression,
                    exist_ok=True,
                )
                dataset.create_tensor(
                    name="statement_id",
                    htype="text",
                    sample_compression=sample_compression,
                    exist_ok=True,
                )
                for c in metadata_cols:
                    tensor_name = f"metadata/{c}"
                    htype = "generic" if c.endswith("Count") else "text"
                    dataset.create_tensor(name=tensor_name, htype=htype, exist_ok=True)
            except Exception as e:
                print(f"Failed to create tensors: {e}")
                raise

            # Process statements
            for statement_id in tqdm.tqdm(saved_ids, desc="Exporting statements", total=len(saved_ids)):
                try:
                    # Validate file and metadata
                    statement_path = pathlib.Path(
                        statements_metadata.loc[
                            statements_metadata["Statement ID"] == statement_id,
                            "LocalLink",
                        ].iloc[0]
                    )
                    if not statement_path.is_file():
                        raise FileNotFoundError(f"File not found: {statement_path}")

                    # Read PDF data
                    with open(statement_path, mode="rb") as fd:
                        pdf_data_bytes = fd.read()

                    # Read PDF text and metadata
                    with fitz.open(statement_path) as pdf_reader:
                        statement_text = [
                            pdf_reader.load_page(i).get_text("text") for i in range(pdf_reader.page_count)
                        ]
                        word_count = sum(len(page.get_text("words")) for page in pdf_reader)
                        img_count = sum(len(page.get_images(full=True)) for page in pdf_reader)

                        # Update statements_metadata dataframe
                        statements_metadata.loc[
                            statements_metadata["Statement ID"] == statement_id,
                            "PageCount",
                        ] = pdf_reader.page_count
                        statements_metadata.loc[
                            statements_metadata["Statement ID"] == statement_id,
                            "WordCount",
                        ] = word_count
                        statements_metadata.loc[
                            statements_metadata["Statement ID"] == statement_id,
                            "ImageCount",
                        ] = img_count

                    # Prepare data for appending
                    output_data = {
                        "pdf_data": np.frombuffer(pdf_data_bytes, dtype=np.uint8),
                        "fitz/text": fitz_text_page_token.join(statement_text),
                        "statement_id": statement_id,
                    }

                    statement_metadata = statements_metadata.loc[
                        statements_metadata["Statement ID"] == statement_id
                    ].iloc[0]

                    output_metadata = {f"metadata/{c}": statement_metadata[c] for c in metadata_cols}

                    # Append data to the dataset
                    dataset.append({**output_data, **output_metadata})

                except Exception as e:
                    print(f"FAILED to process statement: {statement_id}\n\tReason: {e}")
                    failed_ids.append(statement_id)

            # Update additional tensors
            dataset.info.update(
                {
                    "created_on": datetime.datetime.now().isoformat(),
                    "register_csv_path": str(statements_csv_path),
                    "target_years": target_years,
                    "failed_statement_ids": list(failed_ids),
                    "repo_version": qut01.__version__,
                    "tokens": {"fitz_text_page_token": fitz_text_page_token},
                }
            )

        # Commit changes
        print("Committing changes to the dataset...")
        dataset.commit("origin")
    except Exception as e:
        print(f"Critical error: {e}")
        print("Rolling back changes...")


def extract_sentences_from_pdfs(statements_metadata, failed_statements, successful_statements, df, output_dir):
    """Extract sentences from the PDF files and store in a DataFrame.
    Args:
        statements_metadata (pd.DataFrame): DataFrame containing the metadata of the PDF files
        failed_statements (list): List of failed statement IDs
        successful_statements (list): List of successful statement IDs
        df (pd.DataFrame): DataFrame to store the extracted sentences
    Returns:
        pd.DataFrame: DataFrame containing the extracted sentences
    """
    print("Extracting sentences from PDFs...")
    unique_statement_ids = statements_metadata["Statement ID"].unique()
    for statement_id in tqdm.tqdm(
        unique_statement_ids,
        desc="Extracting sentences",
        total=len(unique_statement_ids),
    ):
        # we will open this PDF and extract its raw text using PyMuPDF (fitz), an open source library
        local_link = statements_metadata.loc[statements_metadata["Statement ID"] == statement_id, "LocalLink"].values[0]
        # Download the pdf file again if fitz.open throws an error
        with fitz.open(local_link) as pdf_reader:
            statement_text = []
            for page_idx in range(pdf_reader.page_count):
                page = pdf_reader.load_page(page_idx)
                statement_text.append(page.get_text("text"))
            statement_text = "\n".join(statement_text)
        # note: of course, if the above PDF only contains embedded text, you won't get anything here
        if len(statement_text) == 0:
            failed_statements.append(statement_id)
            continue
        assert len(statement_text) != 0
        successful_statements.append(statement_id)
        statement_processed_data = qut01.data.statement_utils.StatementProcessedData.create(
            statement_tensor_data={"fitz/text": statement_text},
            load_annotations=False,  # assume none exist
        )
        if df is None:
            df = pd.DataFrame(
                {
                    "sentence": statement_processed_data.sentences,
                    "statement_id": statement_id,
                }
            )
        else:
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "sentence": statement_processed_data.sentences,
                            "statement_id": statement_id,
                        }
                    ),
                ]
            )

        # Save df to a csv file after every 3000 statements
        if len(df) % 3000 == 0:
            df.to_csv(f"{output_dir}/processed_statement_data_{len(df)}.csv", index=False)
    return df


if __name__ == "__main__":
    main()
