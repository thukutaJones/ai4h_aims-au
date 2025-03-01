import argparse

import pandas as pd


def add_context(
    df,
    context_length,
    left_context_boundary_token="<|start_header_id|>",
    right_context_boundary_token="<|end_header_id|>",
):
    """
    Adds context to each sentence based on the context_length.

    Args:
        df (pd.DataFrame): DataFrame with 'sentence' and 'statement_id' columns.
        context_length (int): Total number of words for the context. Half for left and half for right.
        left_context_boundary_token: "<|start_header_id|>"
        right_context_boundary_token: "<|end_header_id|>"

    Returns:
        pd.DataFrame: New DataFrame with 'sentence', 'context', and 'statement_id' columns,
                      preserving the original order of sentences.
    """
    context_half = max((context_length // 2) - 1, 0)

    # Initialize a new column for the context
    df["context"] = ""

    # Group by statement_id to process each group separately
    grouped = df.groupby("statement_id")

    result = []

    for statement_id, group in grouped:
        # Iterate through each sentence in the group
        for i, row in group.iterrows():
            if context_half == 0:
                df.loc[row.name, "context"] = row["sentence"].strip()

            else:
                # Get left context
                left_context = " ".join(". ".join(group.loc[: i - 1, "sentence"].tolist()).split()[-context_half:])

                # Get right context
                right_context = " ".join(". ".join(group.loc[i + 1 :, "sentence"].tolist()).split()[:context_half])

                # Update the context column in the original DataFrame
                df.loc[
                    row.name, "context"
                ] = f"{left_context}.  {left_context_boundary_token} {row['sentence']} {right_context_boundary_token}. {right_context}".strip()

    return df


def main():
    """
    Script to generate a csv file by adding context to each sentence in the input csv file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context_length",
        type=int,
        help="Context length for each sentence such that context_length/2 words are added to left and right of the target sentence.",
    )

    parser.add_argument(
        "--left_context_boundary_token",
        type=str,
        default="<|start_header_id|>",
        help="Token to indicate the start of the target sentence.",
    )

    parser.add_argument(
        "--right_context_boundary_token",
        type=str,
        default="<|end_header_id|>",
        help="Token to indicate the end of the target sentence.",
    )

    parser.add_argument(
        "--data_filename",
        type=str,
        help="CSV file containing the sentences from the statement files",
    )

    parser.add_argument(
        "--output_filename",
        type=str,
        help="CSV file where the results will be saved",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.data_filename)
    df["sentence"] = df["sentence"].astype(str)  # Convert all sentences to strings
    result_df = add_context(
        df, args.context_length, args.left_context_boundary_token, args.right_context_boundary_token
    )
    result_df.to_csv(args.output_filename, index=False)


if __name__ == "__main__":
    main()
