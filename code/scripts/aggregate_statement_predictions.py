import argparse

import pandas as pd
import torch


def main(csv_filepath, output_filepath):
    df = pd.read_csv(csv_filepath)

    aggregated_dict = {
        "statement_id": [],
        "targets": [],
        "predictions": [],
    }

    # Process each statement separately
    for statement_id, group in df.groupby("statement_id"):
        aggregated_targets = torch.zeros(11, dtype=torch.int)
        aggregated_predictions = torch.zeros(11)

        # Iterate through each sentence in the statement and aggregate the targets and predictions
        for _, row in group.iterrows():
            targets = torch.tensor(eval(row["targets"]), dtype=torch.int)
            predictions = torch.tensor(eval(row["predictions"]), dtype=torch.float32)

            # Aggregate using max
            aggregated_targets = torch.maximum(aggregated_targets, targets)
            aggregated_predictions = torch.maximum(aggregated_predictions, predictions)

        aggregated_dict["statement_id"].append(statement_id)
        aggregated_dict["targets"].append(aggregated_targets.tolist())
        aggregated_dict["predictions"].append(aggregated_predictions.tolist())

    aggregated_df = pd.DataFrame.from_dict(aggregated_dict)
    aggregated_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregates predictions with respect to statement_id",
        epilog="""
        The provided csv file must contain a column 'targets' with the ground truth labels,
        a column 'predictions' with the label prediction probabilities,
        and a column 'statement_id' with the statement ids.
        """,
    )
    parser.add_argument(
        "--input_csv_filepath",
        help="csv file containing classification predictions and ground truth labels",
        required=True,
    )

    parser.add_argument("--output_csv_filepath", help="file where the aggregated results will be saved", required=True)

    args = vars(parser.parse_args())
    main(csv_filepath=args["input_csv_filepath"], output_filepath=args["output_csv_filepath"])
