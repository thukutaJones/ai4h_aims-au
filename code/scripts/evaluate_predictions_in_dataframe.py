import argparse

import pandas as pd
import torch
import torchmetrics
from torch.utils.data import DataLoader, Dataset


def generate_all_metrics(num_labels=11, threshold=0.5, label_ignore_index=-1):
    """
    Dynamically generates a dictionary of metrics for multi-label classification using torchmetrics.

    Args:
        num_labels (int, optional): The number of labels in the classification task. Default is 11.
        threshold (float, optional): The decision threshold for classification. Default is 0.5.
        label_ignore_index (int, optional): Index to ignore in the label tensor. Default is -1.

    Returns:
        dict: A dictionary with metric names as keys and their corresponding torchmetrics objects as values.
    """
    common_kwargs = {
        "task": "multilabel",
        "num_labels": num_labels,
        "threshold": threshold,
        "ignore_index": label_ignore_index,
    }

    # Define metric types and their additional arguments
    metric_types = {
        "accuracy": torchmetrics.classification.Accuracy,
        "f1": torchmetrics.classification.F1Score,
        "precision": torchmetrics.classification.Precision,
        "recall": torchmetrics.classification.Recall,
    }

    # Define averages for each metric type
    averages = {
        "macro": "macro",  # Macro average
        "label": None,  # No averaging (per-label metrics)
    }

    metrics = {}

    # Loop through each metric type and average
    for metric_name, metric_class in metric_types.items():
        for avg_name, avg_value in averages.items():
            # Create metric key
            key = f"{metric_name}-{avg_name}" if avg_name else metric_name
            # Initialize the metric with relevant kwargs
            metrics[key] = metric_class(**common_kwargs, average=avg_value)

    return metrics


class ClassificationDataset(Dataset):
    """
    A custom Dataset class for metric calculation.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the target classes and prediction probabilities.
    """

    def __init__(self, df):
        """
        Initializes the ClassificationDataset with target classes and prediction probabilities.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing the target classes and prediction probabilities.
        """
        self.targets = torch.tensor(df["targets"].apply(eval).tolist(), dtype=torch.float32)
        self.predictions = torch.tensor(df["predictions"].apply(eval).tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.targets[idx], self.predictions[idx]


def main(csv_filepath, output_filepath):
    df = pd.read_csv(csv_filepath)
    metrics_dict = generate_all_metrics()

    # Prepare DataLoader
    dataset = ClassificationDataset(df)
    dataloader = DataLoader(dataset, batch_size=32)

    # Initialize accumulators for metrics
    accumulated_metrics = {key: torchmetrics.MetricCollection(metrics) for key, metrics in metrics_dict.items()}

    for batch_targets, batch_predictions in dataloader:
        for _, metric in accumulated_metrics.items():
            metric.update(batch_predictions, batch_targets)

    # Compute metrics
    metrics = {key: next(iter(metric.compute().values())).tolist() for key, metric in accumulated_metrics.items()}

    # Arrange results into a df
    metric_values = [
        ["f1", metrics["f1-macro"]] + metrics["f1-label"],
        ["accuracy", metrics["accuracy-macro"]] + metrics["accuracy-label"],
        ["precision", metrics["precision-macro"]] + metrics["precision-label"],
        ["recall", metrics["recall-macro"]] + metrics["recall-label"],
    ]
    metrics_df = pd.DataFrame(
        metric_values,
        columns=[
            "metric",
            "macro-average",
            "approval",
            "signature",
            "c1 (reporting entity)",
            "c2 (structure)",
            "c2 (operations)",
            "c2 (supply chains)",
            "c3 (risk description)",
            "c4 (risk mitigation)",
            "c4 (remediation)",
            "c5 (effectiveness)",
            "c6 (consultation)",
        ],
    )

    print(metrics_df)
    metrics_df.to_csv(output_filepath, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes evaluation metrics for the sentence relevance classification task.",
        epilog="""
        The provided csv file must contain a column 'targets' with the ground truth labels
        and a column 'predictions' with the label prediction probabilities.
        """,
    )
    parser.add_argument(
        "--input_csv_filepath",
        help="csv file containing classification predictions and ground truth labels",
        required=True,
    )

    parser.add_argument("--output_csv_filepath", help="file where the results will be saved", required=True)

    args = vars(parser.parse_args())
    main(csv_filepath=args["input_csv_filepath"], output_filepath=args["output_csv_filepath"])
