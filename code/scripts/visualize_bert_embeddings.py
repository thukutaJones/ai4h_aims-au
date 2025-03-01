import pathlib

import dotenv
import hydra
import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import omegaconf
import pandas as pd
import plotly.express as px
import torch
import tqdm
import umap
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertModel, BertTokenizer

import qut01

try:
    # comet likes to be imported before torch for auto-logging to work
    # as of 2023-03-15 with comet_ml==3.32.4, we get a warning if not
    # (this is likely one of the easiest places to perform the import)
    import comet_ml
except ImportError:
    pass


dotenv.load_dotenv(override=True, verbose=True)
logger = qut01.utils.get_logger(__name__)


def reduce_dims(num_dims: int, dim_technique: str, sentence_embeddings):
    """Reduce dimensions for visualization.

    Args:
        num_dims (int): Number of dimesions to reduce to.
        dim_technique (str): Whether to use tSNE or UMAP.
        sentence_embeddings (np.array): Embeddings of a model.
    """
    # Dimesionality reduction option
    if dim_technique == "UMAP":
        umap_model = umap.UMAP(
            n_components=num_dims,
            random_state=42,
        )
        return umap_model.fit_transform(sentence_embeddings)
    else:
        # Perform t-SNE for dimensionality reduction
        tsne = TSNE(n_components=num_dims, random_state=42)
        return tsne.fit_transform(sentence_embeddings)


def get_categories(datamodule):
    example_item = next(datamodule.train_dataloader().__iter__())
    class_names = example_item["class_names"]
    categories = list(class_names)
    return categories


def get_labels_from_target(target, categories):
    labels = [categories[i] for i, val in enumerate(target) if val == 1]
    return labels


def get_main_category(label):
    # Split the label by space and take the first part before "(" if it exists
    return label.split(" ")[0]


def test(config: omegaconf.DictConfig) -> None:
    """Runs the testing pipeline based on a specified model checkpoint path.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    exp_name, run_name, run_type, job_name = config.experiment_name, config.run_name, config.run_type, config.job_name
    logger.info(f"Launching ({exp_name}: {run_name}, '{run_type}', job={job_name})")
    num_dims = 2  # Change to 3 for 3D visualization
    dim_technique = "UMAP"  # options UMAP or tsne
    is_finetuned = False  # set True for pretrained weights
    model_name = "sbert"  # options bert, sbert, only bert is supported for finetuned visualization.

    hydra_config = qut01.utils.config.get_hydra_config()
    output_dir = pathlib.Path(hydra_config.runtime.output_dir)
    logger.info(f"Output directory: {output_dir.absolute()}")
    qut01.utils.config.extra_inits(config, output_dir=output_dir)

    logger.info(f"Instantiating datamodule: {config.data.datamodule._target_}")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)
    datamodule.setup("validate")

    if is_finetuned:
        assert model_name == "bert", "Only BERT finetuned model is expected."
        all_embeddings, df = generate_embeddings_from_finetunedBERT(config, datamodule)
        all_embeddings = np.concatenate(all_embeddings)

    else:
        data = []
        for item in tqdm.tqdm(datamodule.val_dataloader()):
            get_data_from_batch(data, item)

        df = pd.DataFrame(
            data, columns=["sentence_text", "sentence_statement_id", "sentence_orig_idxs", "target_classes"]
        )

        if model_name == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertModel.from_pretrained("bert-base-uncased")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Move model to GPU
            model = model.to(device)

            def get_sentence_embeddings(sentences, batch_size=32):
                # Tokenize the sentences
                inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=128)

                # Move the inputs to the GPU
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                # Create DataLoader for batching
                data_loader = DataLoader(TensorDataset(input_ids, attention_mask), batch_size=batch_size)

                all_embeddings = []
                with torch.no_grad():
                    for batch in tqdm.tqdm(data_loader):
                        input_ids, attention_mask = batch
                        outputs = model(input_ids, attention_mask=attention_mask)
                        all_embeddings.append(outputs.pooler_output.cpu())

                return torch.cat(all_embeddings).numpy()

            # Generate sentence embeddings
            sentence_texts = [str(sentence) for sentence in df["sentence_text"].tolist() if isinstance(sentence, str)]
            all_embeddings = get_sentence_embeddings(sentence_texts)

        else:
            # SentenceBERT model
            model = SentenceTransformer("all-MiniLM-L6-v2")

            def get_sentence_embeddings(sentences):
                return model.encode(sentences)

            all_embeddings = np.array(get_sentence_embeddings(df["sentence_text"]))

    reduced_embeddings = reduce_dims(num_dims, dim_technique, all_embeddings)
    print("Embedding reducted to shape: ", reduced_embeddings.shape)

    categories = get_categories(datamodule)
    label_sets = [get_labels_from_target(target, categories) for target in df["target_classes"]]
    df["label_sets"] = label_sets
    print(df.head())
    # Visualize embeddings using matplotlib
    if num_dims == 2:
        save_combined_plot(
            reduced_embeddings,
            categories,
            label_sets,
            use_main_categories=False,
            output_file=f"all_{dim_technique}_categories_2d.png",
            plot_title=f"{dim_technique} 2D Visualization of Sentence Embeddings of finetuned BERT",
        )

        save_combined_plot(
            reduced_embeddings,
            categories,
            label_sets,
            use_main_categories=True,
            output_file=f"all_{dim_technique}_main_categories_2d.png",
            plot_title=f"{dim_technique} 2D Visualization of Sentence Embeddings of finetuned BERT",
        )

    else:
        generate_3d_visualization(
            reduced_embeddings,
            df,
            label_sets,
            categories=categories,
            use_main_categories=False,
            output_file=f"all_{dim_technique}_categories_3d_interactive.html",
            plot_title=f"{dim_technique} 3D Visualization of Sentence Embeddings of finetuned BERT",
        )

        generate_3d_visualization(
            reduced_embeddings,
            df,
            label_sets,
            categories=categories,
            use_main_categories=True,
            output_file=f"all_{dim_technique}_main_categories_3d_interactive.html",
            plot_title=f"{dim_technique} 3D Visualization of Sentence Embeddings of finetuned BERT",
        )


def generate_embeddings_from_finetunedBERT(config, datamodule):
    """Using a pretrained checkpoint, return embeddings from the CLS token from a finetuned BERT
    model.

    Args:
        config: Configuration file.
        datamodule: Data

    Returns:
        pd.array, pd.DataFrame: Embeddings and a dataframe with the sentences that will be used for visualization.
    """
    model_class = qut01.utils.config.get_model(config).__class__
    model = model_class.load_from_checkpoint(config.ckpt_path)

    val_dataloader = datamodule.val_dataloader()

    model.head = None  # trick to force the model to return the embeddings
    model.eval()

    all_embeddings = []
    data = []
    for batch in tqdm.tqdm(val_dataloader):
        result = model(batch)
        only_cls = result[:, 0, :].detach()
        all_embeddings.append(only_cls.cpu().numpy())  # Move to CPU and convert to NumPy

        get_data_from_batch(data, batch)

        del result  # Free up memory
        torch.cuda.empty_cache()  # Clear GPU memory

    df = pd.DataFrame(data, columns=["sentence_text", "sentence_statement_id", "sentence_orig_idxs", "target_classes"])

    return all_embeddings, df


def get_data_from_batch(data, batch):
    """Collect sentences from the data.

    Args:
        data: Array of data.
        batch: Current data batch from the dataloader.
    """
    for i, sentence_text in enumerate(batch["sentence_orig_text"]):
        sentence_statement_id = int(batch["statement_id"][i])
        sentence_orig_idxs = batch["sentence_orig_idxs"][i]
        assert len(sentence_orig_idxs) == 1
        sentence_orig_idxs = int(sentence_orig_idxs[0])
        text_with_context = batch["text"][i]
        assert (
            text_with_context == sentence_text
        ), f"context must be disabled in this experiment. Found '{text_with_context}'"

        target_classes = [int(x) for x in batch["relevance"][i, :]]

        data.append([sentence_text, sentence_statement_id, sentence_orig_idxs, target_classes])


def format_hover_text(text, max_len=50):
    """Inserts line breaks in long text for better readability in hover tooltips.

    Args:
        text: Text to be formatted.
        max_len: Maximum length of the text box.
    Returns:
        str: Formatted text.
    """
    if len(text) > max_len:
        wrapped_text = "<br>".join([text[i : i + max_len] for i in range(0, len(text), max_len)])
    else:
        wrapped_text = text
    return f"<b>{wrapped_text}<br>"


def generate_3d_visualization(
    reduced_embeddings,
    df,
    label_sets,
    categories=None,
    use_main_categories=False,
    output_file="3d_visualization.html",
    plot_title="3D Visualization of Sentence Embeddings",
):
    """Generates a 3D scatter plot for sentence embeddings using Plotly.

    Args:
        reduced_embeddings (ndarray): Reduced dimensional embeddings.
        df (DataFrame): DataFrame containing sentence information.
        label_sets (list): List of label sets for each sentence.
        categories (list): List of categories (optional).
        use_main_categories (bool): Flag to indicate if main categories should be used (default False).
        output_file (str): Path to save the plot as an interactive HTML file (default '3d_visualization.html').
        plot_title (str): Title for the plot (default '3D Visualization of Sentence Embeddings').
    """
    umap_df = pd.DataFrame(reduced_embeddings, columns=["Dim 1", "Dim 2", "Dim 3"])
    umap_df["text"] = df["sentence_text"].tolist()
    umap_df["labels"] = [", ".join(labels) for labels in label_sets]
    umap_df["labels"] = label_sets
    umap_df_exploded = umap_df.explode("labels")
    umap_df_exploded = umap_df_exploded.dropna(subset=["labels"])

    umap_df_exploded["hover_text"] = umap_df_exploded.apply(lambda row: format_hover_text(row["text"]), axis=1)

    # Handle main categories if needed
    if use_main_categories and categories is not None:
        # Create a dictionary to map labels to main categories
        main_categories_dict = {label: get_main_category(label) for label in categories}
        umap_df_exploded["main_category"] = [main_categories_dict[label] for labels in label_sets for label in labels]
        color_column = "main_category"
        color_sequence = px.colors.qualitative.T10  # Color scheme for main categories
    else:
        color_column = "labels"
        color_sequence = px.colors.qualitative.Dark24  # Color scheme for regular categories

    fig = px.scatter_3d(
        umap_df_exploded,
        x="Dim 1",
        y="Dim 2",
        z="Dim 3",
        color=color_column,
        hover_name="hover_text",
        title=plot_title,
        color_discrete_sequence=color_sequence,
        width=1200,
        height=800,
    )

    fig.update_traces(marker=dict(size=5, opacity=0.6))

    fig.update_layout(
        legend=dict(
            itemsizing="constant",
            font=dict(size=12),
        ),
        legend_title=dict(font=dict(size=16)),
    )

    fig.write_html(output_file)
    print(f"Saved image : {output_file}")


def save_combined_plot(
    reduced_embeddings,
    categories,
    label_sets,
    use_main_categories=True,
    output_file="2d_visualization.png",
    plot_title="Visualization of Sentence Embeddings with Main Categories using finetuned BERT",
    alpha=0.6,
):
    """Saves a 2D scatter plot of embeddings with labels and optional main categories.

    Args:
        reduced_embeddings (ndarray): Reduced dimensional embeddings (2D array).
        categories (list): List of categories or labels.
        label_sets (list): List of label sets for each embedding.
        use_main_categories (bool): Whether to use main categories (default: True).
        output_file (str): Path to save the plot (default: '3d_visualization.html').
        plot_title (str): Title for the plot (default: 'tsne Visualization of Sentence Embeddings').
        alpha (float): Transparency level of the points (default: 0.6).
    """
    # Determine the main categories if needed
    if use_main_categories:
        main_categories_dict = {label: get_main_category(label) for label in categories}
        unique_categories = sorted(set(main_categories_dict.values()))
        colormap = plt.cm.get_cmap("tab10", len(unique_categories))  # Color map for main categories

        def category_mapper(label):
            return unique_categories.index(main_categories_dict[label])

    else:
        colormap = plt.cm.get_cmap("tab20", len(categories))

        def category_mapper(label):
            return categories.index(label)

    # Plot
    plt.figure(figsize=(10, 6))
    handles = {}  # Dictionary to store handles for the legend

    for embedding, labels in zip(reduced_embeddings, label_sets):
        for label in labels:
            color = colormap(category_mapper(label))
            # Only add a scatter plot for the label if it hasn't been added yet
            if label not in handles:
                scatter_handle = plt.scatter(
                    embedding[0], embedding[1], marker="o", color=color, alpha=alpha, label=label
                )
                handles[label] = scatter_handle
            else:
                plt.scatter(embedding[0], embedding[1], marker="o", color=color, alpha=alpha)

    # Add plot labels and title
    plt.title(plot_title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    # Create a legend, avoiding duplicate entries

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Save the plot as a file
    plt.savefig(output_file, format="png", dpi=300, bbox_inches="tight")
    print(f"Saved image : {output_file}")

    plt.close()


@hydra.main(version_base=None, config_path="qut01/configs/", config_name="test.yaml")
def main(config):
    """Code to reload a model and compute the sentence embeddings."""
    import qut01  # importing here to avoid delay w/ hydra tab completion

    return test(config)


if __name__ == "__main__":
    main()
