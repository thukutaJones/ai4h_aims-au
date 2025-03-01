import pathlib

import dotenv
import hydra
import lightning.pytorch as pl
import omegaconf
import pandas as pd
import torch
import tqdm

import qut01

dotenv.load_dotenv(override=True, verbose=True)


@hydra.main(version_base=None, config_path="../qut01/configs/", config_name="test.yaml")
def main(config):
    hydra_config = qut01.utils.config.get_hydra_config()
    output_dir = pathlib.Path(hydra_config.runtime.output_dir)
    qut01.utils.config.extra_inits(config, output_dir=output_dir)

    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="test")
    dataloader = datamodule.test_dataloader()

    data = []
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader):
            get_data_from_batch(data, batch)

    df = pd.DataFrame(
        data,
        columns=[
            "statement_id",
            "sentence_orig_idxs",
            "context",
            "targets",
        ],
    )
    df.to_csv(config.output_csv_file, index=False)


def get_data_from_batch(data, batch):
    print(batch["text_cls_token_indices"])
    for i, sentence_text in enumerate(batch["sentence_orig_text"]):
        statement_id = int(batch["statement_id"][i])
        sentence_orig_idxs = batch["sentence_orig_idxs"][i][0]
        text_with_context = batch["text"][i]
        target_classes = [int(x) for x in batch["relevance"][i, :]]
        data.append(
            [
                statement_id,
                sentence_orig_idxs,
                text_with_context,
                target_classes,
            ]
        )


if __name__ == "__main__":
    """Extracts contextualized sentences from the deeplake dataloaders and saves it in a csv file.
    The file has to be called by using custom args.
    For example: python extract_data_to_dataframe.py ckpt_path=checkpoint.ckpt experiment=config.yaml +output_csv_file=/a/file/path.csv
    """
    main()
