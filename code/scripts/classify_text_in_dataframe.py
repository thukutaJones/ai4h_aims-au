import pathlib

import dotenv
import hydra
import omegaconf
import pandas as pd
import torch

import qut01

dotenv.load_dotenv(override=True, verbose=True)


class SentenceRelevanceClassifier:
    def __init__(self, config: omegaconf.DictConfig) -> None:
        hydra_config = qut01.utils.config.get_hydra_config()
        output_dir = pathlib.Path(hydra_config.runtime.output_dir)
        qut01.utils.config.extra_inits(config, output_dir=output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.backbone_model_name == "bert-base-uncased":
            self.tokenizer = hydra.utils.instantiate(config.data.tokenizer)
        elif config.backbone_model_name == "meta-llama/Llama-3.2-3B":
            self.tokenizer = hydra.utils.instantiate(config.wrapped_tokenizer)
        else:
            raise ValueError("Only Llama3.2 3B and BERT are currently supported")

        model_class = qut01.utils.config.get_model(config).__class__
        self.model = model_class.load_from_checkpoint(config.ckpt_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        tokenized_text = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        batch = {
            "text_token_ids": tokenized_text.input_ids.to(self.device),
            "text_attention_mask": tokenized_text.attention_mask.to(self.device),
            "batch_size": 1,
            "text_cls_token_indices": 0,
        }

        with torch.no_grad():
            probs = [float(x) for x in torch.sigmoid(self.model(batch)[0, :])]
            return probs


@hydra.main(version_base=None, config_path="../qut01/configs/", config_name="test.yaml")
def main(config):
    """Minimal code to reload a fine-tuned model and perform inference on text stored in a pandas dataframe.
    The file has to be called by using custom args.
    For example: python classify_text_in_dataframe.py ckpt_path=checkpoint.ckpt experiment=config.yaml +input_csv_file=/a/file/path.csv +output_csv_file=/a/file/path.csv
    """

    df = pd.read_csv(config.input_csv_file)
    classifier = SentenceRelevanceClassifier(config)
    df["predictions"] = df.apply(lambda row: classifier.predict(row["context"]), axis=1)
    df.to_csv(config.output_csv_file)


if __name__ == "__main__":
    main()
