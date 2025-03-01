import pathlib

import dotenv
import hydra
import lightning.pytorch as pl
import omegaconf

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


def test(config: omegaconf.DictConfig) -> None:
    """Runs the testing pipeline based on a specified model checkpoint path.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """
    exp_name, run_name, run_type, job_name = config.experiment_name, config.run_name, config.run_type, config.job_name
    logger.info(f"Launching ({exp_name}: {run_name}, '{run_type}', job={job_name})")

    hydra_config = qut01.utils.config.get_hydra_config()
    output_dir = pathlib.Path(hydra_config.runtime.output_dir)
    logger.info(f"Output directory: {output_dir.absolute()}")
    qut01.utils.config.extra_inits(config, output_dir=output_dir)

    logger.info(f"Instantiating datamodule: {config.data.datamodule._target_}")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(config.data.datamodule)

    model_class = qut01.utils.config.get_model(config).__class__
    model = model_class.load_from_checkpoint(config.ckpt_path)

    datamodule.setup("validate")
    val_dataloader = datamodule.val_dataloader()
    batch = next(iter(val_dataloader))
    model.head = None  # trick to force the model to return the embeddings
    result = model(batch)
    only_cls = result[:, 0, :]


@hydra.main(version_base=None, config_path="configs/", config_name="test.yaml")
def main(config):
    """Code to reload a model and compute the sentence embeddings."""
    import qut01  # importing here to avoid delay w/ hydra tab completion

    return test(config)


if __name__ == "__main__":
    main()
