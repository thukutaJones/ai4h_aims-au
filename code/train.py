import dotenv
import hydra

try:
    # comet likes to be imported before torch for auto-logging to work
    # as of 2023-03-15 with comet_ml==3.32.4, we get a warning if not
    # (this is likely one of the easiest places to perform the import)
    import comet_ml
except ImportError:
    pass


dotenv.load_dotenv(override=True, verbose=True)


@hydra.main(version_base=None, config_path="qut01/configs/", config_name="train.yaml")
def main(config):
    """Main entrypoint for the training pipeline."""
    import qut01  # importing here to avoid delay w/ hydra tab completion

    return qut01.train(config)


if __name__ == "__main__":
    main()
