import dotenv
import hydra

dotenv.load_dotenv(override=True, verbose=True)


@hydra.main(version_base=None, config_path="qut01/configs/", config_name="profiler.yaml")
def main(config):
    """Main entrypoint for the model profiler pipeline."""
    import qut01  # importing here to avoid delay w/ hydra tab completion

    return qut01.model_profiler(config)


if __name__ == "__main__":
    main()
