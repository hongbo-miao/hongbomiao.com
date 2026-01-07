from config.register_activation_config import register_activation_config
from config.register_model_config import register_model_config
from config.register_optimizer_config import register_optimizer_config
from hydra_zen import make_config, store


def register_config() -> None:
    register_activation_config()
    register_model_config()
    register_optimizer_config()

    config = make_config(
        defaults=[
            "_self_",
            {"model": "small"},
            {"activation": "relu"},
            {"optimizer": "adam"},
        ],
        model=None,
        activation=None,
        optimizer=None,
        epoch_count=10,
        batch_size=32,
    )
    store(config, name="config")

    store.add_to_hydra_store()
