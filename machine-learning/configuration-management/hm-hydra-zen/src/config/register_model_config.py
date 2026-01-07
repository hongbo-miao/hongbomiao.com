from config.constants import INPUT_SIZE, OUTPUT_SIZE
from hydra_zen import builds, store
from models.large_net import LargeNet
from models.small_net import SmallNet


def register_model_config() -> None:
    small_net_config = builds(
        SmallNet,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        populate_full_signature=True,
    )
    large_net_config = builds(
        LargeNet,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        populate_full_signature=True,
    )

    model_store = store(group="model")
    model_store(small_net_config, name="small")
    model_store(large_net_config, name="large")
