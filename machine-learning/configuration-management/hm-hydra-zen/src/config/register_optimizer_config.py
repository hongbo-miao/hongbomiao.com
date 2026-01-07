import torch
from hydra_zen import builds, store


def register_optimizer_config() -> None:
    adam_config = builds(torch.optim.Adam, populate_full_signature=True, lr=0.001)
    sgd_config = builds(
        torch.optim.SGD,
        populate_full_signature=True,
        lr=0.01,
        momentum=0.9,
    )

    optimizer_store = store(group="optimizer")
    optimizer_store(adam_config, name="adam")
    optimizer_store(sgd_config, name="sgd")
