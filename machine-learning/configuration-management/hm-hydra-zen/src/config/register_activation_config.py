from hydra_zen import just, store
from torch import nn


def register_activation_config() -> None:
    activation_store = store(group="activation")
    activation_store(just(nn.ReLU), name="relu")
    activation_store(just(nn.LeakyReLU), name="leaky_relu")
    activation_store(just(nn.GELU), name="gelu")
