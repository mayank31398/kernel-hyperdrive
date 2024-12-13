import torch


def init_inductor() -> None:
    torch._dynamo.config.cache_size_limit = 64
