import inspect
from contextlib import ContextDecorator
from functools import wraps
from time import perf_counter
from typing import Any, Callable

import torch

from .synchronization import device_synchronize


class AutoTune(ContextDecorator):
    def __init__(
        self, configs: list[dict], trigger_keys: list[str], num_iterations: int = 100, in_place_op: bool = False
    ) -> None:
        self.configs = configs
        self._check_configs()

        self.trigger_keys = set(trigger_keys)
        self.num_iterations = num_iterations
        self.in_place_op = in_place_op

        self.best_config = {}
        self.best_time = float("inf")

        if self.in_place_op:
            raise NotImplementedError()

        self.signature = None

    def __call__(self, func):
        self._get_signature(func)

        @wraps(func)
        def inner(*args, **kwds):
            print(self.best_config)
            best_key = self._get_best_key(args, kwds)

            with self._recreate_cm():
                if best_key not in self.best_config:
                    for config in self.configs:
                        elapsed_time = self._run_benchmark(func, *args, **kwds, **config)

                        if elapsed_time < self.best_time:
                            self.best_config[best_key] = config
                            self.best_time = elapsed_time

                return func(*args, **kwds, **self.best_config[best_key])

        return inner

    def _get_best_key(self, args: list, kwargs: dict) -> Any:
        best_keys = []

        for i, value in enumerate(args):
            key = self.signature.args[i]

            if key in self.trigger_keys:
                best_keys.append(value.size() if isinstance(value, torch.Tensor) else value)

        for key, value in kwargs.items():
            best_keys.append(value.size() if isinstance(value, torch.Tensor) else value)

        return tuple(best_keys)

    def _run_benchmark(self, func: Callable, *args, **kwargs) -> float:
        start_time = perf_counter()

        for _ in range(self.num_iterations):
            func(*args, **kwargs)
        device_synchronize()

        end_time = perf_counter()
        elapsed_time = end_time - start_time

        return elapsed_time

    def _get_signature(self, func: Callable) -> None:
        if self.signature is not None:
            return

        self.signature = inspect.getfullargspec(func)

        for config in self.configs:
            for key in config:
                assert key in self.signature.args, f"unexpected arg ({key}) found in config"

        for key in self.trigger_keys:
            assert key in self.signature.args, f"unexpected arg ({key}) found in trigger_keys"

    def _check_configs(self) -> None:
        keys = set(self.configs[0].keys())

        for config in self.configs:
            assert set(config.keys()) == keys

    def __enter__(self) -> Any:
        return

    def __exit__(self, exception_type, exception_value, exception_traceback) -> Any:
        return


def get_vectorized_autotune_configs() -> list[dict]:
    configs = []

    for vectorized_loop_size in [1, 2, 4, 8]:
        for block_size in [64, 128, 256, 512, 1024]:
            configs.append(
                {"dtype": torch.float16, "vectorized_loop_size": vectorized_loop_size, "BLOCK_SIZE": block_size}
            )
            configs.append(
                {"dtype": torch.bfloat16, "vectorized_loop_size": vectorized_loop_size, "BLOCK_SIZE": block_size}
            )

    for vectorized_loop_size in [1, 2, 4]:
        for block_size in [64, 128, 256, 512, 1024]:
            configs.append(
                {"dtype": torch.float32, "vectorized_loop_size": vectorized_loop_size, "BLOCK_SIZE": block_size}
            )

    return configs
