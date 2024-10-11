import inspect
import os
from contextlib import ContextDecorator
from functools import wraps
from time import perf_counter
from typing import Any, Callable

import torch
import torch.distributed

from .synchronization import device_synchronize


_DEBUG_AUTOTUNE = bool(os.getenv("DEBUG_KHD_AUTOTUNE", 0))


class Config:
    def __init__(self, config: dict, condition: Callable = None) -> None:
        self.config = config
        self.condition = condition

    def is_condition_valid(self, **kwargs) -> bool:
        if self.condition is None:
            return True
        return self.condition(**kwargs)

    def get_key_values(self) -> dict:
        return self.config

    def __str__(self) -> str:
        return str(self.config)


class AutoTune(ContextDecorator):
    def __init__(
        self, configs: list[Config], trigger_keys: list[str] = {}, num_iterations: int = 100, in_place_op: bool = False
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
            best_key = self._get_best_key(args, kwds)

            with self._recreate_cm():
                if best_key not in self.best_config:
                    for config in self.configs:
                        if not config.is_condition_valid(self._get_kwargs_from_args_and_kwargs(*args, **kwds)):
                            continue

                        elapsed_time = self._run_benchmark(func, *args, **kwds, **config.get_key_values())

                        if elapsed_time < self.best_time:
                            self.best_config[best_key] = config
                            self.best_time = elapsed_time

                    if _DEBUG_AUTOTUNE:
                        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                            print(f"config {config} achieved the best time ({elapsed_time} sec) for {best_key}")

                return func(*args, **kwds, **self.best_config[best_key].get_key_values())

        return inner

    def _get_kwargs_from_args_and_kwargs(self, *args, **kwargs) -> dict:
        result = {}
        for i, value in enumerate(args):
            key = self.signature.args[i]
            result[key] = value

        result.update(kwargs)
        return result

    def _get_best_key(self, args: list, kwargs: dict) -> Any:
        best_keys = []

        for i, value in enumerate(args):
            key = self.signature.args[i]

            if key in self.trigger_keys:
                if isinstance(value, torch.Tensor):
                    best_keys.append(value.size())
                    best_keys.append(value.dtype)
                else:
                    best_keys.append(value)

        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                best_keys.append(value.size())
                best_keys.append(value.dtype)
            else:
                best_keys.append(value)

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


def get_vectorized_autotune_configs(extra_config_condition: Callable = None) -> list[dict]:
    configs = []

    # common configs for fp32, fp16 and bf16
    for vectorized_loop_size in [1, 2, 4]:
        for block_size in [64, 128, 256, 512, 1024]:
            configs.append(Config({"vectorized_loop_size": vectorized_loop_size, "BLOCK_SIZE": block_size}))

    for block_size in [64, 128, 256, 512, 1024]:
        configs.append(Config({"vectorized_loop_size": 8, "BLOCK_SIZE": block_size}, condition=extra_config_condition))

    return configs
