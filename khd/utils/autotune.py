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
        return True if self.condition is None else self.condition(**kwargs)

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

        if self.in_place_op:
            raise NotImplementedError()

        self.best_configs = {}
        self.signature = None

    def __call__(self, func):
        self._get_signature(func)

        @wraps(func)
        def inner(*args, **kwargs):
            input_key = self._get_input_key(args, kwargs)

            with self._recreate_cm():
                if input_key not in self.best_configs:
                    best_config, best_time = self._autotune(func, *args, **kwargs)
                    self.best_configs[input_key] = best_config

                    if _DEBUG_AUTOTUNE and (
                        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                    ):
                        print(f"config {best_config} achieved the best time ({best_time} sec) for {input_key}")

                return func(*args, **kwargs, **self.best_configs[input_key].get_key_values())

        return inner

    def _autotune(self, func: Callable, *args, **kwargs) -> tuple[Config, float]:
        def _get_kwargs_from_args_and_kwargs(*args, **kwargs) -> dict:
            result = {}
            for i, value in enumerate(args):
                key = self.signature.args[i]
                result[key] = value

            result.update(kwargs)
            return result

        best_config = None
        best_time = float("inf")

        for config in self.configs:
            if not config.is_condition_valid(**_get_kwargs_from_args_and_kwargs(*args, **kwargs)):
                continue

            elapsed_time = self._run_benchmark(func, *args, **kwargs, **config.get_key_values())

            if elapsed_time < best_time:
                best_config = config
                best_time = elapsed_time

        return best_config, best_time

    def _get_input_key(self, args: list, kwargs: dict) -> Any:
        input_key = []

        def _add_key(key: str, value) -> None:
            if isinstance(value, torch.Tensor):
                split_key = key.split(".")

                if len(split_key) == 1:
                    input_key.append(value.size())
                    input_key.append(value.stride())
                    input_key.append(value.dtype)
                else:
                    for _key in split_key:
                        _value = self._get_tensor_attribute(value, _key)
                        input_key.append(_value)
            else:
                input_key.append(value)

        for i, value in enumerate(args):
            key = self.signature.args[i]

            if key in self.trigger_keys:
                _add_key(value)

        for key, value in kwargs.items():
            if key in self.trigger_keys:
                _add_key(value)

        return tuple(input_key)

    def _get_tensor_attribute(self, tensor: torch.Tensor, key: str) -> int | torch.dtype:
        if key == "dtype":
            attribute = tensor.dtype
        else:
            is_size = key.startswith("size")
            is_shape = key.startswith("shape")
            is_stride = key.startswith("stride")

            if not (is_shape or is_size or is_stride):
                raise RuntimeError(f"unexpected key found ({key})")

            if is_size:
                prefix = "size("
                suffix = ")"
            elif is_shape:
                prefix = "shape["
                suffix = "]"
            elif is_stride:
                prefix = "stride("
                suffix = ")"

            key = key.split(prefix)[1]
            key = key.split(suffix)[0]
            dim = int(key)

            if is_size or is_shape:
                attribute = tensor.size(dim)
            elif is_stride:
                attribute = tensor.stride(dim)

        return attribute

    def _run_benchmark(self, func: Callable, *args, **kwargs) -> float:
        device_synchronize()
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
            config = config.get_key_values()
            for key in config:
                assert key in self.signature.args, f"unexpected arg ({key}) found in config"

        for key in self.trigger_keys:
            assert key in self.signature.args, f"unexpected arg ({key}) found in trigger_keys"

    def _check_configs(self) -> None:
        keys = set(self.configs[0].get_key_values().keys())

        for config in self.configs:
            assert set(config.get_key_values().keys()) == keys

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
