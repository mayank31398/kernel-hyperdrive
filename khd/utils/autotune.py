import inspect
import os
from collections import defaultdict
from contextlib import ContextDecorator
from functools import wraps
from time import perf_counter
from typing import Any, Callable

import torch
import torch.distributed

from .synchronization import device_synchronize


_DEBUG_CUTOTUNE = bool(os.getenv("DEBUG_CUTOTUNE", 0))
_DISABLE_CUTOTUNE = bool(os.getenv("DISABLE_CUTOTUNE", 0))
_SEPARATOR = "."


class CutoTuneConfig:
    def __init__(self, config: dict, condition: Callable = None) -> None:
        self.config = config
        self.condition = condition

    def is_condition_valid(self, **kwargs) -> bool:
        return True if self.condition is None else self.condition(**kwargs)

    def get_key_values(self) -> dict:
        return self.config

    def __str__(self) -> str:
        return str(self.config)


class CutoTune(ContextDecorator):
    def __init__(
        self,
        configs: list[CutoTuneConfig],
        triggers: set[str] = set(),
        overrideables: set[str] = set(),
        warmup_iterations: int = 5,
        benchmark_iterations: int = 100,
        in_place_op: bool = False,
    ) -> None:
        self.configs = configs
        self._check_configs()

        self.variable_name_trigger_map = defaultdict(list)
        self.overrideables = overrideables

        for trigger in triggers:
            variable_name, trigger = self._parse_trigger(trigger)
            self.variable_name_trigger_map[variable_name].append(trigger)

        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.in_place_op = in_place_op

        if self.in_place_op:
            raise NotImplementedError()

        self.best_configs = {}
        self.signature = None

    def __call__(self, func: Callable) -> Callable:
        if _DISABLE_CUTOTUNE:

            @wraps(func)
            def inner(*args, **kwargs):
                return func(*args, **kwargs)

        else:
            self._get_signature(func)

            @wraps(func)
            def inner(*args, **kwargs):
                input_key = self._get_input_key(args, kwargs)

                with self._recreate_cm():
                    if input_key not in self.best_configs:
                        best_config, best_time = self._autotune(func=func, args=args, kwargs=kwargs)

                        if _DEBUG_CUTOTUNE and (
                            not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
                        ):
                            print(f"config {best_config} achieved the best time ({best_time} sec) for {input_key}")

                        self.best_configs[input_key] = best_config

                    return func(
                        **self._get_function_arguments(config=self.best_configs[input_key], args=args, kwargs=kwargs)
                    )

        return inner

    def _get_function_arguments(self, config: CutoTuneConfig, args: list, kwargs: dict) -> dict:
        # copy the best_config first so we can override with args or kwargs
        result = {variable_name: value for variable_name, value in config.get_key_values().items()}

        for i, value in enumerate(args):
            variable_name = self.signature.args[i]
            result[variable_name] = value

        result.update(kwargs)

        return result

    def _parse_trigger(self, trigger: str) -> tuple[str, Callable]:
        split_trigger = trigger.split(_SEPARATOR)
        variable_name = split_trigger[0]

        if len(split_trigger) == 1:
            func = None
        elif len(split_trigger) == 2:
            func = split_trigger[1]

            if func == "dtype":
                func = lambda tensor: tensor.dtype
            elif func in ["size()", "shape"]:
                func = lambda tensor: tensor.size()
            elif func == "stride()":
                func = lambda tensor: tensor.stride()
            elif func.startswith("size"):
                dim = int(func[5:][:-1])
                func = lambda tensor: tensor.size(dim)
            elif func.startswith("shape"):
                dim = int(func[6:][:-1])
                func = lambda tensor: tensor.size(dim)
            elif func.startswith("stride"):
                dim = int(func[7:][:-1])
                func = lambda tensor: tensor.stride(dim)
            else:
                raise ValueError(f"unexpected triggeer found ({trigger})")

        return variable_name, func

    @torch.no_grad()
    def _autotune(self, func: Callable, args: list, kwargs: dict) -> tuple[CutoTuneConfig, float]:
        best_config = None
        best_time = float("inf")

        for config in self.configs:
            if not config.is_condition_valid(
                **self._get_function_arguments(config=CutoTuneConfig({}), args=args, kwargs=kwargs)
            ):
                continue

            elapsed_time = self._run_benchmark(
                func=func, **self._get_function_arguments(args=args, config=config, kwargs=kwargs)
            )

            if elapsed_time < best_time:
                best_config = config
                best_time = elapsed_time

        return best_config, best_time

    def _get_input_key(self, args: list, kwargs: dict) -> Any:
        input_key = []

        def _maybe_add_key(variable_name: str, value) -> None:
            if variable_name not in self.variable_name_trigger_map:
                return

            triggers = self.variable_name_trigger_map[variable_name]

            if isinstance(value, torch.Tensor):
                for trigger in triggers:
                    if trigger is None:
                        trigger = lambda tensor: (tensor.dtype, tensor.size(), tensor.stride())

                    input_key.append(f"{variable_name} = {trigger(value)}")
            else:

                assert len(triggers) == 1
                assert (
                    triggers[0] is None
                ), f"trigger ({variable_name}) is not a tensor and shouldn't have a functional trigger"

                input_key.append(f"{variable_name} = {value}")

        for i, value in enumerate(args):
            variable_name = self.signature.args[i]
            _maybe_add_key(variable_name, value)

        for variable_name, value in kwargs.items():
            _maybe_add_key(variable_name, value)

        return tuple(input_key)

    @torch.no_grad()
    def _run_benchmark(self, func: Callable, **kwargs: dict) -> float:
        device_synchronize()

        for _ in range(self.warmup_iterations):
            func(**kwargs)

        device_synchronize()
        start_time = perf_counter()

        for _ in range(self.benchmark_iterations):
            func(**kwargs)

        device_synchronize()
        end_time = perf_counter()
        elapsed_time = end_time - start_time

        return elapsed_time / self.benchmark_iterations

    def _get_signature(self, func: Callable) -> None:
        if self.signature is not None:
            return

        self.signature = inspect.getfullargspec(func)

        for config in self.configs:
            config = config.get_key_values()

            for variable_name in config:
                assert (
                    variable_name in self.signature.args
                ), f"unexpected variable_name ({variable_name}) found in config"

        for variable_name in self.variable_name_trigger_map:
            assert (
                variable_name in self.signature.args
            ), f"unexpected variable_name ({variable_name}) found in triggers"

    def _check_configs(self) -> None:
        variable_names = set(self.configs[0].get_key_values().keys())

        for config in self.configs:
            assert (
                set(config.get_key_values().keys()) == variable_names
            ), "autotune configs have different variable names"

    def __enter__(self) -> Any:
        return

    def __exit__(self, exception_type, exception_value, exception_traceback) -> Any:
        return


def get_default_cuda_autotune_configs(extra_config_condition: Callable = None) -> list[CutoTuneConfig]:
    configs = []

    # common configs for fp32, fp16 and bf16
    for vectorized_loop_size in [1, 2, 4]:
        for block_size in [64, 128, 256, 512, 1024]:
            configs.append(CutoTuneConfig({"vectorized_loop_size": vectorized_loop_size, "BLOCK_SIZE": block_size}))

    for block_size in [64, 128, 256, 512, 1024]:
        configs.append(
            CutoTuneConfig({"vectorized_loop_size": 8, "BLOCK_SIZE": block_size}, condition=extra_config_condition)
        )

    return configs


def get_default_triton_autotune_configs() -> list[CutoTuneConfig]:
    return [CutoTuneConfig({"BLOCK_SIZE": block_size}) for block_size in [64, 128, 256, 512, 1024]]


def make_contiguous(*args) -> list[torch.Tensor]:
    output = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            arg = arg.contiguous()

        output.append(arg)

    return output


def ensure_same_strides(*args, expected_stride: tuple[int], force_contiguous: bool = False) -> list[torch.Tensor]:
    if force_contiguous:
        output = make_contiguous(*args)
    else:
        mismatch = False
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.stride() != expected_stride:
                mismatch = True
                break

        output = make_contiguous(*args) if mismatch else args

    return output
