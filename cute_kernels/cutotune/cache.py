import os
from collections import defaultdict

import yaml

from ..enums import KernelBackend
from .config import CutoTuneConfig


_CUTOTUNE_CACHE_FILENAME = "cute.yml"


class _CutoTuneCache:
    def __init__(self, filename: str) -> None:
        self.full_cache = defaultdict(lambda: defaultdict(list))

        if os.path.exists(filename):
            self.load()

    def add_config(self, function_hash: str, lookup_key: str, config: CutoTuneConfig, time: float) -> None:
        self.full_cache[function_hash][lookup_key].append((config, time))

    def save(self) -> None:
        full_cache_serialized = {}

        for function_hash in self.full_cache:
            full_cache_serialized[function_hash] = {}

            for lookup_key in self.full_cache[function_hash]:
                full_cache_serialized[function_hash][lookup_key] = []

                for config, time in self.full_cache[function_hash][lookup_key]:
                    serialized_config = {}

                    for key, value in config.get_key_values().items():
                        if isinstance(value, KernelBackend):
                            serialized_config[key] = value.value
                        else:
                            serialized_config[key] = value

                    full_cache_serialized[function_hash][lookup_key].append(
                        {"config": serialized_config, "time": time}
                    )

        yaml.dump(full_cache_serialized, open(_CUTOTUNE_CACHE_FILENAME, "w"))

    def load(self) -> None:
        full_cache_serialized = yaml.load(open(_CUTOTUNE_CACHE_FILENAME, "r"), yaml.SafeLoader)

        for function_hash in full_cache_serialized:
            for lookup_key in full_cache_serialized[function_hash]:
                for config_time in full_cache_serialized[function_hash][lookup_key]:
                    config = config_time["config"]
                    time = config_time["time"]

                    unserialized_config = {}
                    for key, value in config.items():
                        if key == "kernel_backend":
                            value = KernelBackend(value)

                        unserialized_config[key] = value

                    self.full_cache[function_hash][lookup_key].append((unserialized_config, time))


_CUTOTUNE_CACHE = None


def get_cutotune_cache() -> _CutoTuneCache:
    global _CUTOTUNE_CACHE, _CUTOTUNE_CACHE_FILENAME

    if _CUTOTUNE_CACHE is None:
        _CUTOTUNE_CACHE = _CutoTuneCache(_CUTOTUNE_CACHE_FILENAME)

    return _CUTOTUNE_CACHE


def save_cutotune_cache() -> None:
    global _CUTOTUNE_CACHE
    assert _CUTOTUNE_CACHE is not None

    _CUTOTUNE_CACHE.save()
