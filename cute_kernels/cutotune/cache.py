import json
import os
from collections import defaultdict
from enum import Enum

from .config import CutoTuneConfig


_CUTOTUNE_CACHE_FILENAME = "cute.json"


class _CutoTuneCache:
    def __init__(self, filename: str) -> None:
        self.full_cache = defaultdict(lambda: defaultdict(list))

        if os.path.exists(filename):
            self.full_cache = json.load(open(filename, "r"))

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
                        if isinstance(value, Enum):
                            serialized_config[key] = value.value
                        else:
                            serialized_config[key] = value

                    full_cache_serialized[function_hash][lookup_key].append((serialized_config, time))

        json.dump(full_cache_serialized, open(_CUTOTUNE_CACHE_FILENAME, "w"), indent=4)


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
