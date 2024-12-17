import os
from collections import defaultdict

import yaml

from ..enums import KernelBackend
from ..utils import get_boolean_env_variable
from .config import CutoTuneConfig


_CUTOTUNE_CACHE_FILENAME = os.path.join(os.path.dirname(__file__), "cutotune_cache.yml")
_LOAD_CUTOTUNE_CACHE = get_boolean_env_variable("LOAD_CUTOTUNE_CACHE")


class _CutoTuneCache:
    def __init__(self) -> None:
        self.full_cache = defaultdict(lambda: defaultdict(list))
        self.best_cache = defaultdict(lambda: defaultdict(list))

        if _LOAD_CUTOTUNE_CACHE and os.path.exists(_CUTOTUNE_CACHE_FILENAME):
            self.load()

    def add_config(self, function_hash: str, lookup_key: str, config: CutoTuneConfig, time: float) -> None:
        # use list instead of tuple since yaml is more cleaner with list
        self.full_cache[function_hash][lookup_key].append([config, time])

        min_time = float("inf")
        if lookup_key in self.best_cache[function_hash][lookup_key]:
            min_time = self.best_cache[function_hash][lookup_key][1]

        if time < min_time:
            self.best_cache[function_hash][lookup_key] = [config, time]

    def save(self) -> None:
        full_cache_serialized = {"all_configs": self.full_cache, "best_configs": self.best_cache}
        yaml.dump(full_cache_serialized, open(_CUTOTUNE_CACHE_FILENAME, "w"))

    def load(self) -> None:
        self.full_cache = yaml.load(open(_CUTOTUNE_CACHE_FILENAME, "r"), yaml.SafeLoader)

    def get_best_configs(self, function_hash: str) -> dict[str, CutoTuneConfig]:
        return self.best_cache[function_hash]


_CUTOTUNE_CACHE = None


def get_cutotune_cache() -> _CutoTuneCache:
    global _CUTOTUNE_CACHE

    if _CUTOTUNE_CACHE is None:
        _CUTOTUNE_CACHE = _CutoTuneCache()

    return _CUTOTUNE_CACHE


def save_cutotune_cache() -> None:
    global _CUTOTUNE_CACHE
    assert _CUTOTUNE_CACHE is not None

    _CUTOTUNE_CACHE.save()
