import json
import os
from collections import defaultdict

from .config import CutoTuneConfig


_CUTOTUNE_CACHE_FILENAME = "cute.json"


class _CutoTuneCache:
    def __init__(self, filename: str) -> None:
        self.full_cache = defaultdict(lambda: defaultdict(list))
        self.min_cache = defaultdict(lambda: defaultdict(dict))

        if os.path.exists(filename):
            self.full_cache = json.load(open(filename, "r"))

            for function_hash, lookup_key_timed_configs in self.full_cache.items():
                for lookup_key, timed_configs in lookup_key_timed_configs.items():
                    # timed_configs is a tuple (config, time)
                    self.min_cache[function_hash][lookup_key] = min(timed_configs, key=lambda x: x[1])

    def add_config(self, function_hash: str, lookup_key: str, config: CutoTuneConfig, time: float) -> None:
        self.full_cache[function_hash][lookup_key].append((config, time))

    def save(self) -> None:
        json.dump(self.full_cache, open(_CUTOTUNE_CACHE_FILENAME, "w"), indent=4)


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
