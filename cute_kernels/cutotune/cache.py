import json
import os
from collections import defaultdict

from .config import CutoTuneConfig


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


_CUTOTUNE_CACHE = None


def get_cutotune_cache() -> _CutoTuneCache:
    global _CUTOTUNE_CACHE
    if _CUTOTUNE_CACHE is None:
        _CUTOTUNE_CACHE = _CutoTuneCache("cute.json")
    return _CUTOTUNE_CACHE
