import json
import os
from collections import defaultdict

from .config import CutoTuneConfig


class CutoTuneCache:
    def __init__(self, filename: str) -> None:
        self.full_cache = defaultdict(defaultdict(list))
        self.min_cache = defaultdict(defaultdict(dict))

        if os.path.exists(filename):
            self.full_cache = json.load(open(filename, "r"))

            for function_hash, lookup_key_timed_configs in self.full_cache.items():
                for lookup_key, timed_configs in lookup_key_timed_configs.items():
                    # timed_configs is a tuple (config, time)
                    self.min_cache[function_hash][lookup_key] = min(timed_configs, key=lambda x: x[1])

    def __getitem__(self, key: tuple) -> CutoTuneConfig:
        function_hash, lookup_key = key
        return self.min_cache[function_hash][lookup_key]

    def add_config(self, function_hash: str, lookup_key: str, config: CutoTuneConfig, time: float) -> None:
        self.full_cache[function_hash][lookup_key].append((config, time))

        min_time = self.min_cache[function_hash][lookup_key][1]
        if time < min_time:
            self.min_cache[function_hash][lookup_key] = (config, time)

    def has_config(self, function_hash: str, lookup_key: str) -> bool:
        if function_hash in self.min_cache:
            return lookup_key in self.min_cache[function_hash]

        return False

    def get_best_config(
        self, function_hash: str, lookup_key: str, default: CutoTuneConfig | None = None
    ) -> CutoTuneConfig:
        lookup_key_configs = self.min_cache.get(function_hash, None)

        if lookup_key_configs is None:
            return default

        return lookup_key_configs.get(lookup_key, default)
