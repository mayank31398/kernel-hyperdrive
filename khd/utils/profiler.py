import torch

from ..constants import LIBRARY_NAME


class LibaryRecordFunction(torch.profiler.record_function):
    def __init__(self, name: str, args: str | None = None):
        super().__init__(f"{LIBRARY_NAME}:{name}", args)
