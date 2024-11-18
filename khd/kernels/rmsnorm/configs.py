import triton

from ...constants import TRITON_BLOCK_SIZES_POWERS_OF_2
from ...enums import KernelBackend
from ...utils import CutoTuneConfig


def _get_cutotune_configs() -> list[CutoTuneConfig]:
    configs = []
    for BLOCK_SIZE_H in TRITON_BLOCK_SIZES_POWERS_OF_2:
        for BLOCK_SIZE_B in [1, 2, 4, 8, 16, 32] + TRITON_BLOCK_SIZES_POWERS_OF_2:
            # we only use realistic configs where the block has between 64 and 64k elements
            if 1024 <= BLOCK_SIZE_B * BLOCK_SIZE_H <= 65536:
                configs.append(
                    CutoTuneConfig(
                        config={
                            "kernel_backend": KernelBackend.triton,
                            "BLOCK_SIZE_B": BLOCK_SIZE_B,
                            "BLOCK_SIZE_H": BLOCK_SIZE_H,
                        },
                        condition=lambda **kwargs: triton.next_power_of_2(kwargs["x"].size(-1))
                        == kwargs["BLOCK_SIZE_H"],
                    )
                )

    return configs
