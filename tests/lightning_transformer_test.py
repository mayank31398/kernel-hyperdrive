import torch
import torch.nn as nn
from parameterized import parameterized

from khd import lightning_transformer_triton

from .test_commons import TestCommons


class LightningTransformerTest(TestCommons):
    @parameterized.expand(TestCommons.get_2d_tensor_sizes(), TestCommons.make_args_matrix([torch.device("cuda")]))
    def test_lightning_transformer(self, sizes: tuple[int], device: torch.device) -> None:
        vocab_size = 49152
        input_ids = torch.randint(0, vocab_size, sizes, device=device, dtype=torch.long)
        wte = nn.Embedding(vocab_size, 4096).to(device)

        assert wte(input_ids).equal(lightning_transformer_triton(input_ids, wte.weight))
