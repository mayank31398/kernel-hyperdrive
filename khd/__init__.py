from .kernel_registry import KernelRegistry
from .scattermoe import MoE_Torch, MoE_Triton
from .vector_addition import vector_addition_cuda, vector_addition_torch, vector_addition_triton


KernelRegistry()
