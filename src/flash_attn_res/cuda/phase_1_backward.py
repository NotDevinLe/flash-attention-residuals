import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def cuda_phase_1_backward_enabled() -> bool:
    return os.getenv("FLASH_ATTN_RES_USE_CUDA_PHASE1_BACKWARD", "0") == "1"


def _supports_cuda_phase_1_backward(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    grad_softmax_outputs: torch.Tensor,
) -> bool:
    if not cuda_phase_1_backward_enabled():
        return False

    if not block_representations.is_cuda:
        return False

    if block_representations.dtype != torch.bfloat16:
        return False

    if pseudo_queries.dtype != torch.bfloat16:
        return False

    if grad_softmax_outputs.dtype not in (torch.bfloat16, torch.float32):
        return False

    num_source_blocks = block_representations.shape[0]
    num_queries = pseudo_queries.shape[0]
    return num_source_blocks <= 16 and num_queries <= 16


@lru_cache(maxsize=1)
def _load_extension():
    csrc_dir = Path(__file__).with_name("csrc")
    sources = [
        str(csrc_dir / "phase_1_backward.cpp"),
        str(csrc_dir / "phase_1_backward_kernel.cu"),
    ]

    return load(
        name="flash_attn_res_phase_1_backward_cuda",
        sources=sources,
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        ],
        verbose=os.getenv("FLASH_ATTN_RES_CUDA_VERBOSE", "0") == "1",
    )


def phase_1_backward_accumulate_cuda(
    block_representations: torch.Tensor,
    pseudo_queries: torch.Tensor,
    lses: torch.Tensor,
    inverse_rms_norms: torch.Tensor,
    attention_logits: torch.Tensor,
    grad_softmax_outputs: torch.Tensor,
    grad_lses: torch.Tensor,
    grad_block_representations: torch.Tensor,
    grad_pseudo_queries: torch.Tensor,
    has_grad_lses: bool,
    accumulate_grad_blocks: bool,
) -> bool:
    if not _supports_cuda_phase_1_backward(
        block_representations,
        pseudo_queries,
        grad_softmax_outputs,
    ):
        return False

    ext = _load_extension()
    ext.phase_1_backward(
        block_representations,
        pseudo_queries,
        lses,
        inverse_rms_norms,
        attention_logits,
        grad_softmax_outputs,
        grad_lses,
        grad_block_representations,
        grad_pseudo_queries,
        has_grad_lses,
        accumulate_grad_blocks,
    )
    return True

