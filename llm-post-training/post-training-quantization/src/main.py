# https://colab.research.google.com/github/pytorch/tutorials/blob/gh-pages/_downloads/6011ce860c893c5ee96624c75d548133/gpu_quantization_torchao_tutorial.ipynb

import logging
from collections.abc import Callable
from typing import Any

import torch
from segment_anything import sam_model_registry
from torch import nn
from torch.utils.benchmark import Timer
from torchao.quantization.quant_api import (
    Int8DynamicActivationInt8WeightConfig,
    quantize_,
)

logger = logging.getLogger(__name__)


@torch.no_grad()
def benchmark(
    f: Callable[..., Any],
    *args: object,
    **kwargs: object,
) -> dict[str, float]:
    """Benchmark function that measures runtime and peak GPU memory usage."""
    # Warmup runs to ensure stable measurements
    for _ in range(3):
        f(*args, **kwargs)
    # Synchronize CUDA operations and reset memory stats for accurate measurement
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    # Create timer and run adaptive benchmark
    t0: Timer = Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
    )
    res = t0.adaptive_autorange(0.03, min_run_time=0.2, max_run_time=20)
    return {"time": res.median * 1e3, "memory": torch.cuda.max_memory_allocated() / 1e9}


def get_sam_model(
    sam_checkpoint_base_path: str,
    model_type: str,
    model_name: str,
    batch_size: int,
    only_one_block: bool,
) -> tuple[nn.Module, torch.Tensor]:
    """Load SAM model and create appropriate input tensor."""
    checkpoint_path = f"{sam_checkpoint_base_path}/{model_name}"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path).cuda()
    # Focus on image encoder (statically sized inputs)
    model: nn.Module = sam.image_encoder.eval()
    image: torch.Tensor = torch.randn(batch_size, 3, 1024, 1024, device="cuda")
    # Option to test single transformer block for detailed analysis
    if only_one_block:
        model = model.blocks[0]
        # Adjust input dimensions for single block (after patch embedding)
        image = torch.randn(batch_size, 64, 64, 1280, device="cuda")
    return model, image


def run_fp32_baseline(
    sam_checkpoint_base_path: str,
    model_type: str,
    model_name: str,
    batch_size: int,
    only_one_block: bool,
) -> None:
    """Run baseline FP32 performance test."""
    logger.info("=== Step 1: Baseline FP32 Performance ===")
    try:
        model, image = get_sam_model(
            sam_checkpoint_base_path,
            model_type,
            model_name,
            batch_size,
            only_one_block,
        )
        fp32_res: dict[str, float] = benchmark(model, image)
        logger.info(
            f"base fp32 runtime of the model is {fp32_res['time']:0.2f}ms and peak memory {fp32_res['memory']:0.2f}GB",
        )
        # Expected: ~186ms runtime, ~6.33GB memory
    except Exception:
        logger.exception("unable to run fp32 model")


def run_bfloat16_optimization(
    sam_checkpoint_base_path: str,
    model_type: str,
    model_name: str,
    batch_size: int,
    only_one_block: bool,
) -> dict[str, float]:
    """Run BFloat16 conversion optimization."""
    logger.info("\n=== Step 2: Convert to BFloat16 (7x speedup) ===")
    # BFloat16 provides better dynamic range than FP16 (same 8 exp bits as FP32)
    # This prevents overflow issues during quantization scaling operations
    model, image = get_sam_model(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )
    model = model.to(torch.bfloat16)  # Convert model weights to bfloat16
    image = image.to(torch.bfloat16)  # Convert input to bfloat16
    bf16_res: dict[str, float] = benchmark(model, image)
    logger.info(
        f"bf16 runtime of the block is {bf16_res['time']:0.2f}ms and peak memory {bf16_res['memory']: 0.2f}GB",
    )
    # Expected: ~25ms runtime (7x improvement), ~3.17GB memory
    return bf16_res


def run_torch_compile_optimization(
    sam_checkpoint_base_path: str,
    model_type: str,
    model_name: str,
    batch_size: int,
    only_one_block: bool,
) -> dict[str, float]:
    """Run torch.compile optimization."""
    logger.info(
        "\n=== Step 3: Add torch.compile with max-autotune (27% additional speedup) ===",
    )
    # torch.compile with max-autotune enables aggressive kernel optimization
    # First run will show AUTOTUNE output as it searches for optimal kernel parameters
    model, image = get_sam_model(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )
    model = model.to(torch.bfloat16)
    image = image.to(torch.bfloat16)
    model_c: nn.Module = torch.compile(model, mode="max-autotune")
    comp_res: dict[str, float] = benchmark(model_c, image)
    logger.info(
        f"bf16 compiled runtime of the block is {comp_res['time']:0.2f}ms and peak memory {comp_res['memory']: 0.2f}GB",
    )
    # Expected: ~20ms runtime (27% improvement), ~2.24GB memory
    return comp_res


def run_int8_quantization(
    sam_checkpoint_base_path: str,
    model_type: str,
    model_name: str,
    batch_size: int,
    only_one_block: bool,
) -> dict[str, float]:
    """Run INT8 dynamic quantization."""
    logger.info("\n=== Step 4: Apply INT8 Dynamic Quantization ===")
    # Dynamic quantization: activations quantized at runtime, weights pre-quantized
    # Good for compute-bound models like transformers
    model, image = get_sam_model(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )
    model = model.to(torch.bfloat16)
    image = image.to(torch.bfloat16)
    # Quantization for GPUs comes in three main forms in torchao which is just native pytorch+python code. This includes:
    # - int8 dynamic quantization: Int8DynamicActivationInt8WeightConfig()
    # - int8 weight-only quantization: Int8WeightOnlyConfig()
    # - int4 weight-only quantization: Int4WeightOnlyConfig()
    # Different models, or sometimes different layers in a model can require different techniques.
    # For models which are heavily compute bound, dynamic quantization tends to work the best since it swaps the normal expensive floating point matmul ops with integer versions.
    # Weight-only quantization works better in memory bound situations where the benefit comes from loading less weight data, rather than doing less computation.
    #
    # Apply INT8 dynamic quantization to linear layers
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    model_c = torch.compile(model, mode="max-autotune")
    quant_res: dict[str, float] = benchmark(model_c, image)
    logger.info(
        f"bf16 compiled runtime of the quantized block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB",
    )
    # Expected: ~19ms runtime (small improvement), ~3.58GB memory (increased due to int32 intermediate results)
    return quant_res


def run_fused_int8_matmul(
    sam_checkpoint_base_path: str,
    model_type: str,
    model_name: str,
    batch_size: int,
    only_one_block: bool,
) -> dict[str, float]:
    """Run INT8 MatMul fusion optimization."""
    logger.info("\n=== Step 5: Enable INT8 MatMul Fusion (reduces memory overhead) ===")
    # Fuse int8 matmul with subsequent rescale operation to avoid storing int32 intermediates
    # This reduces memory usage significantly while maintaining performance
    model, image = get_sam_model(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )
    model = model.to(torch.bfloat16)
    image = image.to(torch.bfloat16)
    # With quantization, we have improved performance a bit more but memory usage increased significantly.
    # This is for two reasons:
    # 1) Quantization adds overhead to the model since we need to quantize and dequantize the input and output.
    #    For small batch sizes this overhead can actually make the model go slower.
    # 2) Even though we are doing a quantized matmul, such as int8 x int8, the result of the multiplication gets stored in an int32 tensor which is twice the size of the result from the non-quantized model.
    #    If we can avoid creating this int32 tensor, our memory usage will improve a lot.
    # We can fix #2 by fusing the integer matmul with the subsequent rescale operation since the final output will be bf16, if we immediately convert the int32 tensor to bf16 and instead store that we'll get better performance in terms of both runtime and memory.
    # Enable fusion of int8 matrix multiplication with rescaling operation
    torch._inductor.config.force_fuse_int_mm_with_mul = True  # noqa: SLF001
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    model_c = torch.compile(model, mode="max-autotune")
    quant_res = benchmark(model_c, image)
    logger.info(
        f"bf16 compiled runtime of the fused quantized block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB",
    )
    # Expected: ~18.78ms runtime (6% total improvement), ~2.37GB memory (much better)
    return quant_res


def run_advanced_compiler_tuning(
    sam_checkpoint_base_path: str,
    model_type: str,
    model_name: str,
    batch_size: int,
    only_one_block: bool,
) -> dict[str, float]:
    """Run advanced compiler optimizations."""
    logger.info("\n=== Step 6: Apply Advanced Compiler Optimizations ===")
    # Additional compiler optimizations for maximum performance
    model, image = get_sam_model(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )
    model = model.to(torch.bfloat16)
    image = image.to(torch.bfloat16)
    torch._inductor.config.force_fuse_int_mm_with_mul = True  # noqa: SLF001
    # Disabling epilogue fusion can sometimes improve performance since the autotuning process can be confused by fusions and choose bad kernel parameters.
    torch._inductor.config.epilogue_fusion = False  # noqa: SLF001
    # Apply coordinate descent tuning in all directions to enlarge the search area for kernel parameters.
    torch._inductor.config.coordinate_descent_tuning = True  # noqa: SLF001
    torch._inductor.config.coordinate_descent_check_all_directions = True  # noqa: SLF001
    quantize_(model, Int8DynamicActivationInt8WeightConfig())
    model_c = torch.compile(model, mode="max-autotune")
    quant_res = benchmark(model_c, image)
    logger.info(
        f"bf16 compiled runtime of the final quantized block is {quant_res['time']:0.2f}ms and peak memory {quant_res['memory']: 0.2f}GB",
    )
    # Expected: ~18.16ms runtime (10x total improvement from original), ~2.49GB memory
    return quant_res


def main() -> None:
    # Configuration parameters for SAM model
    sam_checkpoint_base_path = "data"
    model_type = "vit_h"  # Vision Transformer Huge variant
    model_name = "sam_vit_h_4b8939.pth"
    batch_size = 16
    only_one_block = True  # Focus on single transformer block for easier analysis

    # ==================== BASELINE: FP32 PERFORMANCE ====================
    run_fp32_baseline(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )

    # ==================== OPTIMIZATION 1: BFLOAT16 CONVERSION ====================
    run_bfloat16_optimization(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )

    # ==================== OPTIMIZATION 2: TORCH.COMPILE ====================
    run_torch_compile_optimization(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )

    # ==================== OPTIMIZATION 3: INT8 DYNAMIC QUANTIZATION ====================
    run_int8_quantization(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )

    # ==================== OPTIMIZATION 4: FUSE INT8 MATMUL WITH RESCALE ====================
    run_fused_int8_matmul(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )

    # ==================== OPTIMIZATION 5: ADVANCED COMPILER TUNING ====================
    run_advanced_compiler_tuning(
        sam_checkpoint_base_path,
        model_type,
        model_name,
        batch_size,
        only_one_block,
    )


if __name__ == "__main__":
    logging.basicConfig(
        force=True,
        level=logging.INFO,
        format="%(message)s",
    )
    main()
