#!/usr/bin/env python3
"""Convert small Gemma 4 Unified multimodal embedders to Core ML packages.

The text decoder stack in CoreMLProbe already consumes `[1, seq, 3840]` hidden
states. Gemma 4 Unified's encoder-free vision/audio paths are much smaller
frontends that project raw modality features into that same language-model
hidden width. This script converts those frontend projection modules with fixed
small shapes for iPhone smoke tests.
"""

from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto
import numpy as np
import torch
from transformers import AutoModelForImageTextToText


class VisionEmbedderWrapper(torch.nn.Module):
    def __init__(self, unified_model: torch.nn.Module):
        super().__init__()
        self.embed_vision = unified_model.embed_vision

    def forward(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_vision(pixel_values, image_position_ids)


class AudioEmbedderWrapper(torch.nn.Module):
    def __init__(self, unified_model: torch.nn.Module):
        super().__init__()
        self.embed_audio = unified_model.embed_audio

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.embed_audio(input_features)


def package_size(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for filename in files:
            total += os.path.getsize(os.path.join(root, filename))
    return total


def resolve_unified_model(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "embed_vision"):
        return model.model
    if hasattr(model, "embed_vision"):
        return model
    raise AttributeError("could not find Gemma4UnifiedModel with embed_vision")


def convert_package(
    *,
    name: str,
    wrapper: torch.nn.Module,
    example_inputs: tuple[torch.Tensor, ...],
    input_types: list[ct.TensorType],
    output_name: str,
    out_dir: Path,
    quant_config: cto.OptimizationConfig,
    block_size: int,
    keep_fp16: bool,
    force: bool,
) -> None:
    fp16_path = out_dir / f"{name}_fp16.mlpackage"
    int4_path = out_dir / f"{name}_int4_block{block_size}.mlpackage"

    if int4_path.exists() and not force:
        print("skip_existing", int4_path, "bytes", package_size(int4_path), flush=True)
        return

    print("convert_multimodal", name, flush=True)
    wrapper = wrapper.eval().cuda()
    with torch.inference_mode():
        output = wrapper(*example_inputs)
        print("forward_ok", name, tuple(output.shape), output.dtype, flush=True)
        traced = torch.jit.trace(wrapper, example_inputs, strict=False)

    traced = traced.eval().cpu()
    print("trace_ok", name, flush=True)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=input_types,
        outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=ct.precision.FLOAT16,
    )

    if keep_fp16:
        mlmodel.save(fp16_path)
        print("saved_fp16", fp16_path, package_size(fp16_path), flush=True)

    qmodel = cto.linear_quantize_weights(mlmodel, config=quant_config)
    qmodel.save(int4_path)
    print(
        "saved_int4",
        int4_path,
        "bytes",
        package_size(int4_path),
        "mib",
        round(package_size(int4_path) / 1024**2, 2),
        flush=True,
    )

    del qmodel, mlmodel, traced, wrapper, output
    gc.collect()
    torch.cuda.empty_cache()


def make_grid_position_ids(patch_count: int, *, device: str) -> torch.Tensor:
    side = int(np.ceil(np.sqrt(patch_count)))
    coords: list[list[int]] = []
    for index in range(patch_count):
        coords.append([index % side, index // side])
    return torch.tensor([coords], device=device, dtype=torch.int32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default="/workspace/gemma12b/models/gemma-4-12b-it-qat-q4_0-unquantized",
    )
    parser.add_argument(
        "--out-dir",
        default="/workspace/gemma12b/coreml-multimodal-int4",
    )
    parser.add_argument("--target", choices=["all", "image", "audio"], default="image")
    parser.add_argument("--image-patches", type=int, default=32)
    parser.add_argument("--audio-tokens", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument(
        "--model-dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Load/trace dtype. Use float32 if Core ML layer_norm dtype checks reject fp16 gamma/epsilon.",
    )
    parser.add_argument("--keep-fp16", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("coremltools", ct.__version__, flush=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
    print("loading", args.model_dir, flush=True)
    model_dtype = torch.float32 if args.model_dtype == "float32" else torch.float16
    np_model_dtype = np.float32 if args.model_dtype == "float32" else np.float16
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        dtype=model_dtype,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    unified_model = resolve_unified_model(model)
    text_config = model.config.get_text_config()
    vision_config = model.config.vision_config
    audio_config = model.config.audio_config

    print(
        "loaded_sec",
        round(time.time() - t0, 2),
        "cuda_alloc_gb",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
        "hidden",
        text_config.hidden_size,
        "vision_patch_dim",
        vision_config.model_patch_size**2 * 3,
        "vision_output_proj_dims",
        vision_config.output_proj_dims,
        "audio_output_proj_dims",
        audio_config.output_proj_dims,
        "model_dtype",
        args.model_dtype,
        flush=True,
    )

    quant_config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric",
            dtype="int4",
            granularity="per_block",
            block_size=args.block_size,
        )
    )

    if args.target in {"all", "image"}:
        patch_dim = vision_config.model_patch_size**2 * 3
        example_pixel_values = torch.zeros(
            (1, args.image_patches, patch_dim),
            device="cuda",
            dtype=model_dtype,
        )
        example_image_position_ids = make_grid_position_ids(args.image_patches, device="cuda")
        convert_package(
            name=f"gemma4_12b_image_embedder_patches{args.image_patches}",
            wrapper=VisionEmbedderWrapper(unified_model),
            example_inputs=(example_pixel_values, example_image_position_ids),
            input_types=[
                ct.TensorType(name="pixel_values", shape=example_pixel_values.shape, dtype=np_model_dtype),
                ct.TensorType(name="image_position_ids", shape=example_image_position_ids.shape, dtype=np.int32),
            ],
            output_name="image_hidden",
            out_dir=out_dir,
            quant_config=quant_config,
            block_size=args.block_size,
            keep_fp16=args.keep_fp16,
            force=args.force,
        )

    if args.target in {"all", "audio"}:
        example_input_features = torch.zeros(
            (1, args.audio_tokens, audio_config.output_proj_dims),
            device="cuda",
            dtype=model_dtype,
        )
        convert_package(
            name=f"gemma4_12b_audio_embedder_tokens{args.audio_tokens}",
            wrapper=AudioEmbedderWrapper(unified_model),
            example_inputs=(example_input_features,),
            input_types=[
                ct.TensorType(name="input_features", shape=example_input_features.shape, dtype=np_model_dtype),
            ],
            output_name="audio_hidden",
            out_dir=out_dir,
            quant_config=quant_config,
            block_size=args.block_size,
            keep_fp16=args.keep_fp16,
            force=args.force,
        )

    print("done", flush=True)


if __name__ == "__main__":
    main()
