#!/usr/bin/env python3
"""Convert Hy-MT2 dense checkpoints to MLX format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlx_lm.convert import convert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hf-path",
        default="tencent/Hy-MT2-1.8B",
        help="Hugging Face repo id or local official checkpoint path.",
    )
    parser.add_argument(
        "--mlx-path",
        required=True,
        help="Output directory for the MLX checkpoint. It must not already exist.",
    )
    parser.add_argument(
        "--dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Non-quantized parameter dtype.",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--q-bits", type=int, choices=[2, 3, 4, 6, 8], default=4)
    parser.add_argument("--q-group-size", type=int, default=64)
    parser.add_argument(
        "--q-mode",
        choices=["affine", "mxfp4", "nvfp4", "mxfp8"],
        default="affine",
    )
    parser.add_argument(
        "--quant-predicate",
        choices=["mixed_2_6", "mixed_3_4", "mixed_3_6", "mixed_4_6"],
        default=None,
        help="Optional mixed-bit recipe supported by mlx-lm.",
    )
    return parser.parse_args()


def write_manifest(args: argparse.Namespace, out_dir: Path) -> None:
    manifest = {
        "source_model": args.hf_path,
        "format": "mlx-lm",
        "dtype": args.dtype,
        "quantized": args.quantize,
        "q_bits": args.q_bits if args.quantize else None,
        "q_group_size": args.q_group_size if args.quantize else None,
        "q_mode": args.q_mode if args.quantize else None,
        "quant_predicate": args.quant_predicate,
    }
    (out_dir / "mlx_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.mlx_path).expanduser().resolve()
    convert(
        hf_path=args.hf_path,
        mlx_path=str(out_dir),
        dtype=args.dtype,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        quantize=args.quantize,
        q_bits=args.q_bits if args.quantize else None,
        q_group_size=args.q_group_size if args.quantize else None,
        q_mode=args.q_mode,
        quant_predicate=args.quant_predicate,
    )
    write_manifest(args, out_dir)
    print(f"Wrote MLX checkpoint to {out_dir}")


if __name__ == "__main__":
    main()
