#!/usr/bin/env python3
"""Stage a Hy-MT2 MLX checkpoint for Hugging Face upload."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlx-path", required=True, help="Converted MLX checkpoint.")
    parser.add_argument("--official-path", required=True, help="Official HF snapshot.")
    parser.add_argument("--stage-path", required=True, help="HF staging directory.")
    parser.add_argument("--repo-id", required=True, help="Target HF repo id.")
    parser.add_argument("--source-repo", default="tencent/Hy-MT2-1.8B")
    parser.add_argument("--default-repo", default=None)
    return parser.parse_args()


def copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def strip_front_matter(text: str) -> str:
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    return text[end + len("\n---\n") :].lstrip()


def build_readme(official_readme: str, repo_id: str, source_repo: str) -> str:
    repo_name = repo_id.split("/", 1)[-1]
    variant_key = repo_name.replace("Hy-MT2-", "")
    variant_label = variant_key.replace("-4bit", " 4-bit").replace("-8bit", " 8-bit").replace(
        "-bfloat16", " bfloat16"
    )
    best_for = {
        "1.8B-bfloat16": "high-quality 1.8B baseline",
        "1.8B-8bit": "smaller 1.8B checkpoint",
        "1.8B-4bit": "smallest 1.8B checkpoint and Space load smoke test",
        "7B-bfloat16": "high-precision 7B conversion",
        "7B-8bit": "7B size/quality middle ground",
        "7B-4bit": "smallest 7B checkpoint",
    }.get(variant_key, "Apple Silicon MLX inference")
    front_matter = (
        "---\n"
        "license: apache-2.0\n"
        "library_name: mlx\n"
        "pipeline_tag: translation\n"
        "base_model:\n"
        f"- {source_repo}\n"
        "tags:\n"
        "- mlx\n"
        "- apple-silicon\n"
        "- mlx-lm\n"
        "- translation\n"
        "- hunyuan_v1_dense\n"
        "language:\n"
        "- multilingual\n"
        "---\n\n"
    )
    note = (
        f"Part of the [Hy-MT2 MLX](https://huggingface.co/collections/mlx-community/hy-mt2-6a15a173a4e2d27031541558) collection.\n\n"
        f"# {repo_name} (MLX)\n\n"
        f"Apple MLX weights for [{source_repo}](https://github.com/Tencent-Hunyuan/Hy-MT2), "
        "Tencent Hunyuan's multilingual translation model.\n\n"
        "## TL;DR\n\n"
        "| | |\n"
        "|---|---|\n"
        f"| **Variant** | {variant_label} |\n"
        f"| **Best for** | {best_for} |\n"
        "| **Runtime** | [`mlx-lm`](https://github.com/ml-explore/mlx-lm) |\n"
        "| **Official code** | [`Tencent-Hunyuan/Hy-MT2`](https://github.com/Tencent-Hunyuan/Hy-MT2) |\n"
        "| **MLX code** | [`ailuntx/Hy-MT2-MLX`](https://github.com/ailuntx/Hy-MT2-MLX) |\n"
        "| **Hardware** | Apple Silicon recommended; HF Spaces CPU fallback is only a load smoke test |\n\n"
        "## Quick Start\n\n"
        "```bash\n"
        "pip install mlx-lm\n"
        "PROMPT=$'Translate the following text into English. Note that you should only output the translated result without any additional explanation:\\n\\n今天天气真好。'\n"
        f"mlx_lm.generate --model {repo_id} \\\n"
        "  --prompt \"$PROMPT\" \\\n"
        "  --max-tokens 128 \\\n"
        "  --temp 0.7 \\\n"
        "  --top-p 0.6 \\\n"
        "  --top-k 20\n"
        "```\n\n"
        "For conversion and staging tools:\n\n"
        "```bash\n"
        "git clone https://github.com/ailuntx/Hy-MT2-MLX.git\n"
        "cd Hy-MT2-MLX\n"
        "python scripts/infer_mlx.py --model /path/to/mlx/checkpoint --text \"今天天气真好。\" --target-lang English\n"
        "```\n\n"
        "## Variants\n\n"
        "| Variant | Best for |\n"
        "|---|---|\n"
        "| [`Hy-MT2-1.8B-bfloat16`](https://huggingface.co/mlx-community/Hy-MT2-1.8B-bfloat16) | high-quality 1.8B baseline |\n"
        "| [`Hy-MT2-1.8B-8bit`](https://huggingface.co/mlx-community/Hy-MT2-1.8B-8bit) | smaller 1.8B checkpoint |\n"
        "| [`Hy-MT2-1.8B-4bit`](https://huggingface.co/mlx-community/Hy-MT2-1.8B-4bit) | smallest 1.8B checkpoint |\n"
        "| [`Hy-MT2-7B-bfloat16`](https://huggingface.co/mlx-community/Hy-MT2-7B-bfloat16) | high-precision 7B conversion |\n"
        "| [`Hy-MT2-7B-8bit`](https://huggingface.co/mlx-community/Hy-MT2-7B-8bit) | 7B size/quality middle ground |\n"
        "| [`Hy-MT2-7B-4bit`](https://huggingface.co/mlx-community/Hy-MT2-7B-4bit) | smallest 7B checkpoint |\n\n"
        "## Conversion Notes\n\n"
        "| Component | Source | MLX handling |\n"
        "|---|---|---|\n"
        "| model weights | official dense Hy-MT2 checkpoint | converted with MLX/`mlx-lm` tooling |\n"
        "| tokenizer/config | official checkpoint | copied through for `mlx-lm` loading |\n"
        "| quantized variants | bfloat16 MLX baseline | derived with MLX quantization settings |\n\n"
        "## Validation\n\n"
        "Local Apple Silicon is the intended runtime. The Hy-MT2 HF Space starts and loads the model on Linux CPU fallback, "
        "but `cpu-basic` can exceed request timeouts even for very small generation tests.\n\n"
        "## License and Citation\n\n"
        "License follows the upstream Tencent Hunyuan release. Cite the original Hy-MT2 project for the model and cite "
        "[`ailuntx/Hy-MT2-MLX`](https://github.com/ailuntx/Hy-MT2-MLX) for this MLX conversion tooling.\n\n"
        "## Original Model Card\n\n"
    )
    return front_matter + note + strip_front_matter(official_readme)


def main() -> None:
    args = parse_args()
    mlx_path = Path(args.mlx_path).expanduser().resolve()
    official_path = Path(args.official_path).expanduser().resolve()
    stage_path = Path(args.stage_path).expanduser().resolve()
    copy_tree(mlx_path, stage_path)

    for name in ["LICENSE.txt", "README_CN.md"]:
        src = official_path / name
        if src.exists():
            shutil.copy2(src, stage_path / name)

    for subdir in ["imgs"]:
        src = official_path / subdir
        if src.exists():
            dst = stage_path / subdir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)

    official_readme = (official_path / "README.md").read_text(encoding="utf-8")
    (stage_path / "README.md").write_text(
        build_readme(official_readme, args.repo_id, args.source_repo),
        encoding="utf-8",
    )

    manifest_path = stage_path / "mlx_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target_repo"] = args.repo_id
    manifest["source_repo"] = args.source_repo
    if args.default_repo:
        manifest["default_repo"] = args.default_repo
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Staged {args.repo_id} at {stage_path}")


if __name__ == "__main__":
    main()
