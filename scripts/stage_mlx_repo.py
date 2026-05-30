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
    front_matter = (
        "---\n"
        "license: apache-2.0\n"
        "library_name: mlx\n"
        "pipeline_tag: translation\n"
        "tags:\n"
        "- mlx\n"
        "- mlx-lm\n"
        "- hunyuan_v1_dense\n"
        "- translation\n"
        "---\n\n"
    )
    note = (
        "> [!IMPORTANT]\n"
        f"> This is an MLX conversion of `{source_repo}` for Apple Silicon.\n"
        "> The original model, license, and model card remain authoritative.\n"
        "> Use the upstream repository for non-MLX instructions and future official updates.\n\n"
        "## MLX Usage\n\n"
        "```bash\n"
        "pip install mlx-lm\n"
        f"mlx_lm.generate --model {repo_id} \\\n"
        "  --prompt \"Translate the following text into English. Note that you should only output the translated result without any additional explanation:\\n\\n今天天气真好。\" \\\n"
        "  --max-tokens 128 --temp 0.7 --top-p 0.6 --top-k 20\n"
        "```\n\n"
        "The default conversion follows the official 1.8B/7B inference settings where applicable: "
        "`temperature=0.7`, `top_p=0.6`, `top_k=20`, and `repetition_penalty=1.05`.\n\n"
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
