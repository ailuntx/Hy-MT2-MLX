# Hy-MT2 MLX

This directory contains helper scripts for converting and testing Hy-MT2 dense
models with `mlx-lm`.

## Supported first targets

- `Tencent-Hunyuan/Hy-MT2-1.8B`
- `Tencent-Hunyuan/Hy-MT2-7B`

Both use `model_type: hunyuan_v1_dense`, which is supported by recent `mlx-lm`
versions.

`Tencent-Hunyuan/Hy-MT2-30B-A3B` uses `model_type: hy_v3` and should be handled as a
separate MoE target after confirming MLX architecture support.

## Download official weights

Tencent publishes Hy-MT2 on both Hugging Face and ModelScope. Prefer
ModelScope for local downloads when available; it avoids HF/Xet routing and was
substantially faster for the 7B checkpoint in this workspace.

```bash
python -m pip install -U modelscope
modelscope download \
  --model Tencent-Hunyuan/Hy-MT2-7B \
  --local_dir /Volumes/usb_main/home/index_mlx/models/Hy-MT2-7B-official \
  --max-workers 4
```

## Convert

```bash
python scripts/convert_mlx.py \
  --hf-path /Volumes/usb_main/home/index_mlx/models/Hy-MT2-1.8B-official \
  --mlx-path /Volumes/usb_main/home/index_mlx/models/Hy-MT2-1.8B-bf16 \
  --dtype bfloat16
```

Quantized 4-bit:

```bash
python scripts/convert_mlx.py \
  --hf-path /Volumes/usb_main/home/index_mlx/models/Hy-MT2-1.8B-official \
  --mlx-path /Volumes/usb_main/home/index_mlx/models/Hy-MT2-1.8B-4bit \
  --dtype bfloat16 \
  --quantize --q-bits 4 --q-group-size 64
```

## Inference

```bash
python scripts/infer_mlx.py \
  --model /Volumes/usb_main/home/index_mlx/models/Hy-MT2-1.8B-bf16 \
  --prompt "Translate the following text into English. Note that you should only output the translated result without any additional explanation:\n\n今天天气真好。" \
  --max-tokens 128
```

## Stage for Hugging Face

```bash
python scripts/stage_mlx_repo.py \
  --mlx-path /Volumes/usb_main/home/index_mlx/models/Hy-MT2-1.8B-bf16 \
  --official-path /Volumes/usb_main/home/index_mlx/models/Hy-MT2-1.8B-official \
  --stage-path /Volumes/usb_main/home/index_mlx/huggingface/Hy-MT2-1.8B \
  --repo-id mlx-community/Hy-MT2-1.8B \
  --source-repo Tencent-Hunyuan/Hy-MT2-1.8B
```
