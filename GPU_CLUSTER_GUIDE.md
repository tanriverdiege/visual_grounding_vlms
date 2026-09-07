# GPU Cluster Guide

This guide is for the current Linux CUDA cluster, not the macOS development
machine.

## Cluster profile

- Ubuntu 20.04.6 LTS
- NVIDIA driver 550.144.03
- CUDA 12.4
- 8x Quadro RTX 5000 GPUs
- 16 GB VRAM per GPU
- Turing / sm_75, so this is pre-Ampere hardware

## Recommended path

1. Use Python 3.11.
2. Start with `bash setup_gpu.sh`.
3. Keep `--vllm` off until the transformers lane is working end to end.
4. The script now pins a CUDA 12.4 torch wheel for you, so it should not pull
  a CUDA 13.x build that your driver cannot use.
5. `flash-attn` is skipped automatically on this hardware.
6. Test with the 7B configs first.

## What to expect

- The repo does not currently shard one model across all 8 GPUs.
- The main wrapper moves a model onto a single device, so one 16 GB card is the
  practical unit of work.
- 7B models are the safest baseline.
- 8B and 13B models may require later work such as quantization, offload, or
  explicit multi-GPU support.
- `flash-attn` is not a required dependency on this cluster; SDPA fallback is
  expected.

## Suggested first check

After environment creation, verify the basics before attempting heavier runs:

```bash
cd /home/ege/visual_grounding_vlms
conda activate visual_grounding_vlms
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_capability(0))
PY
```

If this prints the RTX 5000 name and capability `(7, 5)`, the cluster matches
the assumptions in the repo.