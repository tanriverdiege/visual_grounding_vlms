#!/usr/bin/env bash
# GPU box (Linux + CUDA) environment for VLM inference.
#
# Current cluster profile: Ubuntu 20.04, 8x Quadro RTX 5000 (sm_75, 16 GB
# each). Treat this as a pre-Ampere cluster: flash-attn is usually not worth
# forcing, and the safest first pass is the transformers lane without vLLM.
#
#   bash setup_gpu.sh          # transformers lane only
#   bash setup_gpu.sh --vllm   # also install the vLLM serving engine
#
# THE ONE RULE THAT MATTERS: never pre-install torch, and never `pip install -U
# torch` afterwards. vLLM pins an exact torch build and silently breaks its
# compiled kernels if torch is swapped underneath it. Let vLLM pull its own
# torch, then layer everything else on top.
set -euo pipefail

ENV_NAME="${ENV_NAME:-visual-grounding-vlms}"
WITH_VLLM=0
[[ "${1:-}" == "--vllm" ]] && WITH_VLLM=1

echo "==> creating conda env '$ENV_NAME' (python 3.11)"
conda create -n "$ENV_NAME" python=3.11 -y

run() { conda run -n "$ENV_NAME" --no-capture-output "$@"; }

run python -m pip install --upgrade pip

if [[ $WITH_VLLM -eq 1 ]]; then
  # vllm FIRST so it owns the torch pin. Its wheels are manylinux cp38-abi3,
  # so this step only works on Linux -- there is no macOS vLLM wheel.
  echo "==> installing vLLM (brings its own pinned torch + CUDA runtime)"
  run python -m pip install "vllm==0.28.0"
else
  # Pin to a CUDA 12.4 wheel set so pip does not pull a newer cu13x build
  # that is incompatible with this driver's 12.4 runtime.
  echo "==> installing torch (CUDA 12.4 build)"
  run python -m pip install \
    --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1
fi

echo "==> installing transformers lane + project deps"
run python -m pip install \
  "transformers==4.57.6" \
  accelerate "huggingface_hub[hf_transfer]" safetensors sentencepiece protobuf \
  pyyaml \
  qwen-vl-utils timm einops av \
  pillow "numpy<2.3" scipy opencv-python-headless pydicom matplotlib \
  datasets pandas tqdm

# flash-attn LAST: it compiles against whatever torch is already present, so
# installing it earlier would build against a torch that vLLM then replaces.
# Skip it entirely on pre-Ampere GPUs such as the RTX 5000s in this cluster.
if run python - <<'PY'
import torch, sys
if not torch.cuda.is_available():
    sys.exit(0)
major, minor = torch.cuda.get_device_capability(0)
sys.exit(0 if major < 8 else 1)
PY
then
  echo "==> skipping flash-attn on pre-Ampere GPUs (models will use sdpa)"
else
  echo "==> optional: flash-attn (Ampere+ only)"
  run python -m pip install flash-attn --no-build-isolation || \
    echo "    flash-attn failed; continuing without it (models fall back to sdpa)"
fi

echo
echo "==> verifying"
run python - <<'PY'
import torch, transformers
print("torch       ", torch.__version__)
print("CUDA avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      ", torch.cuda.get_device_name(0))
    cap = torch.cuda.get_device_capability(0)
    print("capability  ", f"sm_{cap[0]}{cap[1]}")
print("transformers", transformers.__version__)
try:
    import vllm; print("vllm        ", vllm.__version__)
except ImportError:
    print("vllm         (not installed)")
PY

echo
echo "done. conda activate $ENV_NAME"
