#!/bin/bash
# Sets up the "spinetk" conda environment for this repo.
# Target: NVIDIA H200 (Hopper, sm_90), driver 580.x (CUDA 13.0), Rocky Linux 9, gcc 11.
# The previous torch 1.8 + cu101 stack is kept in install_legacy_cu101.sh; it has
# no sm_90 kernels and fails with "no kernel image is available" on these GPUs.
#
# Usage (creates the env itself, no need to activate anything first):
#   bash install.sh
#   ENV_NAME=spinetk2 bash install.sh   # different env name
#
# Version choices, and why:
#   - python 3.9: same interpreter as the original stack, so code behavior is unchanged.
#   - torch 2.5.1 + cu124: oldest torch line with first-class Hopper support, and the
#     last release before torch.load() defaulted to weights_only=True (2.6), which
#     can refuse to load detectron2/fvcore checkpoints such as output/keypoint_cnn.pth.
#   - detectron2: no prebuilt wheels exist past torch 1.10, so it is compiled from a
#     pinned commit against the torch above. Compiling needs nvcc; it comes from the
#     NVIDIA conda channel (CUDA 12.4.1, matching torch's cu124) because this machine
#     has no system CUDA toolkit.
#   - numpy<2, pillow<10, opencv<4.10: same pins as the original stack, so array/image
#     behavior matches what the repo's code was written against.

set -euo pipefail

ENV_NAME="${ENV_NAME:-spinetk}"
DETECTRON2_COMMIT="1e3e13bbf607b54f62205c4c33922521822fb298"
# 9.0 = H200/H100; 7.5/8.0/8.6 keep the build usable on Turing/Ampere boxes too.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5;8.0;8.6;9.0}"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "==> conda env '$ENV_NAME' already exists, installing into it"
else
  echo "==> creating conda env '$ENV_NAME' (python 3.9)"
  conda create -n "$ENV_NAME" python=3.9 -y
fi

run() { conda run -n "$ENV_NAME" --no-capture-output "$@"; }
PREFIX="$(run python -c 'import sys; print(sys.prefix)')"

run python -m pip install --upgrade pip

echo "==> installing torch 2.5.1 (CUDA 12.4 build)"
run python -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1 torchvision==0.20.1

# Pins go in before detectron2 so its dependency resolution cannot bump them.
echo "==> installing pinned repo dependencies"
run python -m pip install \
  "pyyaml>=5.1,<7" "numpy<2" "pillow<10" "opencv-python-headless<4.10" \
  pandas scikit-learn matplotlib ninja

echo "==> installing nvcc + CUDA 12.4.1 headers into the env (build-time only)"
conda install -n "$ENV_NAME" -y -c "nvidia/label/cuda-12.4.1" \
  cuda-nvcc cuda-cudart-dev cuda-libraries-dev

echo "==> building detectron2 @ ${DETECTRON2_COMMIT:0:7} for archs $TORCH_CUDA_ARCH_LIST"
export CUDA_HOME="$PREFIX"
export FORCE_CUDA=1
export CPATH="$PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$PREFIX/targets/x86_64-linux/lib:$PREFIX/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export MAX_JOBS="${MAX_JOBS:-16}"
run python -m pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/detectron2.git@${DETECTRON2_COMMIT}"

echo
echo "==> verifying"
run python - <<'PY'
import numpy, PIL, cv2, torch, detectron2
from detectron2 import _C  # compiled CUDA ops
from detectron2.layers import nms
print("torch       ", torch.__version__, "(CUDA", torch.version.cuda + ")")
print("detectron2  ", detectron2.__version__)
print("numpy/pillow/cv2", numpy.__version__, PIL.__version__, cv2.__version__)
print("CUDA avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      ", torch.cuda.get_device_name(0))
    b = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11.]], device="cuda")
    keep = nms(b, torch.tensor([0.9, 0.8], device="cuda"), 0.5)
    print("GPU nms     ", "ok" if keep.tolist() == [0] else f"unexpected {keep.tolist()}")
PY

echo
echo "Install complete. conda activate $ENV_NAME"
