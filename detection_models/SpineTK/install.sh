#!/bin/bash
# Sets up the "spinetk" conda environment for this repo.
# Tested with: Python 3.9, NVIDIA driver 550.x (CUDA 12.4), Turing-class GPUs (e.g. Quadro RTX 5000).
#
# Usage:
#   conda create -n spinetk python=3.9 -y
#   conda activate spinetk
#   ./install.sh

set -e

pip install pyyaml==5.1

# workaround: install old version of pytorch since detectron2 hasn't released packages for pytorch 1.9 (issue: https://github.com/facebookresearch/detectron2/issues/3158)
pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

# install detectron2 that matches pytorch 1.8
# See https://detectron2.readthedocs.io/tutorials/install.html for instructions
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html

# --- pins required to make the above stack work with modern package resolvers ---
# numpy>=2 breaks torch 1.8's compiled extensions (ABI change); torchvision/detectron2
# also pull in a numpy>=2-requiring opencv/pillow unless pinned below.
pip install "numpy<2"
# Pillow >=10 removed the legacy Image.LINEAR/BILINEAR/etc. constants that
# detectron2 0.6 relies on internally.
pip install "pillow<10"

# other dependencies used directly by this repo's scripts
pip install "opencv-python-headless<4.10" pandas scikit-learn

echo "Install complete. Verify with: python -c 'import torch, detectron2; print(torch.__version__, torch.cuda.is_available(), detectron2.__version__)'"
