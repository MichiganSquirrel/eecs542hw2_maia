#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but not found."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | rg -q "^maia\\s"; then
  echo "Conda env 'maia' already exists. Reusing it."
else
  echo "Creating conda env 'maia' from environment.yml ..."
  conda env create -f environment.yml
fi

conda activate maia

if ! command -v uv >/dev/null 2>&1; then
  pip install uv
fi

echo "Installing Python dependencies ..."
uv pip install -e .
uv pip install google-genai

echo "Downloading precomputed exemplars ..."
bash download_exemplars.sh

python3 - <<'PY'
import os
import torch

print("Python environment check passed.")
print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("HF_TOKEN set:", bool(os.getenv("HF_TOKEN")))
print("GEMINI_API_KEY set:", bool(os.getenv("GEMINI_API_KEY")))
PY

echo "Setup finished."
