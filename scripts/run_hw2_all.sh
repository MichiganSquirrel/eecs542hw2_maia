#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "Running HW2 ResNet task ..."
bash scripts/run_hw2_resnet.sh

echo "Running HW2 DINO task ..."
bash scripts/run_hw2_dino.sh

echo "Running HW2 CLIP task ..."
bash scripts/run_hw2_clip.sh

echo "Done. Check outputs under ./results/gemini-2.5-flash/"
