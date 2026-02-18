#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

AGENT="${AGENT:-gemini-2.5-flash}"
DEVICE="${DEVICE:-0}"
PROMPTS_PATH="${PROMPTS_PATH:-./prompts/gemini}"
SAVE_PATH="${SAVE_PATH:-./results}"

# Avoids both banned and reference nodes by default.
DINO_LAYER="${DINO_LAYER:-blocks.7.mlp.fc1}"
DINO_UNIT="${DINO_UNIT:-37}"

if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  echo "GEMINI_API_KEY is not set."
  exit 1
fi

python3 main.py \
  --agent "${AGENT}" \
  --model dino_vits8 \
  --unit_mode manual \
  --units "${DINO_LAYER}=${DINO_UNIT}" \
  --device "${DEVICE}" \
  --path2prompts "${PROMPTS_PATH}" \
  --path2save "${SAVE_PATH}" \
  "$@"
