#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH="$PROJECT_ROOT/runs/pix2pix_maps/checkpoints/best.pt"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  printf 'Best checkpoint not found: %s\n' "$CHECKPOINT_PATH" >&2
  exit 1
fi

exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/train_pix2pix.py" \
  --config "$PROJECT_ROOT/configs/pix2pix_maps.yaml" \
  --data-root "$PROJECT_ROOT/datasets/maps" \
  --resume "$CHECKPOINT_PATH" \
  "$@"
