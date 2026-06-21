#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH="$PROJECT_ROOT/runs/pix2pix_maps/checkpoints/best.pt"
INPUT_PATH="${1:-$PROJECT_ROOT/datasets/maps/val}"
OUTPUT_DIR="$PROJECT_ROOT/runs/inference"

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  printf 'Checkpoint not found: %s\n' "$CHECKPOINT_PATH" >&2
  exit 1
fi

rm -rf "$OUTPUT_DIR"

shift $(( $# > 0 ? 1 : 0 ))

exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/infer_pix2pix.py" \
  --checkpoint "$CHECKPOINT_PATH" \
  --input-path "$INPUT_PATH" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
