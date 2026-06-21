#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/train_pix2pix.py" \
  --config "$PROJECT_ROOT/configs/pix2pix_maps.yaml" \
  --data-root "$PROJECT_ROOT/datasets/maps" \
  "$@"
