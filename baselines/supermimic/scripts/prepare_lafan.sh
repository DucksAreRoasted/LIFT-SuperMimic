#!/usr/bin/env bash
# Server entry: sync uv environment, download ModelScope G1, retarget R1 + K1.
set -euo pipefail
BASELINE_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd -- "$BASELINE_ROOT/../.." && pwd)"
cd "$PROJECT_ROOT"
if ! command -v uv >/dev/null 2>&1; then
  echo 'uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/' >&2
  exit 2
fi
uv sync --extra retarget --no-dev --locked
exec uv run --extra retarget --no-dev --locked python "$BASELINE_ROOT/scripts/data/prepare_lafan_modelscope.py" "$@"
