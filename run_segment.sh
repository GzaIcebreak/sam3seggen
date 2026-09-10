#!/usr/bin/env bash
# Example: source env.sh first, then run this, or just execute it.
set -euo pipefail
ROOT=/root/autodl-tmp/sam3seggen
# shellcheck disable=SC1091
source "$ROOT/env.sh"
PY=/root/autodl-tmp/envs/trellis2/bin/python
exec "$PY" "$ROOT/segment_api.py" "$@"
