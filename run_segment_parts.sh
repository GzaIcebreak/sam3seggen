#!/usr/bin/env bash
# Part extraction: over-segment with several full_seg samples, name with SAM3 voting.
# Example: ./run_segment_parts.sh --glb monk.glb --prompts head arm leg --out out/parts.glb
set -euo pipefail
ROOT=/root/autodl-tmp/sam3seggen
# shellcheck disable=SC1091
source "$ROOT/env.sh"
PY=/root/autodl-tmp/envs/trellis2/bin/python
exec "$PY" "$ROOT/segment_parts.py" "$@"
