#!/usr/bin/env bash
# usage: finetune/run_ft.sh ext_bench.py montage --assets dog
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/env.sh"
exec /root/autodl-tmp/envs/trellis2/bin/python "$ROOT/finetune/$@"
