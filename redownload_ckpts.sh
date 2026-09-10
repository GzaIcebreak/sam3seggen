#!/usr/bin/env bash
# Fresh SegviGen checkpoint download (no curl resume; resume was corrupting files).
set -euo pipefail
ROOT=/root/autodl-tmp/sam3seggen
CKPT="$ROOT/ckpt"
LOG=/root/autodl-tmp/tmp/dl-logs
mkdir -p "$CKPT" "$LOG"
export TMPDIR=/root/autodl-tmp/tmp

declare -A SIZE=(
  [full_seg.ckpt]=7860370137
  [full_seg_w_2d_map.ckpt]=7860370137
  [interactive_seg.ckpt]=7860389817
)

for name in full_seg.ckpt full_seg_w_2d_map.ckpt interactive_seg.ckpt; do
  dest="$CKPT/$name"
  expect="${SIZE[$name]}"
  if [[ -f "$dest" && $(stat -c%s "$dest") -eq "$expect" ]]; then
    echo "skip $name (already $expect)"
    continue
  fi
  rm -f "$dest" "$dest.part"
  echo "==== $(date +%H:%M:%S) start $name ===="
  ok=0
  for attempt in 1 2 3; do
    echo "  attempt $attempt/3"
    if curl -L --retry 20 --retry-delay 5 --retry-all-errors \
      --output "$dest.part" \
      "https://hf-mirror.com/fenghora/SegviGen/resolve/main/${name}"; then
      got=$(stat -c%s "$dest.part")
      echo "  got=$got expected=$expect"
      if [[ "$got" -eq "$expect" ]]; then
        mv -f "$dest.part" "$dest"
        echo "OK $name $got"
        ok=1
        break
      fi
    fi
    echo "  FAIL attempt $attempt"
    rm -f "$dest.part"
    sleep 5
  done
  if [[ "$ok" -ne 1 ]]; then
    echo "FAIL $name"
    exit 1
  fi
done

echo "==== all ckpt downloads finished ===="
ls -lh "$CKPT"
stat -c '%n %s' "$CKPT"/*.ckpt
