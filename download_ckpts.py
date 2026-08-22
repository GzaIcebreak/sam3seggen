"""Download SegviGen checkpoints via hf-mirror + curl (resume-friendly)."""
import os
import subprocess
import sys

CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
FILES = {
    "full_seg.ckpt": 7860370137,
    "full_seg_w_2d_map.ckpt": 7860370137,
    "interactive_seg.ckpt": 7860389817,
}
BASE = "https://hf-mirror.com/fenghora/SegviGen/resolve/main"


def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    for name, expected in FILES.items():
        dest = os.path.join(CKPT_DIR, name)
        if os.path.isfile(dest) and os.path.getsize(dest) == expected:
            print(f"skip {name} ({expected} bytes)")
            continue
        url = f"{BASE}/{name}"
        print(f"downloading {name} -> {dest}")
        cmd = [
            "curl.exe", "-L",
            "--retry", "40",
            "--retry-delay", "5",
            "--retry-all-errors",
            "-C", "-",
            "--output", dest,
            url,
        ]
        subprocess.check_call(cmd)
        size = os.path.getsize(dest)
        print(f"  -> {size} bytes")
        if size != expected:
            raise SystemExit(f"size mismatch for {name}: got {size}, expected {expected}")
    print("DONE")


if __name__ == "__main__":
    main()
