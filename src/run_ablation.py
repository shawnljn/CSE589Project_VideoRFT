"""
Run a small grid of decoding configs and save separate eval outputs.

Usage:
  python src/run_ablation.py --model_path checkpoints/VideoRFT-3B --file_name videommmu_subset

It will launch eval_bench.py multiple times with different TEMP/TOP_P/MAX_TOKENS
and write outputs with suffixes like eval_*_<file_name>_<tag>.json.
Edit the CONFIGS list to add/remove settings.
"""

import argparse
import os
import subprocess
from pathlib import Path


CONFIGS = [
    {"tag": "greedy", "TEMP": "0.0", "TOP_P": "0.0", "MAX_TOKENS": "1024"},
    {"tag": "t01_p001", "TEMP": "0.1", "TOP_P": "0.001", "MAX_TOKENS": "1024"},
    {"tag": "t01_p01", "TEMP": "0.1", "TOP_P": "0.01", "MAX_TOKENS": "1024"},
    {"tag": "t02_p05", "TEMP": "0.2", "TOP_P": "0.05", "MAX_TOKENS": "1024"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--file_name", required=True, help="Base file_name arg passed to eval_bench.py")
    parser.add_argument("--python", default="python", help="Python executable to use")
    parser.add_argument("--extra-env", nargs="*", default=[], help="Additional ENV assignments KEY=VAL")
    args = parser.parse_args()

    base_env = os.environ.copy()
    for pair in args.extra_env:
        if "=" in pair:
            k, v = pair.split("=", 1)
            base_env[k] = v

    script = Path(__file__).parent / "eval_bench.py"
    for cfg in CONFIGS:
        tag = cfg["tag"]
        env = base_env.copy()
        env["TEMP"] = cfg["TEMP"]
        env["TOP_P"] = cfg["TOP_P"]
        env["MAX_TOKENS"] = cfg["MAX_TOKENS"]
        run_name = f"{args.file_name}_{tag}"

        cmd = [
            args.python,
            str(script),
            "--model_path", args.model_path,
            "--file_name", run_name,
        ]
        print(f"\n=== Running config {tag} (TEMP={cfg['TEMP']}, TOP_P={cfg['TOP_P']}, MAX_TOKENS={cfg['MAX_TOKENS']}) ===")
        print(" ".join(cmd))
        try:
            subprocess.run(cmd, check=True, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[warn] run {tag} failed: {e}")


if __name__ == "__main__":
    main()
