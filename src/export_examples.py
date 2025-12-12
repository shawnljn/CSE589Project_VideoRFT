"""
Export a few qualitative examples from an eval results JSON.

For each selected sample, save:
  - a first video frame PNG (if the video is available)
  - a markdown summary with question, options, CoT, prediction, and ground truth

Usage:
  python src/export_examples.py \
    --input eval_results/eval_videommmu_videommmu_subset_greedy_output.json \
    --output-dir eval_results/examples \
    --num 6
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict


def extract_answer(text: str):
    import re

    pattern = r"<answer>\s*(.*?)\s*</answer>"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else text


def extract_think(text: str):
    import re

    pattern = r"<think>\s*(.*?)\s*</think>"
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip() if m else ""


def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", [])


def select_examples(results: List[Dict], num: int):
    correct = [r for r in results if r.get("reward") == 1.0 or r.get("correct")]
    incorrect = [r for r in results if not (r.get("reward") == 1.0 or r.get("correct"))]

    out = []
    half = num // 2
    out.extend(correct[:half])
    out.extend(incorrect[: num - len(out)])
    if len(out) < num:
        out.extend(results[: num - len(out)])
    return out[:num]


def ensure_video_path(sample_path: str):
    # Make absolute if possible and try /newdata1 prefix to match eval usage.
    p = Path(sample_path)
    if p.is_absolute():
        return p
    rel = Path(sample_path.lstrip("./"))
    if rel.exists():
        return rel.resolve()
    pref = Path("/newdata1") / rel
    return pref


def save_first_frame(video_path: Path, out_path: Path):
    try:
        import decord

        vr = decord.VideoReader(str(video_path))
        frame = vr[0].asnumpy()  # HWC, RGB
        from PIL import Image

        Image.fromarray(frame).save(out_path)
        return True, None
    except Exception as e_decord:
        try:
            from torchvision.io import read_video
            import torch

            video, _, _ = read_video(str(video_path), start_pts=0, end_pts=None)
            if video.numel() == 0:
                return False, f"torchvision returned empty tensor for {video_path}"
            frame0 = video[0].to(torch.uint8).cpu().numpy()  # T,H,W,C
            from PIL import Image

            Image.fromarray(frame0).save(out_path)
            return True, None
        except Exception as e_tv:
            return False, f"decord error: {e_decord}; torchvision error: {e_tv}"


def main():
    parser = argparse.ArgumentParser(description="Export qualitative examples.")
    parser.add_argument("--input", type=Path, required=True, help="Eval results JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results/examples"))
    parser.add_argument("--num", type=int, default=6, help="Number of samples to export.")
    args = parser.parse_args()

    results = load_results(args.input)
    samples = select_examples(results, args.num)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    md_lines = ["# Qualitative Examples", ""]

    for idx, r in enumerate(samples, 1):
        q = r.get("problem", "")
        if r.get("problem_type") == "multiple choice":
            opts = r.get("options", [])
            q = q + "\n" + "\n".join(opts)

        gt = extract_answer(r.get("solution", ""))
        pred = r.get("prediction") or extract_answer(r.get("output", ""))
        think = ""
        if "process" in r and r["process"]:
            think = extract_think(r["process"])
        if not think:
            think = extract_think(r.get("output", ""))

        video_path = ensure_video_path(r.get("path", ""))
        frame_file = None
        frame_status = ""
        if video_path.exists():
            frame_file = out_dir / f"example_{idx}.png"
            ok, msg = save_first_frame(video_path, frame_file)
            if not ok:
                frame_status = f"(frame extraction failed: {msg})"
                frame_file = None
        else:
            frame_status = f"(video not found: {video_path})"

        md_lines.append(f"## Example {idx}")
        if frame_file:
            md_lines.append(f"![frame]({frame_file.name})")
        elif frame_status:
            md_lines.append(frame_status)
        md_lines.append("")
        md_lines.append(f"**Path:** {r.get('path','')}")
        md_lines.append(f"**Problem ID:** {r.get('problem_id','')}")
        md_lines.append("")
        md_lines.append("**Question:**")
        md_lines.append("")
        md_lines.append(q)
        md_lines.append("")
        md_lines.append("**Model reasoning (CoT):**")
        md_lines.append("")
        md_lines.append(think if think else "(none)")
        md_lines.append("")
        md_lines.append(f"**Prediction:** {pred}")
        md_lines.append(f"**Ground truth:** {gt}")
        md_lines.append(f"**Correct:** {r.get('correct')}")
        md_lines.append("")

    md_path = out_dir / "examples.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote {len(samples)} examples to {md_path}")
    print(f"Images (if extracted) saved under {out_dir}")


if __name__ == "__main__":
    main()
