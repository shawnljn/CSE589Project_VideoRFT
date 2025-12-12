"""
Compare multiple eval result JSONs and write a summary report.

Usage:
  python src/compare_results.py --inputs eval_results/eval_videommmu_videommmu_subset_greedy_output.json eval_results/eval_videommmu_videommmu_subset_t01_p01_output.json --out report.txt

It will compute overall acc/mean_reward, split acc, and top subjects for each file.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_subject_info(sample_path: str):
    name = Path(sample_path).name
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return "unknown", "unknown"
    split = parts[0]
    subject = "_".join(parts[1:-1]) if len(parts) >= 3 else parts[1]
    return split, subject


def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", []), data.get("final_acc")


def summarize(results):
    total = len(results)
    correct = sum(1 for r in results if r.get("reward") == 1.0 or r.get("correct"))
    mean_reward = sum(r.get("reward", 0.0) for r in results) / total if total else 0.0

    split_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    subject_stats = defaultdict(lambda: {"total": 0, "correct": 0})
    for r in results:
        split, subject = parse_subject_info(r.get("path", "unknown"))
        split_stats[split]["total"] += 1
        subject_stats[subject]["total"] += 1
        is_correct = r.get("reward") == 1.0 or r.get("correct")
        if is_correct:
            split_stats[split]["correct"] += 1
            subject_stats[subject]["correct"] += 1

    split_out = {
        k: {
            "total": v["total"],
            "acc": v["correct"] / v["total"] if v["total"] else 0.0,
        }
        for k, v in split_stats.items()
    }
    subject_out = {
        k: {
            "total": v["total"],
            "acc": v["correct"] / v["total"] if v["total"] else 0.0,
        }
        for k, v in subject_stats.items()
    }
    return {
        "total": total,
        "correct": correct,
        "acc": correct / total if total else 0.0,
        "mean_reward": mean_reward,
        "split": split_out,
        "subject": subject_out,
    }


def format_summary(name, summary, final_acc, top_k=10):
    lines = []
    lines.append(f"=== {name} ===")
    lines.append(f"overall: total={summary['total']}, acc={summary['acc']:.3f}, mean_reward={summary['mean_reward']:.3f}")
    if final_acc:
        lines.append(f"final_acc_field: {final_acc}")
    lines.append("by split:")
    for split, stats in sorted(summary["split"].items()):
        lines.append(f"  {split}: total={stats['total']}, acc={stats['acc']:.3f}")
    lines.append(f"top {top_k} subjects by count:")
    by_count = Counter({k: v["total"] for k, v in summary["subject"].items()}).most_common(top_k)
    for subj, count in by_count:
        lines.append(f"  {subj}: total={count}, acc={summary['subject'][subj]['acc']:.3f}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare eval result JSONs.")
    parser.add_argument("--inputs", nargs="+", required=True, help="List of eval result JSON files.")
    parser.add_argument("--out", type=Path, default=Path("eval_results/comparison_report.txt"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    reports = []
    for inp in args.inputs:
        path = Path(inp)
        results, final_acc = load_results(path)
        summary = summarize(results)
        reports.append(format_summary(path.name, summary, final_acc, args.top_k))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(reports), encoding="utf-8")
    print(f"Written report to {args.out}")


if __name__ == "__main__":
    main()
