import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_subject_info(sample_path: str):
    """Extract split (dev/test/validation/new/...) and subject from filename."""
    name = Path(sample_path).name
    stem = name.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 2:
        return "unknown", "unknown"
    split = parts[0]
    if len(parts) >= 3:
        subject = "_".join(parts[1:-1])
    else:
        subject = parts[1]
    return split, subject


def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("results", []), data.get("final_acc")


def summarize(results):
    summary = {}
    total = len(results)
    correct = sum(1 for r in results if r.get("reward") == 1.0 or r.get("correct"))
    mean_reward = sum(r.get("reward", 0.0) for r in results) / total if total else 0.0
    summary["overall"] = {"total": total, "correct": correct, "acc": correct / total if total else 0.0,
                          "mean_reward": mean_reward}

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

    summary["split"] = {
        k: {
            "total": v["total"],
            "correct": v["correct"],
            "acc": v["correct"] / v["total"] if v["total"] else 0.0,
        }
        for k, v in split_stats.items()
    }

    summary["subject"] = {
        k: {
            "total": v["total"],
            "correct": v["correct"],
            "acc": v["correct"] / v["total"] if v["total"] else 0.0,
        }
        for k, v in subject_stats.items()
    }
    return summary


def save_plots(summary, output_dir: Path, top_k: int = 15):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover - plotting is optional
        print(f"[warn] matplotlib not available, skipping plots ({e})")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Split-level accuracy
    splits = list(summary["split"].keys())
    split_accs = [summary["split"][s]["acc"] for s in splits]
    plt.figure(figsize=(6, 4))
    plt.bar(splits, split_accs)
    plt.ylabel("Accuracy")
    plt.title("Accuracy by split")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "acc_by_split.png")
    plt.close()

    # Top subjects by count
    subjects = Counter({k: v["total"] for k, v in summary["subject"].items()})
    top = subjects.most_common(top_k)
    if top:
        labels = [k for k, _ in top]
        counts = [summary["subject"][k]["correct"] / summary["subject"][k]["total"] if summary["subject"][k]["total"] else 0.0 for k, _ in top]
        plt.figure(figsize=(10, 5))
        plt.barh(labels, counts)
        plt.xlabel("Accuracy")
        plt.title(f"Top {top_k} subjects by sample count")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / "acc_by_subject.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize evaluation results.")
    parser.add_argument("--input", type=Path, default=Path("eval_results/eval_videommmu_videommmu_subset_greedy_output.json"),
                        help="Path to eval results JSON.")
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"),
                        help="Where to write plots.")
    parser.add_argument("--top-k", type=int, default=15, help="Top-N subjects to plot.")
    args = parser.parse_args()

    results, final_acc = load_results(args.input)
    summary = summarize(results)

    print("Overall:")
    print(f"  total={summary['overall']['total']}, correct={summary['overall']['correct']}, "
          f"acc={summary['overall']['acc']:.3f}, mean_reward={summary['overall']['mean_reward']:.3f}")

    print("\nBy split:")
    for split, stats in sorted(summary["split"].items()):
        print(f"  {split}: total={stats['total']}, acc={stats['acc']:.3f}")

    print("\nTop subjects by count:")
    by_count = Counter({k: v["total"] for k, v in summary["subject"].items()}).most_common(args.top_k)
    for subj, count in by_count:
        acc = summary["subject"][subj]["acc"]
        print(f"  {subj}: total={count}, acc={acc:.3f}")

    save_plots(summary, args.output_dir, args.top_k)
    if final_acc:
        print(f"\nfinal_acc from file: {final_acc}")


if __name__ == "__main__":
    main()
