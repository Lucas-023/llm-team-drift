"""Curves over training, mean over seeds, for runs matching a prefix.

    python scripts/plot_runs.py e1_default e2_beta0.0 --metrics eval/acc eval/gap eval/ppl
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def load(prefix, runs_dir="runs"):
    frames = []
    for d in sorted(glob.glob(os.path.join(runs_dir, prefix + "_s*"))):
        rows = [json.loads(line) for line in open(os.path.join(d, "metrics.jsonl"))]
        df = pd.DataFrame(rows)
        df["seed"] = d.rsplit("_s", 1)[-1]
        frames.append(df)
    return pd.concat(frames) if frames else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="+")
    ap.add_argument("--metrics", nargs="+", default=["eval/acc", "eval/acc_out_base", "eval/gap",
                                                    "eval/ppl", "eval/honesty", "eval/vocab_js"])
    ap.add_argument("--out", default="runs/plot.png")
    args = ap.parse_args()
    fig, axes = plt.subplots(1, len(args.metrics), figsize=(4 * len(args.metrics), 3.2))
    for prefix in args.prefixes:
        df = load(prefix)
        if df is None:
            print("no runs for", prefix)
            continue
        for ax, m in zip(axes, args.metrics):
            if m not in df:
                continue
            g = df.dropna(subset=[m]).groupby("step")[m]
            mean, sd = g.mean(), g.std().fillna(0)
            ax.plot(mean.index, mean, label=prefix)
            ax.fill_between(mean.index, mean - sd, mean + sd, alpha=0.2)
            ax.set_title(m)
            ax.set_xlabel("step")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("saved", args.out)


if __name__ == "__main__":
    main()
