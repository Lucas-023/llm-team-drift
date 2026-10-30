"""Results table: final eval metrics of every run, grouped by setting (run name minus _s<seed>).

    python scripts/summarize.py                  # all runs
    python scripts/summarize.py e2 e3            # only run names starting with e2 / e3

Writes runs/summary.csv (one row per run) and runs/summary.md (mean ± sd over seeds), and
adds eval_final*.json results (outside readers, cross-seed) when they exist.
"""

import argparse
import glob
import json
import os
import re

import pandas as pd

KEY = ["eval/acc", "eval/acc_out_base", "eval/gap", "eval/ppl", "eval/honesty",
       "eval/false_attr_rate", "eval/vocab_js", "eval/msg_len"]


def last_eval(run_dir):
    rows = [json.loads(line) for line in open(os.path.join(run_dir, "metrics.jsonl"))]
    evals = [r for r in rows if "eval/acc" in r]
    return evals[-1] if evals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*", default=[""])
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(os.path.join(args.runs_dir, "*", "metrics.jsonl"))):
        run_dir = os.path.dirname(d)
        name = os.path.basename(run_dir)
        if not any(name.startswith(p) for p in args.prefixes):
            continue
        ev = last_eval(run_dir)
        if ev is None:
            continue
        m = re.match(r"(.*)_s(\d+)$", name)
        row = {"setting": m.group(1) if m else name, "seed": int(m.group(2)) if m else 0,
               "run": name, "step": ev["step"], **{k: ev.get(k) for k in KEY}}
        for f in glob.glob(os.path.join(run_dir, "eval_final*.json")):
            post = json.load(open(f))
            tag = os.path.basename(f)[len("eval_final"):-len(".json")] or ""
            if tag.startswith("_x_"):  # cross-seed pairing: matched minus crossed accuracy
                row["acc_cross"] = post["eval/acc"]
            else:
                for k, v in post.items():
                    if k.startswith("eval/acc_out_") and not k.endswith(("_lo", "_hi")):
                        row[k] = v
        rows.append(row)
    if not rows:
        print("no finished runs found")
        return

    df = pd.DataFrame(rows)
    if "acc_cross" in df:
        df["cross_seed_drop"] = df["eval/acc"] - df["acc_cross"]
    df.to_csv(os.path.join(args.runs_dir, "summary.csv"), index=False)

    metrics = [c for c in df.columns if c.startswith("eval/") or c in ("acc_cross", "cross_seed_drop")]
    g = df.groupby("setting")
    lines = ["| setting | seeds | " + " | ".join(m.replace("eval/", "") for m in metrics) + " |",
             "|" + "---|" * (len(metrics) + 2)]
    for setting, sub in g:
        cells = []
        for m in metrics:
            v = sub[m].dropna()
            cells.append("" if v.empty else f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0:.3f}")
        lines.append(f"| {setting} | {len(sub)} | " + " | ".join(cells) + " |")
    md = "\n".join(lines)
    with open(os.path.join(args.runs_dir, "summary.md"), "w") as f:
        f.write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
