"""Evaluate a finished run's checkpoint: heavier outside readers, and cross-seed pairing.

Training only uses the base model as the outside reader (cheap). After training, run
this once per run with larger or other-family readers:

    python scripts/eval_checkpoint.py runs/e1_default_s0 \
        --readers Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-1B-Instruct

Cross-seed legibility: seed 0's speaker with seed 1's listener. If accuracy drops versus
the matched pair, each run invented its own private code:

    python scripts/eval_checkpoint.py runs/e1_default_s0 --listener-run runs/e1_default_s1

Writes runs/<run>/eval_<ckpt>[_x_<other run>].json
"""

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drift.agents import Team, load_reader  # noqa: E402
from drift.evaluate import evaluate, load_wordlist  # noqa: E402
from drift.game import ReferentialGame  # noqa: E402
from drift.utils import seed_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run directory, e.g. runs/e1_default_s0")
    ap.add_argument("--ckpt", default="final", help="checkpoint subdir: final | ckpt_1000 | ...")
    ap.add_argument("--listener-run", default=None, help="take the listener from this run (cross-seed)")
    ap.add_argument("--readers", nargs="*", default=[], help="extra frozen outside readers (HF names)")
    ap.add_argument("--n-games", type=int, default=None, help="default: the run's n_eval_games")
    args = ap.parse_args()

    with open(os.path.join(args.run, "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    if args.n_games:
        cfg["n_eval_games"] = args.n_games
    seed_all(cfg["seed"])
    listener_ckpt = os.path.join(args.listener_run, args.ckpt) if args.listener_run else None
    team = Team(cfg, ckpt=os.path.join(args.run, args.ckpt), listener_ckpt=listener_ckpt)
    readers = {os.path.basename(n): load_reader(n, cfg) for n in args.readers}

    # same fixed held-out games as during training (they depend only on the game config)
    env = ReferentialGame(cfg["n_distractors"], cfg["similarity"], cfg["heldout_frac"])
    res, _, samples = evaluate(team, env.eval_set(cfg["n_eval_games"]), cfg, readers=readers,
                               wordlist=load_wordlist(cfg["wordlist"]))
    name = f"eval_{args.ckpt}" + (f"_x_{os.path.basename(args.listener_run.rstrip('/'))}"
                                  if args.listener_run else "")
    out = os.path.join(args.run, name + ".json")
    with open(out, "w") as f:
        json.dump({"run": args.run, "ckpt": args.ckpt, "listener_run": args.listener_run,
                   "readers": args.readers, **res, "samples": samples}, f, indent=1)
    for k in sorted(k for k in res if not k.endswith(("_lo", "_hi"))):
        print(f"{k:28s} {res[k]:.4f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
