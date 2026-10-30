"""E0: the untrained base model as both speaker and listener, at every message limit L.
Gives the starting accuracy and the 'natural English' reference value of every metric.

    python scripts/eval_zero_shot.py --set n_distractors=8
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drift.agents import Team, load_reader  # noqa: E402
from drift.evaluate import evaluate, load_wordlist  # noqa: E402
from drift.game import ReferentialGame  # noqa: E402
from drift.utils import load_config, seed_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--lengths", type=int, nargs="+", default=[32, 16, 8, 4])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_all(cfg["seed"])
    team = Team({**cfg, "finetune": "lora"})  # adapters are never used: speaker=listener=None
    readers = {os.path.basename(n): load_reader(n, cfg) for n in cfg["outside_readers"]}
    env = ReferentialGame(cfg["n_distractors"], cfg["similarity"], cfg["heldout_frac"])
    games = env.eval_set(cfg["n_eval_games"])
    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], f"e0_D{cfg['n_distractors']}_sim{cfg['similarity']}.jsonl")
    with open(out, "w") as f:
        for L in args.lengths:
            res, _, samples = evaluate(team, games, {**cfg, "max_msg_tokens": L},
                                       speaker=None, listener=None, readers=readers,
                                       wordlist=load_wordlist(cfg["wordlist"]))
            print(f"L={L:2d}  acc {res['eval/acc']:.3f}  ppl {res['eval/ppl']:.1f}  "
                  f"honesty {res['eval/honesty']:.3f}  e.g. {samples[0]['message']!r}")
            f.write(json.dumps({"L": L, **res, "samples": samples}) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
