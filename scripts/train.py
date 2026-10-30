"""Train a speaker-listener team with GRPO and track drift metrics.

    python scripts/train.py --config configs/base.yaml --set seed=1 run_name=e1_default_s1
    python scripts/train.py --config configs/base.yaml --smoke     # 3 quick steps, checks the pipeline
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drift.agents import LISTENER, SPEAKER, Team, load_reader, make_optimizer  # noqa: E402
from drift.evaluate import evaluate, load_wordlist  # noqa: E402
from drift.game import ReferentialGame  # noqa: E402
from drift.grpo import Optim, train_step  # noqa: E402
from drift.utils import Logger, load_config, seed_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="key=value overrides")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.smoke:
        cfg.update(steps=3, batch_games=4, group_size=4, micro_bs=8, n_eval_games=16, eval_every=2,
                   save_every=0, outside_readers=[], run_name=f"smoke_{cfg['finetune']}",
                   wandb_project=None)
    if cfg["finetune"] == "full" and cfg["lr_full"] is not None:
        cfg["lr_speaker"] = cfg["lr_listener"] = cfg["lr_full"]

    seed_all(cfg["seed"])
    rng = random.Random(cfg["seed"])
    run_dir = os.path.join(cfg["out_dir"], cfg["run_name"])
    log = Logger(run_dir, cfg)

    env = ReferentialGame(cfg["n_distractors"], cfg["similarity"], cfg["heldout_frac"])
    eval_games = env.eval_set(cfg["n_eval_games"])
    team = Team(cfg)
    for i in range(1, cfg["listener_pool"]):
        team.add_listener(f"{LISTENER}_{i}")

    def make_opt_l():
        return Optim(make_optimizer([p for n in team.listeners for p in team.params(n)],
                                    cfg["lr_listener"], cfg), team, cfg["grad_clip"])

    opt_s = Optim(make_optimizer(team.params(SPEAKER), cfg["lr_speaker"], cfg), team, cfg["grad_clip"])
    opt_l = make_opt_l()
    print(f"{cfg['finetune']}: trainable speaker {sum(p.numel() for p in team.params(SPEAKER)):,} params, "
          f"listeners {sum(p.numel() for n in team.listeners for p in team.params(n)):,}")

    readers = {os.path.basename(n): load_reader(n, cfg) for n in cfg["outside_readers"]}
    wordlist = load_wordlist(cfg["wordlist"])

    # step 0 = untrained team; its message token counts are the vocabulary-drift reference
    res, base_counts, samples = evaluate(team, eval_games, cfg, readers=readers, wordlist=wordlist)
    res["eval/vocab_js"] = 0.0
    log.log(0, res)
    log.text("eval_messages", 0, samples)
    print(f"step 0  acc {res['eval/acc']:.3f}  out {res['eval/acc_out_base']:.3f}  ppl {res['eval/ppl']:.1f}")

    for step in range(1, cfg["steps"] + 1):
        if cfg["listener_reset_every"] and step % cfg["listener_reset_every"] == 0:
            for n in team.listeners:
                team.reset(n)
            opt_l = make_opt_l()
        stats, group = train_step(team, env, opt_s, opt_l, cfg, rng, step)
        if step % 10 == 0 or args.smoke:
            log.log(step, stats)
            print(f"step {step}  reward {stats['train/reward']:.3f}  len {stats['train/msg_len']:.1f}  "
                  f"kl {stats.get('train/speaker_kl_per_token', float('nan')):.4f}  e.g. {group[0]!r}")
        if step % cfg["eval_every"] == 0:
            res, _, samples = evaluate(team, eval_games, cfg, listener=LISTENER, readers=readers,
                                       base_counts=base_counts, wordlist=wordlist)
            log.log(step, res)
            log.text("eval_messages", step, samples)
            print(f"[eval {step}] acc {res['eval/acc']:.3f}  out {res['eval/acc_out_base']:.3f}  "
                  f"gap {res['eval/gap']:.3f}  ppl {res['eval/ppl']:.1f}  js {res['eval/vocab_js']:.3f}")
        if cfg["save_every"] and step % cfg["save_every"] == 0:
            team.save(os.path.join(run_dir, f"ckpt_{step}"))
    team.save(os.path.join(run_dir, "final"))


if __name__ == "__main__":
    main()
