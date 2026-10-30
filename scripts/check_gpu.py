"""One-time checks on the GPU VM, before any experiment (about 10-15 minutes).

    python scripts/check_gpu.py                 # both checks, both modes
    python scripts/check_gpu.py --modes lora    # skip full fine-tuning

1. fp16 vs fp32: does fp16 autocast change what the base model says/answers?
   (Qwen was trained in bf16; the V100 only has fp16, which can overflow.)
2. Speed and peak memory of real-size training steps, per mode (lora / full),
   and the projected hours for one full run.

Writes runs/gpu_check.json. Send that file back to decide amp, micro_bs and step budget.
"""

import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drift.agents import LISTENER, SPEAKER, Team, listen_logits, make_optimizer, sequence_logprobs, speak  # noqa: E402
from drift.evaluate import evaluate  # noqa: E402
from drift.game import ReferentialGame  # noqa: E402
from drift.grpo import Optim, train_step  # noqa: E402
from drift.utils import load_config, seed_all  # noqa: E402


def fp16_check(cfg, n_games=128):
    """Generate messages in fp32, then score the same messages in fp32 and in fp16."""
    team = Team({**cfg, "finetune": "lora"})
    games = ReferentialGame(cfg["n_distractors"], cfg["similarity"]).eval_set(n_games)
    targets = torch.tensor([g.target_idx for g in games], device=team.device)
    scored = {}
    with torch.no_grad():
        team.amp_dtype = None
        P, M, texts = speak(team, games, cfg["max_msg_tokens"], greedy=True, role=None)
        for name, amp in [("fp32", None), ("fp16", torch.float16)]:
            team.amp_dtype = amp
            scored[name] = (listen_logits(team, games, texts, role=None),
                            torch.cat(sequence_logprobs(team, P, M, role=None)))
        _, _, texts16 = speak(team, games, cfg["max_msg_tokens"], greedy=True, role=None)
    (l32, lp32), (l16, lp16) = scored["fp32"], scored["fp16"]
    res = {
        "finite_fp16": bool(torch.isfinite(l16).all() and torch.isfinite(lp16).all()),
        "listener_answer_agreement": (l32.argmax(-1) == l16.argmax(-1)).float().mean().item(),
        "acc_fp32": (l32.argmax(-1) == targets).float().mean().item(),
        "acc_fp16": (l16.argmax(-1) == targets).float().mean().item(),
        "max_abs_logprob_diff": (lp32 - lp16).abs().max().item(),
        "greedy_message_agreement": sum(a == b for a, b in zip(texts, texts16)) / n_games,
        "example_message": texts[0],
    }
    res["verdict"] = ("fp16 OK" if res["finite_fp16"] and res["listener_answer_agreement"] > 0.97
                      and abs(res["acc_fp32"] - res["acc_fp16"]) < 0.02 else "use amp=null")
    del team
    torch.cuda.empty_cache()
    return res


def speed_check(cfg, mode, n_steps=3):
    cfg = {**cfg, "finetune": mode}
    if mode == "full":
        cfg["lr_speaker"] = cfg["lr_listener"] = cfg["lr_full"]
    torch.cuda.reset_peak_memory_stats()
    team = Team(cfg)
    opt_s = Optim(make_optimizer(team.params(SPEAKER), cfg["lr_speaker"], cfg), team, cfg["grad_clip"])
    opt_l = Optim(make_optimizer(team.params(LISTENER), cfg["lr_listener"], cfg), team, cfg["grad_clip"])
    env = ReferentialGame(cfg["n_distractors"], cfg["similarity"])
    rng = random.Random(0)
    train_step(team, env, opt_s, opt_l, cfg, rng, 1)  # warm-up (allocations, kernels)
    torch.cuda.synchronize()
    t0 = time.time()
    for s in range(n_steps):
        train_step(team, env, opt_s, opt_l, cfg, rng, s + 2)
    torch.cuda.synchronize()
    sec_step = (time.time() - t0) / n_steps
    t0 = time.time()
    evaluate(team, env.eval_set(cfg["n_eval_games"]), cfg)
    sec_eval = time.time() - t0
    n_evals = cfg["steps"] // cfg["eval_every"] + 1
    res = {"sec_per_step": sec_step, "sec_per_eval": sec_eval,
           "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
           "total_mem_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
           "hours_per_run": (sec_step * cfg["steps"] + sec_eval * n_evals) / 3600,
           "micro_bs": cfg["micro_bs"]}
    del team, opt_s, opt_l
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--modes", nargs="+", default=["lora", "full"])
    args = ap.parse_args()
    assert torch.cuda.is_available(), "run this on the GPU VM"
    cfg = load_config(args.config, args.set)
    seed_all(0)
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "bf16_supported": torch.cuda.is_bf16_supported(), "config_overrides": args.set}
    print(json.dumps(report, indent=1))

    print("\n[1/2] fp16 vs fp32 ...")
    report["fp16_check"] = fp16_check(cfg)
    print(json.dumps(report["fp16_check"], indent=1))

    report["speed"] = {}
    for mode in args.modes:
        print(f"\n[2/2] speed + memory, {mode} ...")
        try:
            report["speed"][mode] = speed_check(cfg, mode)
        except torch.cuda.OutOfMemoryError:
            report["speed"][mode] = {"error": "OOM: try --set micro_bs=16 or optim=adamw8bit"}
            torch.cuda.empty_cache()
        print(json.dumps(report["speed"][mode], indent=1))

    os.makedirs(cfg["out_dir"], exist_ok=True)
    out = os.path.join(cfg["out_dir"], "gpu_check.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
