from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import yaml


def load_config(path: str, overrides: list[str] | None = None) -> dict:
    """YAML config plus `key=value` overrides (values parsed as YAML: 0.05, true, null...)."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for o in overrides or []:
        k, v = o.split("=", 1)
        if k not in cfg:
            raise KeyError(f"unknown config key: {k}")
        cfg[k] = yaml.safe_load(v)
    return cfg


def seed_all(seed: int):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    """Appends one JSON line per log call to <run_dir>/metrics.jsonl (and wandb if enabled)."""

    def __init__(self, run_dir: str, cfg: dict):
        os.makedirs(run_dir, exist_ok=True)
        self.run_dir = run_dir
        with open(os.path.join(run_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f)
        self.f = open(os.path.join(run_dir, "metrics.jsonl"), "a")
        self.wandb = None
        if cfg.get("wandb_project"):
            import wandb
            self.wandb = wandb.init(project=cfg["wandb_project"], name=os.path.basename(run_dir), config=cfg)
        self.t0 = time.time()

    def log(self, step: int, d: dict):
        d = {"step": step, "time_min": (time.time() - self.t0) / 60, **d}
        self.f.write(json.dumps(d) + "\n")
        self.f.flush()
        if self.wandb:
            self.wandb.log(d, step=step)

    def text(self, name: str, step: int, obj):
        with open(os.path.join(self.run_dir, f"{name}.jsonl"), "a") as f:
            f.write(json.dumps({"step": step, "data": obj}) + "\n")
