"""The fp16 autocast + GradScaler path, which only switches on with CUDA (run on the VM)."""

import random

import pytest

torch = pytest.importorskip("torch")
pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]

from drift.agents import LISTENER, SPEAKER, Team, make_optimizer  # noqa: E402
from drift.game import ReferentialGame  # noqa: E402
from drift.grpo import Optim, train_step  # noqa: E402


@pytest.mark.parametrize("mode", ["lora", "full"])
def test_fp16_train_step(make_cfg, mode):
    cfg = make_cfg(f"finetune={mode}", "amp=float16", "dtype=float32")
    team = Team(cfg)
    assert team.device.type == "cuda" and team.amp_dtype == torch.float16
    opt_s, opt_l = [Optim(make_optimizer(team.params(r), 1e-3, cfg), team, 1.0) for r in (SPEAKER, LISTENER)]
    assert opt_s.scaler.is_enabled()
    for step in (1, 2, 3):
        stats, _ = train_step(team, ReferentialGame(), opt_s, opt_l, cfg, random.Random(step), step)
    assert all(torch.isfinite(p).all() for p in team.params(SPEAKER))
    assert stats["train/grad_scale"] > 0
    # weights stay fp32 (master copy); only the forward pass runs in fp16
    assert team.params(SPEAKER)[0].dtype == torch.float32
