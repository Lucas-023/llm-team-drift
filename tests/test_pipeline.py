"""The real training/eval code on a tiny random-weight Qwen2 (CPU ok).

These catch the bugs that silently ruin an RL run: wrong log-prob positions, an agent
that never updates, a reference model that moves, checkpoints that don't load back.
"""

import random
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.slow

from drift.agents import LISTENER, SPEAKER, Team, letter_ids, listen_logits, make_optimizer, sequence_logprobs, speak  # noqa: E402
from drift.game import ReferentialGame  # noqa: E402
from drift.grpo import Optim, train_step  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODES = ["lora", "full"]


def snapshot(params):
    return [p.detach().clone() for p in params]


def moved(before, params):
    return sum((a - b).abs().sum().item() for a, b in zip(before, params)) > 0


def optims(team, cfg, lr=1e-3):
    return [Optim(make_optimizer(team.params(r), lr, cfg), team, 1.0) for r in (SPEAKER, LISTENER)]


def test_letters_are_single_tokens(make_cfg):
    team = Team(make_cfg())
    assert len(set(letter_ids(team.tok, 17))) == 17  # up to D=16


def test_message_logprobs_match_naive_computation(make_cfg):
    """sequence_logprobs uses left padding + logits_to_keep; compare with the plain way."""
    team = Team(make_cfg())
    games = [ReferentialGame().sample(random.Random(i)) for i in range(3)]
    P, M, _ = speak(team, games, 8, n_samples=2)
    fast = sequence_logprobs(team, P, M, role=SPEAKER)
    for p, m, f in zip(P, M, fast):
        x = torch.cat([p, m])[None]
        with team.ctx(SPEAKER) as model:
            logp = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), -1)
        naive = logp.gather(-1, x[0, 1:, None])[:, 0][-len(m):]
        assert torch.allclose(naive, f, atol=1e-4)


@pytest.mark.parametrize("mode", MODES)
def test_train_step_updates_both_agents_only(make_cfg, mode):
    cfg = make_cfg(f"finetune={mode}")
    team = Team(cfg)
    s0, l0 = snapshot(team.params(SPEAKER)), snapshot(team.params(LISTENER))
    ref0 = snapshot(team.ref.parameters()) if mode == "full" else None
    opt_s, opt_l = optims(team, cfg)
    stats, _ = train_step(team, ReferentialGame(), opt_s, opt_l, cfg, random.Random(0), 1)
    assert moved(s0, team.params(SPEAKER)), "speaker did not update"
    assert moved(l0, team.params(LISTENER)), "listener did not update"
    if mode == "full":
        assert not moved(ref0, team.ref.parameters()), "reference model changed"
    assert stats["train/speaker_kl_per_token"] == pytest.approx(0, abs=1e-6)  # starts at reference
    stats, _ = train_step(team, ReferentialGame(), opt_s, opt_l, cfg, random.Random(1), 2)
    assert stats["train/speaker_kl_per_token"] > 0
    assert 0 <= stats["train/reward"] <= 1


def test_reference_is_base_model_in_lora_mode(make_cfg):
    """After training, role=None must still give the untrained model's outputs."""
    cfg = make_cfg()
    team = Team(cfg)
    games = ReferentialGame().eval_set(8)
    msgs = ["red chair"] * 8
    with torch.no_grad():
        before = listen_logits(team, games, msgs, role=None)
    opt_s, opt_l = optims(team, cfg)
    train_step(team, ReferentialGame(), opt_s, opt_l, cfg, random.Random(0), 1)
    with torch.no_grad():
        assert torch.allclose(before, listen_logits(team, games, msgs, role=None))
        assert not torch.allclose(before, listen_logits(team, games, msgs, role=LISTENER))


def test_alternate_update_trains_one_agent_per_step(make_cfg):
    cfg = make_cfg("update=alternate")
    team = Team(cfg)
    s0, l0 = snapshot(team.params(SPEAKER)), snapshot(team.params(LISTENER))
    opt_s, opt_l = optims(team, cfg)
    train_step(team, ReferentialGame(), opt_s, opt_l, cfg, random.Random(0), 1)  # odd: listener
    assert moved(l0, team.params(LISTENER)) and not moved(s0, team.params(SPEAKER))


@pytest.mark.parametrize("mode", MODES)
def test_checkpoint_roundtrip_and_cross_pairing(make_cfg, tmp_path, mode):
    cfg = make_cfg(f"finetune={mode}")
    games = ReferentialGame().eval_set(8)
    msgs = ["big red thing"] * 8
    runs = {}
    for seed in (0, 1):
        torch.manual_seed(seed)
        team = Team(cfg)
        opt_s, opt_l = optims(team, cfg)
        train_step(team, ReferentialGame(), opt_s, opt_l, cfg, random.Random(seed), 1)
        team.save(str(tmp_path / f"r{seed}"))
        with torch.no_grad():
            runs[seed] = listen_logits(team, games, msgs, role=LISTENER)
    with torch.no_grad():
        loaded = Team(cfg, ckpt=str(tmp_path / "r0"))
        assert torch.allclose(listen_logits(loaded, games, msgs, role=LISTENER), runs[0], atol=1e-5)
        crossed = Team(cfg, ckpt=str(tmp_path / "r0"), listener_ckpt=str(tmp_path / "r1"))
        assert torch.allclose(listen_logits(crossed, games, msgs, role=LISTENER), runs[1], atol=1e-5)


def test_partner_pool_and_reset(make_cfg):
    cfg = make_cfg("listener_pool=3")
    team = Team(cfg)
    for i in (1, 2):
        team.add_listener(f"{LISTENER}_{i}")
    opt_s, opt_l = optims(team, cfg)
    train_step(team, ReferentialGame(), opt_s, opt_l, cfg, random.Random(0), 1)
    team.reset(LISTENER)
    games, msgs = ReferentialGame().eval_set(4), ["x"] * 4
    with torch.no_grad():  # a reset listener equals the base model again
        assert torch.allclose(listen_logits(team, games, msgs, role=LISTENER),
                              listen_logits(team, games, msgs, role=None), atol=1e-5)


def test_scripts_end_to_end(tiny_model, tmp_path):
    """train.py --smoke -> eval_checkpoint.py (incl. cross pairing) -> summarize.py."""
    def run(*args):
        r = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
        return r.stdout
    common = [f"model={tiny_model}", f"out_dir={tmp_path}"]
    run("scripts/train.py", "--smoke", "--set", *common)
    for seed in (0, 1):  # two seeds of one setting, for the cross-seed pairing and the table
        run("scripts/train.py", "--set", *common, f"seed={seed}", f"run_name=t_s{seed}", "steps=2",
            "batch_games=4", "group_size=4", "micro_bs=8", "n_eval_games=16", "eval_every=2",
            "save_every=0")
    run("scripts/eval_checkpoint.py", str(tmp_path / "t_s0"))
    run("scripts/eval_checkpoint.py", str(tmp_path / "t_s0"), "--listener-run", str(tmp_path / "t_s1"))
    out = run("scripts/summarize.py", "--runs-dir", str(tmp_path), "t")
    assert "| t | 2 |" in out and "cross_seed_drop" in out
    run("scripts/eval_zero_shot.py", "--lengths", "4", "--set", *common, "n_eval_games=8")
