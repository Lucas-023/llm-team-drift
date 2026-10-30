"""One GRPO training step for the speaker-listener team.

Speaker: GRPO over G sampled messages per game, group-normalized advantage,
         per-token KL to the base model (k3 estimator).
Listener: policy gradient over its letter choice with a batch-mean baseline,
          exact KL to the base model over the D+1 letters.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .agents import LISTENER, SPEAKER, Team, listen_logits, sequence_logprobs, speak


class Optim:
    """Optimizer + fp16 GradScaler (a no-op when amp is off) + grad clipping."""

    def __init__(self, opt, team: Team, max_norm: float):
        self.opt, self.max_norm = opt, max_norm
        fp16 = team.amp_dtype == torch.float16 and team.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=fp16)

    def zero_grad(self):
        self.opt.zero_grad(set_to_none=True)

    def backward(self, loss):
        self.scaler.scale(loss).backward()

    def step(self):
        self.scaler.unscale_(self.opt)
        params = [p for g in self.opt.param_groups for p in g["params"] if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(params, self.max_norm)
        self.scaler.step(self.opt)  # skipped automatically if fp16 grads overflowed
        self.scaler.update()


def group_advantage(r: torch.Tensor, G: int, eps: float = 1e-4) -> torch.Tensor:
    r = r.view(-1, G)
    a = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + eps)
    return a.view(-1)


def train_step(team: Team, game_env, opt_s: Optim, opt_l: Optim, cfg, rng: random.Random, step: int):
    B, G, L = cfg["batch_games"], cfg["group_size"], cfg["max_msg_tokens"]
    games = [game_env.sample(rng, "train") for _ in range(B)]
    games_x = [g for g in games for _ in range(G)]  # game-major, matches speak() output
    targets = torch.tensor([g.target_idx for g in games_x], device=team.device)

    # E4: pair the speaker with one listener from the pool this batch
    listener = rng.choice(team.listeners)

    # ---- rollout
    team.train(False)
    P, M, texts = speak(team, games, L, n_samples=G, temperature=cfg["temperature"])
    team.train(True)

    # ---- listener acts (no grad; its update recomputes per micro-batch) and shared reward
    with torch.no_grad():
        probs = listen_logits(team, games_x, texts, role=listener, micro_bs=cfg["micro_bs"]).softmax(-1)
    choice = torch.multinomial(probs, 1)[:, 0]
    r01 = (choice == targets).float()
    p_target = probs.gather(1, targets[:, None])[:, 0]

    r_speaker = p_target if cfg["smooth_speaker_reward"] else r01
    stats = {}
    if cfg["readability_bonus"] > 0:  # E5: outsider (base model) decodes the message?
        with torch.no_grad():
            out_logits = listen_logits(team, games_x, texts, role=None, micro_bs=cfg["micro_bs"])
        acc_out = (out_logits.argmax(-1) == targets).float()
        r_speaker = r_speaker + cfg["readability_bonus"] * acc_out
        stats["train/acc_outsider"] = acc_out.mean().item()

    train_speaker = cfg["update"] == "joint" or step % 2 == 0
    train_listener = cfg["update"] == "joint" or step % 2 == 1
    mb = cfg["micro_bs"]

    # ---- listener update
    if train_listener:
        adv_l = r01 - r01.mean()
        N = len(games_x)
        opt_l.zero_grad()
        kl_sum = 0.0
        for s in range(0, N, mb):
            sl = slice(s, s + mb)
            with torch.no_grad():
                ref_logp = F.log_softmax(listen_logits(team, games_x[sl], texts[sl], role=None,
                                                       micro_bs=mb), -1)
            logp = F.log_softmax(listen_logits(team, games_x[sl], texts[sl], role=listener,
                                               micro_bs=mb), -1)
            pg = -(adv_l[sl] * logp.gather(1, choice[sl, None])[:, 0]).sum()
            kl = (logp.exp() * (logp - ref_logp)).sum(-1).sum()  # exact KL over the letters
            opt_l.backward((pg + cfg["beta_listener"] * kl) / N)
            kl_sum += kl.item()
        opt_l.step()
        stats["train/listener_kl"] = kl_sum / N

    # ---- speaker update (micro-batched backward; see sequence_logprobs)
    if train_speaker:
        adv = group_advantage(r_speaker, G)
        opt_s.zero_grad()
        N = len(P)
        kl_sum, n_tok = 0.0, 0
        for s in range(0, N, mb):
            with torch.no_grad():
                ref = sequence_logprobs(team, P[s:s + mb], M[s:s + mb], role=None, micro_bs=mb)
            cur = sequence_logprobs(team, P[s:s + mb], M[s:s + mb], role=SPEAKER, micro_bs=mb)
            loss = 0.0
            for i, (lp, lr) in enumerate(zip(cur, ref)):
                d = lr - lp
                kl = (d.exp() - d - 1)  # k3 estimator of KL(pi_S || pi_ref), per token
                # GRPO token-mean form: -(A * log pi) averaged over the message tokens
                loss = loss + (-(adv[s + i] * lp) + cfg["beta"] * kl).mean()
                kl_sum += kl.sum().item()
                n_tok += len(lp)
            opt_s.backward(loss / N)
        opt_s.step()
        stats["train/speaker_kl_per_token"] = kl_sum / max(n_tok, 1)

    stats.update({
        "train/reward": r01.mean().item(),
        "train/p_target": p_target.mean().item(),
        "train/msg_len": sum(len(m) for m in M) / len(M),
        "train/reward_std_in_group": r_speaker.view(-1, G).std(1).mean().item(),
    })
    if opt_s.scaler.is_enabled():
        stats["train/grad_scale"] = opt_s.scaler.get_scale()
    return stats, texts[:G]  # also return one game's group for qualitative logging
