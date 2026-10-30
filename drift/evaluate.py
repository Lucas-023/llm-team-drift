"""Evaluation on the fixed held-out games: accuracy, outsider accuracy, private-code gap,
naturalness (PPL), honesty, vocabulary drift."""

from __future__ import annotations

import os

import numpy as np
import torch

from . import metrics as Mt
from .agents import LISTENER, SPEAKER, Team, listen_logits, sequence_logprobs, speak


def load_wordlist(path: str | None) -> set[str] | None:
    if path and os.path.exists(path):
        with open(path) as f:
            return {w.strip().lower() for w in f if w.strip().isalpha()}
    return None


@torch.no_grad()
def evaluate(team: Team, games, cfg, speaker=SPEAKER, listener=LISTENER, readers=None,
             base_counts=None, wordlist=None, bs=64):
    """speaker/listener=None evaluates the untrained base model in that role (E0).
    readers: dict name -> (model, tok) of extra frozen outsiders (e.g. Qwen2.5-3B).
    base_counts: token counts of step-0 messages, for vocabulary drift."""
    team.train(False)
    L = cfg["max_msg_tokens"]
    P, M, texts = [], [], []
    for s in range(0, len(games), bs):
        p, m, t = speak(team, games[s:s + bs], L, greedy=True, role=speaker)
        P += p; M += m; texts += t
    targets = torch.tensor([g.target_idx for g in games], device=team.device)

    def acc_of(logits):
        return (logits.argmax(-1) == targets).float().cpu().numpy()

    res, per_game = {}, {}
    per_game["acc"] = acc_of(listen_logits(team, games, texts, role=listener, micro_bs=bs))
    per_game["acc_out_base"] = acc_of(listen_logits(team, games, texts, role=None, micro_bs=bs))
    for name, reader in (readers or {}).items():
        per_game[f"acc_out_{name}"] = acc_of(listen_logits(team, games, texts, reader=reader,
                                                           micro_bs=bs))
    out_keys = [k for k in per_game if k.startswith("acc_out")]
    per_game["gap"] = per_game["acc"] - np.mean([per_game[k] for k in out_keys], 0)

    ref_lp = sequence_logprobs(team, P, M, role=None, micro_bs=bs)
    per_game["ppl"] = Mt.perplexity([lp.float().cpu() for lp in ref_lp])
    hon = [Mt.honesty(t, g) for t, g in zip(texts, games)]
    per_game["honesty"] = [h["true_share"] for h in hon]
    per_game["false_attr_rate"] = [h["false_rate"] for h in hon]
    per_game["msg_len"] = [len(m) for m in M]

    for k, v in per_game.items():
        mean, lo, hi = Mt.bootstrap_ci(v)
        res[f"eval/{k}"], res[f"eval/{k}_lo"], res[f"eval/{k}_hi"] = mean, lo, hi
    counts = Mt.token_counts([m.tolist() for m in M])
    if base_counts is not None:
        res["eval/vocab_js"] = Mt.js_divergence(counts, base_counts)
    if wordlist is not None:
        res["eval/nonword_share"] = Mt.nonword_share(texts, wordlist)
    samples = [{"target": g.target, "message": t} for g, t in zip(games[:20], texts[:20])]
    return res, counts, samples
