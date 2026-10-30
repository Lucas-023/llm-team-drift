"""Drift metrics. The text-only ones (honesty, vocabulary drift, non-word share, bootstrap)
need no model and are unit-tested in tests/test_metrics.py."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .game import ATTR_NAMES, ATTRIBUTES, Game

_WORD = re.compile(r"[a-z]+")
_ATTR_WORDS = {w: (i, v) for i, name in enumerate(ATTR_NAMES) for v, w in enumerate(ATTRIBUTES[name])}


def honesty(message: str, game: Game) -> dict:
    """Of the attribute words mentioned, the share true of the target (None if none mentioned)."""
    words = _WORD.findall(message.lower())
    mentioned = [_ATTR_WORDS[w] for w in words if w in _ATTR_WORDS]
    if not mentioned:
        return {"n_attr": 0, "true_share": None, "false_rate": 0.0}
    true = sum(game.target[i] == v for i, v in mentioned)
    return {"n_attr": len(mentioned), "true_share": true / len(mentioned),
            "false_rate": 1 - true / len(mentioned)}


def token_counts(token_lists) -> Counter:
    c = Counter()
    for t in token_lists:
        c.update(int(x) for x in t)
    return c


def js_divergence(p: Counter, q: Counter) -> float:
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two token-frequency counters."""
    keys = set(p) | set(q)
    P = np.array([p[k] for k in keys], float)
    Q = np.array([q[k] for k in keys], float)
    P, Q = P / P.sum(), Q / Q.sum()
    Mx = 0.5 * (P + Q)

    def kl(a, b):
        m = a > 0
        return float((a[m] * np.log2(a[m] / b[m])).sum())

    return 0.5 * kl(P, Mx) + 0.5 * kl(Q, Mx)


def nonword_share(messages: list[str], vocab: set[str]) -> float:
    """Share of alphabetic word pieces that are not in an English word list."""
    words = [w for m in messages for w in _WORD.findall(m.lower())]
    return float(np.mean([w not in vocab for w in words])) if words else 0.0


def perplexity(token_logprobs) -> list[float]:
    """PPL of each message from its per-token log-probs under the frozen base model."""
    return [math.exp(-float(lp.mean())) if len(lp) else float("nan") for lp in token_logprobs]


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean and 95% bootstrap CI over evaluation games."""
    x = np.asarray([v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))], float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x))).mean(1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
