"""Text metrics and statistics. Pure Python / numpy."""

import random
from collections import Counter
from pathlib import Path

import pytest

from drift.game import ReferentialGame, describe
from drift.metrics import bootstrap_ci, honesty, js_divergence, nonword_share, perplexity
from drift.utils import load_config

CONFIG = str(Path(__file__).resolve().parents[1] / "configs/base.yaml")


def test_honesty():
    g = ReferentialGame().sample(random.Random(2))
    h = honesty(describe(g.target), g)
    assert h["n_attr"] == 4 and h["true_share"] == 1.0 and h["false_rate"] == 0.0
    assert honesty("zzz qq", g)["true_share"] is None
    wrong = [w for w in ["red", "blue"] if w not in describe(g.target)][0]
    assert honesty(wrong, g)["true_share"] == 0.0


def test_js_divergence():
    p = Counter({1: 5, 2: 5})
    assert js_divergence(p, p) == 0
    assert js_divergence(Counter({1: 1}), Counter({2: 1})) == pytest.approx(1.0)
    assert 0 < js_divergence(p, Counter({1: 9, 2: 1})) < 1


def test_perplexity_and_nonwords():
    import torch
    assert perplexity([torch.zeros(3)]) == [1.0]
    assert nonword_share(["red chair xqz"], {"red", "chair"}) == pytest.approx(1 / 3)


def test_bootstrap_ci():
    m, lo, hi = bootstrap_ci([0, 1] * 50)
    assert lo < m == 0.5 < hi
    assert bootstrap_ci([None, 1.0, 1.0])[0] == 1.0  # None (no attribute mentioned) is skipped


def test_config_overrides(tmp_path):
    cfg = load_config(CONFIG, ["beta=0.2", "outside_readers=[a, b]", "amp=null"])
    assert cfg["beta"] == 0.2 and cfg["outside_readers"] == ["a", "b"] and cfg["amp"] is None
    with pytest.raises(KeyError):
        load_config(CONFIG, ["betta=0.2"])  # typos fail loudly
