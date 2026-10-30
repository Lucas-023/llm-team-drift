"""The environment: object space, split, distractors, prompts. Pure Python."""

import random
from collections import Counter

from drift.game import ReferentialGame, describe, listener_prompt, speaker_prompt


def test_object_space_and_split():
    env = ReferentialGame()
    assert len(env.all) == 1536
    assert len(env.eval_objects) == 307
    assert not set(env.eval_objects) & set(env.train_objects)


def test_split_does_not_depend_on_game_settings():
    a, b = ReferentialGame(n_distractors=2), ReferentialGame(n_distractors=16, similarity=3)
    assert a.eval_objects == b.eval_objects


def test_distractors_share_exactly_k_attributes():
    for k in range(4):
        env = ReferentialGame(n_distractors=4, similarity=k)
        rng = random.Random(0)
        for _ in range(50):
            g = env.sample(rng)
            assert len(set(g.candidates)) == 5 and g.candidates[g.target_idx] == g.target
            for c in g.candidates:
                if c != g.target:
                    assert sum(a == b for a, b in zip(c, g.target)) == k


def test_largest_setting_is_samplable():
    env = ReferentialGame(n_distractors=16, similarity=3)  # E3 extreme
    g = env.sample(random.Random(0))
    assert len(g.candidates) == 17


def test_eval_targets_are_heldout_and_fixed():
    env = ReferentialGame()
    a, b = env.eval_set(100), env.eval_set(100)
    assert [g.target for g in a] == [g.target for g in b]
    assert [g.candidates for g in a] == [g.candidates for g in b]
    assert all(g.target in set(env.eval_objects) for g in a)


def test_train_targets_never_heldout():
    env = ReferentialGame()
    rng, held = random.Random(3), set(env.eval_objects)
    assert all(env.sample(rng, "train").target not in held for _ in range(2000))


def test_target_position_is_uniform():
    env = ReferentialGame()
    rng = random.Random(1)
    pos = Counter(env.sample(rng).target_idx for _ in range(5000))
    assert all(800 < pos[i] < 1200 for i in range(5))


def test_prompts():
    g = ReferentialGame().sample(random.Random(2))
    assert describe(g.target) in speaker_prompt(g, 16)
    lp = listener_prompt(g, "hi")
    assert describe(g.target) in lp and f"({g.target_letter})" in lp
    # the speaker must not see the candidates or their order
    assert "(A)" not in speaker_prompt(g, 16)
