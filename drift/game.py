"""Text referential game: objects, train/eval split, game sampling and prompts.

An object is a tuple of 4 attribute indices (colour, shape, material, size).
One game = a target plus D distractors; the speaker sees only the target, the
listener sees the message and the shuffled candidates.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

ATTRIBUTES: dict[str, list[str]] = {
    "size": ["tiny", "small", "large", "huge"],
    "colour": ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"],
    "material": ["wooden", "metal", "plastic", "glass", "stone", "paper"],
    "shape": ["chair", "table", "lamp", "cup", "box", "ball", "bottle", "bowl"],
}
ATTR_NAMES = list(ATTRIBUTES)  # order used inside object tuples
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

Obj = tuple[int, int, int, int]


def describe(obj: Obj) -> str:
    """Natural English description, e.g. 'large red wooden chair'."""
    return " ".join(ATTRIBUTES[name][v] for name, v in zip(ATTR_NAMES, obj))


def all_objects() -> list[Obj]:
    return list(itertools.product(*(range(len(v)) for v in ATTRIBUTES.values())))


@dataclass
class Game:
    target: Obj
    candidates: list[Obj]  # shuffled, includes the target
    target_idx: int  # position of target in candidates

    @property
    def target_letter(self) -> str:
        return LETTERS[self.target_idx]


class ReferentialGame:
    """Samples games. `heldout_frac` of objects appear only as eval targets."""

    def __init__(self, n_distractors: int = 4, similarity: int = 1,
                 heldout_frac: float = 0.2, split_seed: int = 0):
        assert 0 <= similarity <= 3, "distractors can share 0-3 attributes with the target"
        assert n_distractors + 1 <= len(LETTERS)
        self.n_distractors = n_distractors
        self.similarity = similarity
        objs = all_objects()
        rng = random.Random(split_seed)  # split is fixed across runs/seeds
        rng.shuffle(objs)
        n_held = int(len(objs) * heldout_frac)
        self.eval_objects = objs[:n_held]
        self.train_objects = objs[n_held:]
        self.all = all_objects()

    def _distractor(self, target: Obj, rng: random.Random) -> Obj:
        """Object sharing exactly `similarity` attributes with the target."""
        keep = set(rng.sample(range(4), self.similarity))
        out = []
        for i, v in enumerate(target):
            if i in keep:
                out.append(v)
            else:
                n = len(ATTRIBUTES[ATTR_NAMES[i]])
                out.append(rng.choice([x for x in range(n) if x != v]))
        return tuple(out)

    def sample(self, rng: random.Random, split: str = "train") -> Game:
        pool = self.train_objects if split == "train" else self.eval_objects
        target = rng.choice(pool)
        cands = {target}
        while len(cands) < self.n_distractors + 1:
            cands.add(self._distractor(target, rng))
        cands = list(cands)
        rng.shuffle(cands)  # position carries no information
        return Game(target, cands, cands.index(target))

    def eval_set(self, n: int = 500, seed: int = 12345) -> list[Game]:
        """Fixed evaluation games (same for every run with the same game config)."""
        rng = random.Random(seed)
        return [self.sample(rng, "eval") for _ in range(n)]


# ---------------------------------------------------------------- prompts

def speaker_prompt(game: Game, max_tokens: int) -> str:
    return (
        "You are helping a partner find an object.\n"
        f"Object: a {describe(game.target)}.\n"
        f"Write a message of at most {max_tokens} tokens so your partner can pick it "
        "out from similar objects. Reply with the message only."
    )


def listener_prompt(game: Game, message: str) -> str:
    lines = [f'Your partner says: "{message.strip()}"', "Which object do they mean?"]
    for i, c in enumerate(game.candidates):
        lines.append(f"({LETTERS[i]}) {describe(c)}")
    lines.append("Answer with one letter.")
    return "\n".join(lines)
