"""Deterministic, hash-based option ordering.

Both helpers depend only on the sample id, so the gold position of a sample is the
same across runs, processes and candidate-set sizes.
"""

import hashlib
import random
from typing import List, Sequence, TypeVar

T = TypeVar("T")


def _hash_int(key: str, salt: str) -> int:
    return int(hashlib.sha256(f"{key}|{salt}".encode()).hexdigest(), 16)


def stable_shuffle(items: Sequence[T], key: str, salt: str = "shuf") -> List[T]:
    """Shuffle `items` deterministically, seeded by `key` and `salt`."""
    items = list(items)
    random.Random(_hash_int(key, salt)).shuffle(items)
    return items


def position_balance(sample_id: str, n: int, salt: str = "pos") -> int:
    """Deterministic gold position in [0, n-1], roughly uniform over samples."""
    return _hash_int(sample_id, salt) % max(1, n)
