import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from ..ordering import position_balance, stable_shuffle
from ..prompts import add_instruction, build_prompt

DEFAULT_SEED = 42


class CandidateDataset(ABC):
    """A set of (query, candidates, gold) examples for N-way candidate selection.

    Subclasses implement `load_data()` and set `task_instruction`.

    Randomness: distractor sampling draws from `rng`. To reproduce the paper exactly,
    one `random.Random(42)` is created per process and passed to every dataset built
    in that process, in order (see `make_rng`). Building several datasets from the same
    rng therefore gives different distractors than building each from a fresh rng.

    Option order inside a prompt does *not* depend on `rng`: distractors are ordered by
    a hash of the sample id and the gold answer is placed at `position_balance(id, N)`.
    """

    task_instruction: str = ""

    def __init__(
        self,
        path: str,
        num_options: int,
        rng: Optional[random.Random] = None,
        cot_instruction: Optional[str] = None,
    ):
        self.path = path
        self.num_options = num_options
        self.rng = rng if rng is not None else make_rng()
        self.cot_instruction = cot_instruction
        self.data: List[Dict[str, Any]] = self.load_data()

    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """Return a list of {"id", "query", "options", "label"} dicts (label is in options)."""

    def build_prompt(self, query: str, options: Sequence[str]) -> str:
        """Plain prompt over an arbitrary option list (used by the multi-call methods)."""
        return build_prompt(self.task_instruction, query, options)

    def truncate(self, max_samples: Optional[int]) -> "CandidateDataset":
        if max_samples:
            self.data = self.data[:max_samples]
        return self

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[idx]
        sample_id = str(sample["id"])
        options, label = sample["options"], sample["label"]

        distractors = [o for i, o in enumerate(options) if i != options.index(label)]
        arranged = stable_shuffle(distractors, sample_id, "shuffle")
        gold_position = position_balance(sample_id, len(options), "pos")
        arranged.insert(gold_position, label)

        prompt = self.build_prompt(sample["query"], arranged)
        if self.cot_instruction:
            prompt = add_instruction(prompt, self.cot_instruction)

        return {
            "sample_id": sample_id,
            "query": sample["query"],
            "options": arranged,
            "label": label,
            "gold_position": gold_position,  # 0-based
            "prompt": prompt,
        }


def make_rng(seed: int = DEFAULT_SEED) -> random.Random:
    return random.Random(seed)
