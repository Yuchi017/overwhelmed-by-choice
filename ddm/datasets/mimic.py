from typing import Any, Dict, List, Optional

import torch

from .base import CandidateDataset

NUM_SAMPLES = 500


def _codes(note: Dict[str, Any]) -> str:
    # An option is the string form of the note's first three ICD-9 codes.
    return f"{note['icd9_codes'][:3]}"


class MimicDataset(CandidateDataset):
    """ICD-9 code selection from MIMIC-III discharge notes (credentialed data, not shipped).

    `path/mimic_queries.pt` is a list of {"text", "icd9_codes"} notes. For each of 500
    random notes, the gold option is that note's codes and distractors are the codes of
    N-1 other notes.

    Length-controlled variant (Appendix "Distractor Verbosity Control"): pass
    `option_pool_path`, a directory whose `mimic_queries.pt` supplies the distractor pool.
    Queries still come from `path`; the pool is de-duplicated by code string and
    distractors never repeat the gold codes.
    """

    task_instruction = (
        "You are given a clinical NOTE EVENT text and a list of options.Each option consists a "
        "list a icd_code with long_titles.\n\nYour task is to select the correct option number "
        "that best matches\nthe diagnosis described in the note."
    )

    def __init__(self, path: str, num_options: int, option_pool_path: Optional[str] = None, **kwargs):
        self.option_pool_path = option_pool_path
        super().__init__(path, num_options, **kwargs)

    def load_data(self) -> List[Dict[str, Any]]:
        notes = torch.load(f"{self.path}/mimic_queries.pt", weights_only=False)
        pool = None
        if self.option_pool_path is not None:
            pool, seen = [], set()
            for note in torch.load(f"{self.option_pool_path}/mimic_queries.pt", weights_only=False):
                if _codes(note) not in seen:
                    seen.add(_codes(note))
                    pool.append(note)

        rng = self.rng
        query_order = list(range(len(notes)))
        rng.shuffle(query_order)

        data = []
        for _ in range(NUM_SAMPLES):
            qid = query_order.pop()
            gold = _codes(notes[qid])
            if pool is None:
                candidates = [notes[i] for i in range(len(notes)) if i != qid]
            else:
                candidates = [note for note in pool if _codes(note) != gold]
            rng.shuffle(candidates)
            # pop() takes from the end, as the original implementation did
            options = [gold] + [_codes(candidates.pop()) for _ in range(self.num_options - 1)]
            if pool is not None and len(set(options)) != len(options):
                raise ValueError(f"Duplicate option for note {qid}.")
            rng.shuffle(options)
            data.append({"id": qid, "query": notes[qid]["text"], "options": options, "label": gold})
        return data
