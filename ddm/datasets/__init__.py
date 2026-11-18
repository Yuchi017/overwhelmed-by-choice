from .base import CandidateDataset, make_rng
from .beir import BeirDataset
from .mimic import MimicDataset

DATASETS = {
    "beir": BeirDataset,    # HotpotQA and other BEIR subsets
    "mimic": MimicDataset,
}


def load_dataset(kind: str, path: str, num_options: int, max_samples=None, **kwargs) -> CandidateDataset:
    if kind not in DATASETS:
        raise ValueError(f"Unknown dataset kind {kind!r}; choose from {sorted(DATASETS)}")
    return DATASETS[kind](path, num_options, **kwargs).truncate(max_samples)


__all__ = ["CandidateDataset", "BeirDataset", "MimicDataset", "DATASETS", "load_dataset", "make_rng"]
