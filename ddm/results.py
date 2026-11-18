"""Locate and read result files written by scripts/run_inference.py and scripts/score_options.py.

Layout:
    results/<variant>/<model>/<data>/option{N}_sample{S}[_group{G}|_shuffle{K}][_noise{k}].json
    analysis_results/scores/<source>/<model>/<data>/results_option{N}[...].json
where <variant> is single, single_cot, hierarchical or permutation.
"""

import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional

METHOD_VARIANTS = {  # plot label -> (results variant, file suffix)
    "Single-Pass": ("single", ""),
    "CoT": ("single_cot", ""),
    "Hierarchy": ("hierarchical", "_group5"),
    "Permutation": ("permutation", "_shuffle5"),
}


def run_path(root: str, variant: str, model: str, data: str, n: int,
             samples: int = 200, suffix: str = "") -> str:
    return os.path.join(root, variant, model, data, f"option{n}_sample{samples}{suffix}.json")


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def accuracy(path: str) -> Optional[float]:
    d = load_json(path)
    return None if d is None else d["metrics"]["accuracy"]


def mean(values) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def logsumexp(values: List[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))


# ---- gold-margin statistics -----------------------------------------------------------

def single_pass_score_stats(path: str) -> Optional[Dict[str, float]]:
    """Mean gold score, strongest-distractor score, gold margin and distractor mass (logsumexp)."""
    d = load_json(path)
    if d is None:
        return None
    items = d if isinstance(d, list) else d.get("results", [])
    mass = [logsumexp([s for i, s in enumerate(x["all_scores"]) if i != x["gold"] - 1]) for x in items]
    return {
        "gold": mean(x["gold_score"] for x in items),
        "top_distractor": mean(x["best_wrong_score"] for x in items),
        "margin": mean(x["gold_margin"] for x in items),
        "distractor_mass": mean(mass),
    }


def final_round_margin(path: str, require_gold: bool = True) -> Optional[float]:
    """Mean gold margin in the final round of hierarchical / permutation runs."""
    d = load_json(path)
    if d is None:
        return None
    rows = [r for r in d["results"] if r.get("gold_in_final_round", 1) == 1 or not require_gold]
    return mean(r["final_round"]["gold_margin"] for r in rows)


# ---- position statistics ---------------------------------------------------------------

def position_summary(path: str) -> Optional[Dict]:
    """Where gold and predicted answers sit in the candidate list.

    Unparseable predictions (-1) count towards the total but fall in no bin, and are
    excluded from the expected position.
    """
    d = load_json(path)
    if d is None:
        return None
    results = d["results"]
    n = results[0]["num_options"]

    def describe(positions):
        counts = Counter(positions)
        total = sum(counts.values())
        bins = {f"B{b + 1}": 0.0 for b in range(5)}
        for pos, c in counts.items():
            if 0 <= pos < n:
                bins[f"B{min(int(pos / n * 5), 4) + 1}"] += c / total
        valid = {p: c for p, c in counts.items() if p >= 0}
        mean_pos = sum(p * c for p, c in valid.items()) / sum(valid.values()) if valid else None
        norm = None if mean_pos is None else (0.0 if n <= 1 else mean_pos / (n - 1))
        return bins, norm

    gold_bins, gold_norm = describe([r["gold_position"] for r in results])
    pred_bins, pred_norm = describe([r["predicted_position"] for r in results])
    return {
        "num_samples": len(results),
        "num_options": n,
        "position_bin_distribution": {"gold": gold_bins, "predicted": pred_bins},
        "normalized_expected_position": {"gold": gold_norm, "predicted": pred_norm, "random_baseline": 0.5},
    }
