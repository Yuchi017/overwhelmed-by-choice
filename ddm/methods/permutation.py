"""Permutation-based inference.

Stage 1: query the model on K random permutations of the full candidate set.
Stage 2: de-duplicate the K winners (keeping first occurrence order) and, if more than
one distinct winner remains, ask the model to choose among them.
"""

import random
import time
from typing import Any, Dict, Optional

from ..datasets import CandidateDataset
from ..llm import LLMClient
from .common import RunTimer, finalize, pick, summarize_requests


def permutation_seed(seed_base: int, sample_idx: int, perm_idx: int) -> int:
    return seed_base + sample_idx * 10000 + perm_idx


async def _run_sample(idx: int, sample: Dict[str, Any], dataset: CandidateDataset,
                      client: LLMClient, num_shuffles: int, seed_base: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    gold, options = sample["gold_position"], sample["options"]

    # ---- stage 1 ------------------------------------------------------------
    perms = []
    for k in range(num_shuffles):
        order = list(range(len(options)))
        random.Random(permutation_seed(seed_base, idx, k)).shuffle(order)
        perms.append(order)
    prompts = [dataset.build_prompt(sample["query"], [options[g] for g in order]) for order in perms]
    t1 = time.perf_counter()
    outputs = await client.complete_all(prompts)
    stage1_seconds = time.perf_counter() - t1

    shuffle_logs, winners = [], []
    for k, (order, (text, request)) in enumerate(zip(perms, outputs)):
        local, invalid = pick(text, len(order))
        winners.append((order[local], k))
        shuffle_logs.append({
            "shuffle_idx": k,
            "shuffle_seed": permutation_seed(seed_base, idx, k),
            "num_options": len(order),
            "gold_local_position": order.index(gold),
            "gold_global_position": gold,
            "response_text": text,
            "predicted_local_position": local,
            "predicted_global_position": order[local],
            "invalid_prediction": invalid,
            "is_correct": order[local] == gold,
            "options": [{"local_position": j, "global_position": g, "is_gold": g == gold}
                        for j, g in enumerate(order)],
            "request_metrics": request,
        })

    # ---- stage 2 ------------------------------------------------------------
    unique, seen = [], set()
    for g, k in winners:
        if g not in seen:
            seen.add(g)
            unique.append((g, k))

    final_prompt = final_text = final_request = None
    stage2_seconds = 0.0
    if len(unique) == 1:
        local, invalid = 0, False
    else:
        final_prompt = dataset.build_prompt(sample["query"], [options[g] for g, _ in unique])
        t2 = time.perf_counter()
        final_text, final_request = await client.complete(final_prompt)
        stage2_seconds = time.perf_counter() - t2
        local, invalid = pick(final_text, len(unique))
    pred, from_shuffle = unique[local]
    gold_local = next((j for j, (g, _) in enumerate(unique) if g == gold), None)

    requests = [log["request_metrics"] for log in shuffle_logs] + ([final_request] if final_request else [])
    return {
        "sample_id": sample["sample_id"],
        "gold_position": gold,
        "predicted_position": pred,
        "is_correct": pred == gold,
        "num_options": len(options),
        "shuffle_seed_base": seed_base,
        "num_shuffles": num_shuffles,
        "shuffle_rounds": shuffle_logs,
        "final_round": {
            "skipped": len(unique) == 1,
            "skip_reason": "only_one_unique_candidate" if len(unique) == 1 else None,
            "num_options_before_dedup": len(winners),
            "num_options_after_dedup": len(unique),
            "num_options": len(unique),
            "prompt": final_prompt,
            "gold_local_position": gold_local,
            "gold_global_position": gold,
            "response_text": final_text,
            "predicted_local_position": local,
            "predicted_global_position": pred,
            "invalid_prediction": invalid,
            "winner_from_shuffle_idx": from_shuffle,
            "options": [{"local_position": j, "global_position": g, "from_shuffle_idx": k, "is_gold": g == gold}
                        for j, (g, k) in enumerate(unique)],
            "request_metrics": final_request,
            "wall_seconds": stage2_seconds,
        },
        "cost": {**summarize_requests(requests), "wall_seconds": time.perf_counter() - t0,
                 "stage1_batch_wall_seconds": stage1_seconds, "stage2_wall_seconds": stage2_seconds},
    }


async def run_permutation(dataset: CandidateDataset, client: LLMClient, num_shuffles: int = 5,
                          seed_base: int = 42, config: Optional[Dict[str, Any]] = None,
                          output: Optional[str] = None):
    timer = RunTimer()
    results = []
    for i in range(len(dataset)):
        r = await _run_sample(i, dataset[i], dataset, client, num_shuffles, seed_base)
        results.append(r)
        print(f"[{i + 1}/{len(dataset)}] correct={r['is_correct']} "
              f"final_candidates={r['final_round']['num_options']}")

    fr = [r["final_round"] for r in results]
    extra = {
        "invalid_shuffle_predictions": sum(s["invalid_prediction"] for r in results for s in r["shuffle_rounds"]),
        "invalid_final_predictions": sum(f["invalid_prediction"] for f in fr),
        "final_round_skipped": sum(f["skipped"] for f in fr),
        "final_round_gold_present_count": sum(f["gold_local_position"] is not None for f in fr),
    }
    return finalize({**(config or {}), "method": "permutation", "num_shuffles": num_shuffles,
                     "shuffle_seed_base": seed_base}, results, client, timer, extra, output)
