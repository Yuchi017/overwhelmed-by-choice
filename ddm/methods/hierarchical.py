"""Hierarchical partitioning: a knockout tournament over groups of `group_size` candidates.

Each round splits the surviving candidates (in their current order) into consecutive
groups, asks the model to pick one winner per group, and keeps the winners. This
repeats until one candidate remains. Samples are processed one at a time; the groups
of a round are queried concurrently.
"""

import time
from typing import Any, Dict, List, Optional

from ..datasets import CandidateDataset
from ..llm import LLMClient
from .common import RunTimer, finalize, pick, summarize_requests


async def _run_sample(sample: Dict[str, Any], dataset: CandidateDataset, client: LLMClient,
                      group_size: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    gold = sample["gold_position"]
    alive: List[int] = list(range(len(sample["options"])))  # global positions still in play
    rounds, requests = [], []
    final = {}
    gold_eliminated = (None, None)  # (round, group)

    while len(alive) > 1:
        groups = [alive[i:i + group_size] for i in range(0, len(alive), group_size)]
        prompts = [dataset.build_prompt(sample["query"], [sample["options"][g] for g in grp])
                   for grp in groups]
        t_round = time.perf_counter()
        outputs = await client.complete_all(prompts)

        group_logs, winners = [], []
        for gi, (grp, prompt, (text, request)) in enumerate(zip(groups, prompts, outputs)):
            local, invalid = pick(text, len(grp))
            winner = grp[local]
            gold_local = grp.index(gold) if gold in grp else None
            if gold_local is not None and local != gold_local and gold_eliminated[0] is None:
                gold_eliminated = (len(rounds), gi)
            request.update({"round_idx": len(rounds), "group_idx": gi, "num_group_options": len(grp)})
            requests.append(request)
            winners.append(winner)
            group_logs.append({
                "round_idx": len(rounds),
                "group_idx": gi,
                "num_group_options": len(grp),
                "invalid_prediction": invalid,
                "response_text": text,
                "options": [{"local_position": j, "global_position": g, "is_winner": j == local,
                             "is_gold": g == gold} for j, g in enumerate(grp)],
                "winner": {"local_position": local, "global_position": winner},
                "losers": [{"local_position": j, "global_position": g, "is_gold": g == gold}
                           for j, g in enumerate(grp) if j != local],
                "gold_present_in_group": gold_local is not None,
                "gold_local_position": gold_local,
                "gold_global_position": gold,
                "group_is_correct": gold_local is not None and local == gold_local,
            })
            if len(groups) == 1:  # the final round: kept for the log-prob analysis
                final = {
                    "final_round_prompt": prompt,
                    "final_round_option_num": len(grp),
                    "final_round_pred_local_position": local,
                    "final_round_gold_local_position": gold_local,
                    "final_round_pred_global_position": winner,
                    "final_round_group_idx": gi,
                    "final_round_response_text": text,
                }

        rounds.append({
            "round_idx": len(rounds),
            "num_candidates_before_round": len(alive),
            "num_candidates_after_round": len(winners),
            "num_groups": len(groups),
            "round_wall_seconds": time.perf_counter() - t_round,
            "groups": group_logs,
        })
        alive = winners

    pred = alive[0]
    win_path = [{"round_idx": g["round_idx"], "group_idx": g["group_idx"],
                 "winner_local_position": g["winner"]["local_position"],
                 "winner_global_position": pred}
                for rd in rounds for g in rd["groups"] if g["winner"]["global_position"] == pred]
    defeated = [l for rd in rounds for g in rd["groups"]
                if g["winner"]["global_position"] == pred for l in g["losers"]]

    return {
        "sample_id": sample["sample_id"],
        "gold_position": gold,
        "predicted_position": pred,
        "is_correct": pred == gold,
        "num_options": len(sample["options"]),
        "tournament_num_rounds": len(rounds),
        "cost": {**summarize_requests(requests), "wall_seconds": time.perf_counter() - t0},
        **{k: final.get(k) for k in ("final_round_prompt", "final_round_option_num",
                                     "final_round_pred_local_position", "final_round_gold_local_position",
                                     "final_round_pred_global_position", "final_round_group_idx",
                                     "final_round_response_text")},
        "gold_eliminated_round": gold_eliminated[0],
        "gold_eliminated_group_idx": gold_eliminated[1],
        "final_winner": {"global_position": pred, "win_path": win_path},
        "defeated_by_final_winner": defeated,
        "rounds": rounds,
    }


async def run_hierarchical(dataset: CandidateDataset, client: LLMClient, group_size: int = 5,
                           config: Optional[Dict[str, Any]] = None, output: Optional[str] = None):
    if group_size < 2:
        raise ValueError("group_size must be >= 2")
    timer = RunTimer()
    results = []
    for i in range(len(dataset)):
        r = await _run_sample(dataset[i], dataset, client, group_size)
        results.append(r)
        print(f"[{i + 1}/{len(dataset)}] correct={r['is_correct']} rounds={r['tournament_num_rounds']}")

    invalid = sum(g["invalid_prediction"] for r in results for rd in r["rounds"] for g in rd["groups"])
    return finalize({**(config or {}), "method": "hierarchical", "group_size": group_size},
                    results, client, timer, {"invalid_group_predictions": invalid}, output)
