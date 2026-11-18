"""Single-pass selection: show all N candidates in one prompt (the paper's baseline)."""

import asyncio
from typing import Any, Dict, Optional

from ..datasets import CandidateDataset
from ..llm import LLMClient
from ..parsing import parse_answer
from .common import RunTimer, finalize


async def run_single(dataset: CandidateDataset, client: LLMClient, prefix: str = "",
                     config: Optional[Dict[str, Any]] = None, output: Optional[str] = None):
    """`prefix` is prepended to every prompt (used for the irrelevant-context experiment)."""
    timer = RunTimer()
    samples = [dataset[i] for i in range(len(dataset))]
    outputs = await asyncio.gather(*(client.complete(prefix + s["prompt"]) for s in samples))

    results = []
    for s, (text, request) in zip(samples, outputs):
        pred = parse_answer(text, len(s["options"]))
        results.append({
            "sample_id": s["sample_id"],
            "gold_label": s["label"],
            "gold_position": s["gold_position"],
            "predicted_position": pred,  # -1 = unparseable, counted as wrong
            "predicted_label": s["options"][pred] if pred >= 0 else None,
            "is_correct": pred == s["gold_position"],
            "generated_text": text,
            "num_options": len(s["options"]),
            "cost": request,
        })

    invalid = sum(r["predicted_position"] == -1 for r in results)
    return finalize({**(config or {}), "method": "single"}, results, client, timer,
                    {"invalid_predictions": invalid}, output)
