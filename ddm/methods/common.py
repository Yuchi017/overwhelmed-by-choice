import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..parsing import parse_answer


def distribution(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    pos = (len(ordered) - 1) * 0.95
    lo, frac = int(pos), pos - int(pos)
    p95 = ordered[lo] if frac == 0 else ordered[lo] * (1 - frac) + ordered[lo + 1] * frac
    return {"count": len(values), "mean": statistics.fmean(values), "median": statistics.median(values),
            "p95": p95, "min": ordered[0], "max": ordered[-1]}


def sum_known(values: Sequence[Optional[int]]) -> Optional[int]:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


def summarize_requests(requests: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate cost of a list of per-request metrics (see LLMClient.complete)."""
    return {
        "api_calls": len(requests),
        "sum_api_latency_seconds": sum(r["api_latency_seconds"] for r in requests),
        "sum_queue_wait_seconds": sum(r["queue_wait_seconds"] for r in requests),
        "prompt_characters": sum(r["prompt_characters"] for r in requests),
        "response_characters": sum(r["response_characters"] for r in requests),
        "prompt_tokens": sum_known([r["prompt_tokens"] for r in requests]),
        "completion_tokens": sum_known([r["completion_tokens"] for r in requests]),
        "total_tokens": sum_known([r["total_tokens"] for r in requests]),
    }


def pick(response: str, num_options: int):
    """Parse a choice; an unparseable answer falls back to option 0 (multi-call methods)."""
    idx = parse_answer(response, num_options)
    return (0, True) if idx < 0 else (idx, False)


class RunTimer:
    def __init__(self):
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.perf_counter()

    def report(self) -> Dict[str, Any]:
        return {"started_at_utc": self.started_at,
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "wall_seconds": time.perf_counter() - self._t0}


def finalize(config: Dict[str, Any], results: List[Dict[str, Any]], client, timer: RunTimer,
             extra_metrics: Optional[Dict[str, Any]] = None, output: Optional[str] = None) -> Dict[str, Any]:
    """Compute accuracy + cost, print a summary, and save {config, metrics, results} as JSON."""
    total = len(results)
    correct = sum(int(r["is_correct"]) for r in results)
    timing = timer.report()
    requests = client.request_log
    metrics = {
        "total_samples": total,
        "correct": correct,
        "accuracy": correct / total * 100 if total else 0.0,
        **(extra_metrics or {}),
        "cost": {
            **summarize_requests(requests),
            "api_calls_per_sample": len(requests) / total if total else None,
            "api_latency_seconds": distribution([r["api_latency_seconds"] for r in requests]),
            "sample_wall_seconds": distribution([r["cost"]["wall_seconds"] for r in results
                                                 if "wall_seconds" in r.get("cost", {})]),
            "timing": timing,
        },
    }
    print("=" * 80)
    print(f"{config['method']} | {config['model_name']} | N={config['num_options_per_prompt']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}% ({correct}/{total})")
    for key, value in (extra_metrics or {}).items():
        print(f"{key}: {value}")
    print(f"API calls: {len(requests)} | wall: {timing['wall_seconds']:.1f}s")
    print("=" * 80)

    out = {"config": config, "metrics": metrics, "results": results}
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Saved to {output}")
    return out
