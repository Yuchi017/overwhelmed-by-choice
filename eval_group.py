import argparse
import asyncio
import json
import math
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from vllm_client import AsyncVLLMClient
from datasets import *
from ddm_core import ReasoningConfig


DATASET_CLASSES = {
    "Arxiv": ArxivDataset,
    "Bier": BierDataset,
    "MIMIC": MIMIC_Dataset,
}


def extract_answer_number(response_text: str, num_options: int) -> int:
    """Return the selected option as a 0-based local index, or -1 if invalid."""
    if response_text is None:
        return -1

    match = re.search(r"<Answer>\s*(\d+)\s*</Answer>", response_text, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_options:
            return num - 1

    clean_text = response_text.replace("*", "").replace("_", "")

    match = re.match(r"^\s*(\d+)", clean_text)
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_options:
            return num - 1

    match = re.search(r"(?:answer)\s*(?::|is)?\s*(\d+)", clean_text, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_options:
            return num - 1

    match = re.search(r"(\d+)\s*$", clean_text.strip())
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_options:
            return num - 1

    lines = clean_text.strip().split("\n")
    if lines:
        match = re.search(r"^(\d+)", lines[0].strip())
        if match:
            num = int(match.group(1))
            if 1 <= num <= num_options:
                return num - 1

    return -1


def _safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Read nested values from dict-like or attribute-like response objects."""
    current = obj
    for key in keys:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def normalize_response(resp: Any) -> Tuple[str, Dict[str, Optional[int]]]:
    """
    Convert several common OpenAI/vLLM response formats into response text + usage.

    The custom AsyncVLLMClient may already return a plain string. In that case token
    fields remain None, but API-call counts, latency, prompt characters, and output
    characters are still recorded.
    """
    if isinstance(resp, str):
        return resp, {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    text = _safe_get(resp, "choices", default=None)
    if text:
        first_choice = text[0]
        content = _safe_get(first_choice, "message", "content", default=None)
        if content is None:
            content = _safe_get(first_choice, "text", default=None)
        response_text = "" if content is None else str(content)
    else:
        direct_text = _safe_get(resp, "text", default=None)
        response_text = str(resp) if direct_text is None else str(direct_text)

    usage_obj = _safe_get(resp, "usage", default=None)
    prompt_tokens = _safe_get(usage_obj, "prompt_tokens", default=None)
    completion_tokens = _safe_get(usage_obj, "completion_tokens", default=None)
    total_tokens = _safe_get(usage_obj, "total_tokens", default=None)

    # Some clients return usage as input_tokens/output_tokens.
    if prompt_tokens is None:
        prompt_tokens = _safe_get(usage_obj, "input_tokens", default=None)
    if completion_tokens is None:
        completion_tokens = _safe_get(usage_obj, "output_tokens", default=None)
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    def to_optional_int(value: Any) -> Optional[int]:
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    return response_text, {
        "prompt_tokens": to_optional_int(prompt_tokens),
        "completion_tokens": to_optional_int(completion_tokens),
        "total_tokens": to_optional_int(total_tokens),
    }


def percentile(values: List[float], q: float) -> Optional[float]:
    """Linear-interpolated percentile without a NumPy dependency."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_numeric(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def sum_known(values: List[Optional[int]]) -> Optional[int]:
    known = [v for v in values if v is not None]
    return sum(known) if known else None


async def run_inference(
    # vLLM parameters
    vllm_client,
    model,
    test_img_paths,
    max_tokens,
    concurrency,

    # inference parameters
    dataset_class,
    dataset_path,
    num_options_per_prompt,
    enable_cot,
    max_samples,
    group=5,

    # output
    output_path=None,
):
    experiment_wall_start = time.perf_counter()
    experiment_started_at = datetime.now(timezone.utc).isoformat()

    print("=" * 80)
    print(f"Running {dataset_class} Classification Inference")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Options per prompt: {num_options_per_prompt}")
    print(f"Chain-of-Thought: {enable_cot}")
    print(f"Group size: {group}")
    print(f"Concurrency: {concurrency}")
    print("=" * 80)

    dataset_load_start = time.perf_counter()
    reasoning_config = (
        ReasoningConfig(
            enable_cot=enable_cot,
            cot_instruction="Let's analyze this paper step-by-step to determine its primary subject category.",
        )
        if enable_cot
        else None
    )

    if dataset_class not in DATASET_CLASSES:
        raise ValueError(f"Invalid dataset class: {dataset_class}")

    dataset = DATASET_CLASSES[dataset_class](
        dataset_path=dataset_path,
        num_options_per_prompt=num_options_per_prompt,
        reasoning=reasoning_config,
        group=None,
    )

    if max_samples is not None:
        dataset.data = dataset.data[:max_samples]

    dataset_load_seconds = time.perf_counter() - dataset_load_start
    print(f"Dataset loaded: {len(dataset)} samples in {dataset_load_seconds:.3f}s")

    sem = asyncio.Semaphore(concurrency)
    global_request_counter = 0

    async def run_one_prompt(prompt, test_img_path=None):
        nonlocal global_request_counter
        global_request_counter += 1
        request_id = global_request_counter

        queued_at = time.perf_counter()
        async with sem:
            acquired_at = time.perf_counter()
            api_started_at = time.perf_counter()
            resp = await vllm_client.create_chat_completion(
                model=model,
                prompt=prompt,
                test_img_paths=[test_img_path] if test_img_path is not None else None,
                max_tokens=max_tokens,
            )
            api_finished_at = time.perf_counter()

        response_text, usage = normalize_response(resp)
        finished_at = time.perf_counter()

        request_metrics = {
            "request_id": request_id,
            "queue_wait_seconds": acquired_at - queued_at,
            "api_latency_seconds": api_finished_at - api_started_at,
            "end_to_end_request_seconds": finished_at - queued_at,
            "prompt_characters": len(prompt),
            "response_characters": len(response_text),
            **usage,
        }
        return response_text, request_metrics

    async def run_prompt_batch(prompts, test_img_path=None):
        batch_start = time.perf_counter()
        tasks = [
            asyncio.create_task(run_one_prompt(prompt, test_img_path))
            for prompt in prompts
        ]
        outputs = await asyncio.gather(*tasks)
        batch_wall_seconds = time.perf_counter() - batch_start
        return outputs, batch_wall_seconds

    correct = 0
    total = 0
    results = []
    all_request_metrics: List[Dict[str, Any]] = []

    inference_wall_start = time.perf_counter()

    for idx in range(len(dataset)):
        sample_start = time.perf_counter()
        sample = dataset[idx]
        arranged_options = sample["options"]
        gold_position = sample["gold_position"]

        current_candidates = [
            {"text": opt, "global_position": i}
            for i, opt in enumerate(arranged_options)
        ]

        round_logs = []
        round_idx = 0
        sample_request_metrics: List[Dict[str, Any]] = []

        gold_eliminated_round = None
        gold_eliminated_group_idx = None

        final_round_prompt = None
        final_round_option_num = None
        final_round_pred_local_position = None
        final_round_gold_local_position = None
        final_round_pred_global_position = None
        final_round_group_idx = None
        final_round_response_text = None

        while len(current_candidates) > 1:
            round_wall_start = time.perf_counter()
            grouped_candidates = [
                current_candidates[start:min(start + group, len(current_candidates))]
                for start in range(0, len(current_candidates), group)
            ]

            prompts = []
            for group_candidates in grouped_candidates:
                group_options = [c["text"] for c in group_candidates]
                prompts.append(dataset.format_prompt_group(sample["query"], group_options))

            test_img_path = None if test_img_paths is None else test_img_paths[idx]
            response_outputs, batch_wall_seconds = await run_prompt_batch(
                prompts, test_img_path=test_img_path
            )

            next_round_candidates = []
            round_group_logs = []

            for group_idx, (group_candidates, output) in enumerate(
                zip(grouped_candidates, response_outputs)
            ):
                response_text, request_metrics = output
                request_metrics.update({
                    "sample_idx": idx,
                    "sample_id": sample["sample_id"],
                    "round_idx": round_idx,
                    "group_idx": group_idx,
                    "num_group_options": len(group_candidates),
                })
                sample_request_metrics.append(request_metrics)
                all_request_metrics.append(request_metrics)

                num_group_options = len(group_candidates)
                predicted_local_idx = extract_answer_number(response_text, num_group_options)

                invalid_prediction = False
                if predicted_local_idx < 0 or predicted_local_idx >= num_group_options:
                    predicted_local_idx = 0
                    invalid_prediction = True

                winner = group_candidates[predicted_local_idx]
                option_records = []
                losers = []
                group_gold_local_position = None
                gold_present_in_group = False

                for local_idx, cand in enumerate(group_candidates):
                    is_winner = local_idx == predicted_local_idx
                    is_gold = (
                        gold_position is not None
                        and cand["global_position"] == gold_position
                    )
                    if is_gold:
                        group_gold_local_position = local_idx
                        gold_present_in_group = True

                    option_records.append({
                        "local_position": local_idx,
                        "global_position": cand["global_position"],
                        "is_winner": is_winner,
                        "is_gold": is_gold,
                    })
                    if not is_winner:
                        losers.append({
                            "local_position": local_idx,
                            "global_position": cand["global_position"],
                            "is_gold": is_gold,
                        })

                group_is_correct = (
                    gold_present_in_group
                    and predicted_local_idx == group_gold_local_position
                )

                if gold_present_in_group and not group_is_correct and gold_eliminated_round is None:
                    gold_eliminated_round = round_idx
                    gold_eliminated_group_idx = group_idx

                group_log = {
                    "round_idx": round_idx,
                    "group_idx": group_idx,
                    "num_group_options": num_group_options,
                    "invalid_prediction": invalid_prediction,
                    "response_text": response_text,
                    "request_cost": request_metrics,
                    "options": option_records,
                    "winner": {
                        "local_position": predicted_local_idx,
                        "global_position": winner["global_position"],
                    },
                    "losers": losers,
                    "gold_present_in_group": gold_present_in_group,
                    "gold_local_position": group_gold_local_position,
                    "gold_global_position": gold_position,
                    "group_is_correct": group_is_correct,
                }
                round_group_logs.append(group_log)
                next_round_candidates.append({
                    "text": winner["text"],
                    "global_position": winner["global_position"],
                })

            if len(next_round_candidates) == 1 and final_round_option_num is None:
                final_group_candidates = grouped_candidates[0]
                final_group_log = round_group_logs[0]
                final_round_prompt = prompts[0]
                final_round_option_num = len(final_group_candidates)
                final_round_pred_local_position = final_group_log["winner"]["local_position"]
                final_round_pred_global_position = final_group_log["winner"]["global_position"]
                final_round_group_idx = final_group_log["group_idx"]
                final_round_response_text = final_group_log["response_text"]
                final_round_gold_local_position = next(
                    (opt["local_position"] for opt in final_group_log["options"] if opt["is_gold"]),
                    None,
                )

            round_wall_seconds = time.perf_counter() - round_wall_start
            round_logs.append({
                "round_idx": round_idx,
                "num_candidates_before_round": len(current_candidates),
                "num_candidates_after_round": len(next_round_candidates),
                "num_groups": len(grouped_candidates),
                "api_calls": len(grouped_candidates),
                "batch_wall_seconds": batch_wall_seconds,
                "round_wall_seconds": round_wall_seconds,
                "sum_api_latency_seconds": sum(
                    g["request_cost"]["api_latency_seconds"] for g in round_group_logs
                ),
                "groups": round_group_logs,
            })

            current_candidates = next_round_candidates
            round_idx += 1

        final_winner = current_candidates[0]
        final_predicted_position = final_winner["global_position"]
        is_correct = final_predicted_position == gold_position
        correct += int(is_correct)
        total += 1

        win_path = []
        defeated_by_final_winner = []
        for rd in round_logs:
            for g in rd["groups"]:
                if g["winner"]["global_position"] == final_predicted_position:
                    win_path.append({
                        "round_idx": g["round_idx"],
                        "group_idx": g["group_idx"],
                        "winner_local_position": g["winner"]["local_position"],
                        "winner_global_position": g["winner"]["global_position"],
                    })
                    defeated_by_final_winner.extend(g["losers"])

        sample_wall_seconds = time.perf_counter() - sample_start
        sample_prompt_tokens = sum_known([m["prompt_tokens"] for m in sample_request_metrics])
        sample_completion_tokens = sum_known([m["completion_tokens"] for m in sample_request_metrics])
        sample_total_tokens = sum_known([m["total_tokens"] for m in sample_request_metrics])

        sample_cost = {
            "wall_seconds": sample_wall_seconds,
            "api_calls": len(sample_request_metrics),
            "sum_api_latency_seconds": sum(m["api_latency_seconds"] for m in sample_request_metrics),
            "sum_queue_wait_seconds": sum(m["queue_wait_seconds"] for m in sample_request_metrics),
            "prompt_characters": sum(m["prompt_characters"] for m in sample_request_metrics),
            "response_characters": sum(m["response_characters"] for m in sample_request_metrics),
            "prompt_tokens": sample_prompt_tokens,
            "completion_tokens": sample_completion_tokens,
            "total_tokens": sample_total_tokens,
            "tokens_per_wall_second": (
                sample_total_tokens / sample_wall_seconds
                if sample_total_tokens is not None and sample_wall_seconds > 0
                else None
            ),
        }

        results.append({
            "sample_id": sample["sample_id"],
            "gold_position": gold_position,
            "predicted_position": final_predicted_position,
            "is_correct": is_correct,
            "num_options": len(arranged_options),
            "tournament_num_rounds": len(round_logs),
            "cost": sample_cost,
            "final_round_prompt": final_round_prompt,
            "final_round_option_num": final_round_option_num,
            "final_round_pred_local_position": final_round_pred_local_position,
            "final_round_gold_local_position": final_round_gold_local_position,
            "final_round_pred_global_position": final_round_pred_global_position,
            "final_round_group_idx": final_round_group_idx,
            "final_round_response_text": final_round_response_text,
            "gold_eliminated_round": gold_eliminated_round,
            "gold_eliminated_group_idx": gold_eliminated_group_idx,
            "final_winner": {
                "global_position": final_predicted_position,
                "win_path": win_path,
            },
            "defeated_by_final_winner": defeated_by_final_winner,
            "rounds": round_logs,
        })

        print(
            f"Completed sample idx={idx}, {idx + 1}/{len(dataset)}, "
            f"correct={is_correct}, calls={sample_cost['api_calls']}, "
            f"wall={sample_wall_seconds:.3f}s"
        )

    inference_wall_seconds = time.perf_counter() - inference_wall_start
    experiment_wall_seconds = time.perf_counter() - experiment_wall_start
    experiment_finished_at = datetime.now(timezone.utc).isoformat()

    accuracy = correct / total * 100 if total > 0 else 0.0
    invalid_predictions = sum(
        int(g["invalid_prediction"])
        for r in results
        for rd in r["rounds"]
        for g in rd["groups"]
    )

    request_latencies = [m["api_latency_seconds"] for m in all_request_metrics]
    request_e2e = [m["end_to_end_request_seconds"] for m in all_request_metrics]
    queue_waits = [m["queue_wait_seconds"] for m in all_request_metrics]
    sample_walls = [r["cost"]["wall_seconds"] for r in results]
    calls_per_sample = [float(r["cost"]["api_calls"]) for r in results]

    total_prompt_tokens = sum_known([m["prompt_tokens"] for m in all_request_metrics])
    total_completion_tokens = sum_known([m["completion_tokens"] for m in all_request_metrics])
    total_tokens = sum_known([m["total_tokens"] for m in all_request_metrics])
    total_api_calls = len(all_request_metrics)

    cost_metrics = {
        "experiment_started_at_utc": experiment_started_at,
        "experiment_finished_at_utc": experiment_finished_at,
        "dataset_load_seconds": dataset_load_seconds,
        "inference_wall_seconds": inference_wall_seconds,
        "experiment_wall_seconds": experiment_wall_seconds,
        "total_api_calls": total_api_calls,
        "api_calls_per_sample": total_api_calls / total if total else None,
        "samples_per_second": total / inference_wall_seconds if inference_wall_seconds > 0 else None,
        "requests_per_second": total_api_calls / inference_wall_seconds if inference_wall_seconds > 0 else None,
        "sum_api_latency_seconds": sum(request_latencies),
        "parallelism_speedup_ratio": (
            sum(request_latencies) / inference_wall_seconds
            if inference_wall_seconds > 0
            else None
        ),
        "request_api_latency_seconds": summarize_numeric(request_latencies),
        "request_end_to_end_seconds": summarize_numeric(request_e2e),
        "request_queue_wait_seconds": summarize_numeric(queue_waits),
        "sample_wall_seconds": summarize_numeric(sample_walls),
        "calls_per_sample_distribution": summarize_numeric(calls_per_sample),
        "total_prompt_characters": sum(m["prompt_characters"] for m in all_request_metrics),
        "total_response_characters": sum(m["response_characters"] for m in all_request_metrics),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_inference_second": (
            total_tokens / inference_wall_seconds
            if total_tokens is not None and inference_wall_seconds > 0
            else None
        ),
        "token_usage_available": total_tokens is not None,
    }

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Invalid group predictions: {invalid_predictions}")
    print(f"Total API calls: {total_api_calls}")
    print(f"API calls/sample: {cost_metrics['api_calls_per_sample']:.3f}" if total else "API calls/sample: N/A")
    print(f"Inference wall time: {inference_wall_seconds:.3f}s")
    print(f"Mean sample latency: {cost_metrics['sample_wall_seconds']['mean']:.3f}s" if sample_walls else "Mean sample latency: N/A")
    print(f"P95 sample latency: {cost_metrics['sample_wall_seconds']['p95']:.3f}s" if sample_walls else "P95 sample latency: N/A")
    print(f"Requests/second: {cost_metrics['requests_per_second']:.3f}" if inference_wall_seconds > 0 else "Requests/second: N/A")
    if total_tokens is not None:
        print(f"Total tokens: {total_tokens}")
        print(f"Tokens/inference-second: {cost_metrics['tokens_per_inference_second']:.3f}")
    else:
        print("Token usage: unavailable from current AsyncVLLMClient response")
    print("=" * 80)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{dataset_class}_grouped_results_with_cost_{timestamp}.json"
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "model_name": model,
            "dataset_path": dataset_path,
            "num_options_per_prompt": num_options_per_prompt,
            "max_samples": max_samples,
            "max_tokens": max_tokens,
            "enable_cot": enable_cot,
            "group_size": group,
            "concurrency": concurrency,
            "inference_mode": "multi_round_group_tournament",
        },
        "metrics": {
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "invalid_group_predictions": invalid_predictions,
            "cost": cost_metrics,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    return results, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run grouped vLLM inference and record accuracy, latency, calls, and token cost."
    )
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset-class", type=str, default="Bier", choices=DATASET_CLASSES.keys())
    parser.add_argument("--dataset-path", type=str, default="data/hotpotqa")
    parser.add_argument("--num-options", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--enable-cot", action="store_true")
    parser.add_argument("--group", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.group < 2:
        parser.error("--group must be at least 2; group=1 never reduces the candidate set.")
    if args.concurrency < 1:
        parser.error("--concurrency must be at least 1.")

    vllm_client = AsyncVLLMClient(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.vllm_port}/v1",
        timeout=3600,
    )

    asyncio.run(
        run_inference(
            vllm_client=vllm_client,
            model=args.model,
            test_img_paths=None,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            dataset_class=args.dataset_class,
            dataset_path=args.dataset_path,
            num_options_per_prompt=args.num_options,
            enable_cot=args.enable_cot,
            max_samples=args.max_samples,
            group=args.group,
            output_path=args.output,
        )
    )