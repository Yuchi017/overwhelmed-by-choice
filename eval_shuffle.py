# Run inference on datasets using HuggingFace models.

import argparse
import re
import json
import random
from datetime import datetime
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support
from vllm_client import AsyncVLLMClient
import asyncio
import os
import math
import statistics
import time
from typing import Any, Dict, List, Optional, Tuple
from datasets import *
from ddm_core import ReasoningConfig
from transformers import AutoTokenizer

DATASET_CLASSES = {
    "Arxiv": ArxivDataset,
    "Bier": BierDataset,
    "MIMIC": MIMIC_Dataset,
}

def extract_answer_number(response_text: str, num_options: int) -> int:
    """
    Extract the answer number from model response.

    Looks for patterns like:
    - "<Answer> 5 </Answer>" (primary format)
    - "<Answer>5</Answer>" (no spaces)
    - "Answer: 5"
    - "5" at beginning

    Returns:
        Answer index (0-based) or -1 if not found
    """
    # First, try to find <Answer> tags (primary expected format)
    match = re.search(r'<Answer>\s*(\d+)\s*</Answer>', response_text, re.IGNORECASE)
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_options:
            return num - 1

    # Remove markdown formatting for fallback patterns
    clean_text = response_text.replace('*', '').replace('_', '')

    # Try to find number at the very beginning
    match = re.match(r'^\s*(\d+)', clean_text)
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_options:
            return num - 1

    # Try to find "Answer: N" pattern
    match = re.search(r'(?:answer|Answer|ANSWER)\s*(?::|is)?\s*(\d+)', clean_text)
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_options:
            return num - 1

    # Try to find just a number at the end
    match = re.search(r'(\d+)\s*$', clean_text.strip())
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_options:
            return num - 1

    # Try to find any number in the first line
    lines = clean_text.strip().split('\n')
    if lines:
        match = re.search(r'^(\d+)', lines[0].strip())
        if match:
            num = int(match.group(1))
            if 1 <= num <= num_options:
                return num - 1

    return -1


def _safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    current = obj
    for key in keys:
        if current is None:
            return default
        current = current.get(key, default) if isinstance(current, dict) else getattr(current, key, default)
    return current


def normalize_response(resp: Any) -> Tuple[str, Dict[str, Optional[int]]]:
    """Normalize plain-string or OpenAI-style responses and extract token usage."""
    if isinstance(resp, str):
        return resp, {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    choices = _safe_get(resp, "choices", default=None)
    if choices:
        first = choices[0]
        content = _safe_get(first, "message", "content", default=None)
        if content is None:
            content = _safe_get(first, "text", default=None)
        text = "" if content is None else str(content)
    else:
        direct = _safe_get(resp, "text", default=None)
        text = str(resp) if direct is None else str(direct)

    usage = _safe_get(resp, "usage", default=None)
    pt = _safe_get(usage, "prompt_tokens", default=None)
    ct = _safe_get(usage, "completion_tokens", default=None)
    tt = _safe_get(usage, "total_tokens", default=None)
    if pt is None:
        pt = _safe_get(usage, "input_tokens", default=None)
    if ct is None:
        ct = _safe_get(usage, "output_tokens", default=None)
    if tt is None and pt is not None and ct is not None:
        tt = int(pt) + int(ct)

    def as_int(v):
        try:
            return None if v is None else int(v)
        except (TypeError, ValueError):
            return None

    return text, {"prompt_tokens": as_int(pt), "completion_tokens": as_int(ct), "total_tokens": as_int(tt)}


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return values[lo]
    w = pos - lo
    return values[lo] * (1 - w) + values[hi] * w


def summarize_numeric(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": len(values), "mean": statistics.fmean(values),
        "median": statistics.median(values), "p95": percentile(values, 0.95),
        "min": min(values), "max": max(values),
    }


def sum_known(values):
    known = [v for v in values if v is not None]
    return sum(known) if known else None


async def run_inference(
    # vllm_parameters
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
    num_shuffles=5,
    shuffle_seed_base=42,

    # output
    output_path=None,
):
    experiment_wall_start = time.perf_counter()
    experiment_started_at = datetime.now().astimezone().isoformat()
    print("=" * 80)
    print(f"Running {dataset_class} Classification Inference")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Options per prompt: {num_options_per_prompt}")
    print(f"Chain-of-Thought: {enable_cot}")
    print(f"Num shuffles: {num_shuffles}")
    print(f"Shuffle seed base: {shuffle_seed_base}")
    print(f"Concurrency: {concurrency}")
    print("=" * 80)

    # Load dataset
    dataset_load_start = time.perf_counter()
    print("\nLoading dataset...")
    reasoning_config = ReasoningConfig(
        enable_cot=enable_cot,
        cot_instruction="Let's analyze this paper step-by-step to determine its primary subject category."
    ) if enable_cot else None

    if dataset_class not in DATASET_CLASSES:
        raise ValueError(f"Invalid dataset class: {dataset_class}")

    # IMPORTANT: shuffle logic is handled inside run_inference, not in dataset.__getitem__()
    dataset = DATASET_CLASSES[dataset_class](
        dataset_path=dataset_path,
        num_options_per_prompt=num_options_per_prompt,
        reasoning=reasoning_config,
        group=None,
    )

 
    # Limit samples if specified
    if max_samples:
        dataset.data = dataset.data[:max_samples]

    dataset_load_seconds = time.perf_counter() - dataset_load_start
    print(f"Dataset loaded: {len(dataset)} samples in {dataset_load_seconds:.3f}s")

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    sem = asyncio.Semaphore(concurrency)
    global_request_counter = 0

    async def run_one_prompt(prompt, test_img_path=None):
        nonlocal global_request_counter
        global_request_counter += 1
        request_id = global_request_counter
        queued_at = time.perf_counter()
        async with sem:
            acquired_at = time.perf_counter()
            api_start = time.perf_counter()
            resp = await vllm_client.create_chat_completion(
                model=model, prompt=prompt,
                test_img_paths=[test_img_path] if test_img_path is not None else None,
                max_tokens=max_tokens,
            )
            api_end = time.perf_counter()
        response_text, usage = normalize_response(resp)
        finished = time.perf_counter()
        metrics = {
            "request_id": request_id,
            "queue_wait_seconds": acquired_at - queued_at,
            "api_latency_seconds": api_end - api_start,
            "end_to_end_request_seconds": finished - queued_at,
            "prompt_characters": len(prompt),
            "response_characters": len(response_text),
            **usage,
        }
        return response_text, metrics

    async def run_prompt_batch(prompts, test_img_path=None):
        batch_start = time.perf_counter()
        tasks = [asyncio.create_task(run_one_prompt(prompt, test_img_path)) for prompt in prompts]
        outputs = await asyncio.gather(*tasks)
        return outputs, time.perf_counter() - batch_start

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    correct = 0
    total = 0
    results = []

    all_request_metrics = []
    for idx in range(len(dataset)):
        sample_wall_start = time.perf_counter()
        sample = dataset[idx]
        arranged_options = sample["options"]
        gold_position = sample["gold_position"]

        # each candidate carries:
        # {
        #   "text": option text,
        #   "global_position": original position in arranged_options
        # }
        original_candidates = [
            {
                "text": opt,
                "global_position": i,
            }
            for i, opt in enumerate(arranged_options)
        ]

        test_img_path = None
        if test_img_paths is not None:
            test_img_path = test_img_paths[idx]

        # ==========================================================
        # Stage 1: shuffle num_shuffles times
        # ==========================================================
        shuffle_inputs = []
        shuffle_prompts = []

        for shuffle_idx in range(num_shuffles):
            seed = shuffle_seed_base + idx * 10000 + shuffle_idx
            rng = random.Random(seed)

            shuffled_candidates = original_candidates.copy()
            rng.shuffle(shuffled_candidates)

            gold_local_position = None
            option_records = []

            for local_idx, cand in enumerate(shuffled_candidates):
                is_gold = (gold_position is not None and cand["global_position"] == gold_position)
                if is_gold:
                    gold_local_position = local_idx

                option_records.append({
                    "local_position": local_idx,
                    "global_position": cand["global_position"],
                    "is_gold": is_gold,
                })

            prompt = dataset.format_prompt_group(
                sample["query"],
                [c["text"] for c in shuffled_candidates]
            )

            shuffle_inputs.append({
                "shuffle_idx": shuffle_idx,
                "shuffle_seed": seed,
                "candidates": shuffled_candidates,
                "gold_local_position": gold_local_position,
                "option_records": option_records,
                "prompt": prompt,
            })
            shuffle_prompts.append(prompt)

        shuffle_outputs, stage1_batch_wall_seconds = await run_prompt_batch(
            shuffle_prompts, test_img_path=test_img_path
        )
        shuffle_responses = [x[0] for x in shuffle_outputs]
        shuffle_request_metrics = [x[1] for x in shuffle_outputs]
        all_request_metrics.extend(shuffle_request_metrics)

        shuffle_logs = []
        all_stage1_winners = []

        for shuffle_info, response_text, request_metrics in zip(shuffle_inputs, shuffle_responses, shuffle_request_metrics):
            candidates = shuffle_info["candidates"]
            num_group_options = len(candidates)

            predicted_local_idx = extract_answer_number(response_text, num_group_options)

            invalid_prediction = False
            if predicted_local_idx < 0 or predicted_local_idx >= num_group_options:
                predicted_local_idx = 0
                invalid_prediction = True

            winner = candidates[predicted_local_idx]
            is_correct_this_shuffle = (winner["global_position"] == gold_position)

            shuffle_log = {
                "shuffle_idx": shuffle_info["shuffle_idx"],
                "shuffle_seed": shuffle_info["shuffle_seed"],
                "num_options": num_group_options,
                "gold_local_position": shuffle_info["gold_local_position"],
                "gold_global_position": gold_position,
                "response_text": response_text,
                "predicted_local_position": predicted_local_idx,
                "predicted_global_position": winner["global_position"],
                "invalid_prediction": invalid_prediction,
                "is_correct": is_correct_this_shuffle,
                "options": shuffle_info["option_records"],
                "request_metrics": request_metrics,
            }
            shuffle_logs.append(shuffle_log)

            all_stage1_winners.append({
                "text": winner["text"],
                "global_position": winner["global_position"],
                "from_shuffle_idx": shuffle_info["shuffle_idx"],
            })

        # ==========================================================
        # Stage 1.5: deduplicate stage1 winners by global_position
        # ==========================================================
        unique_stage1_winners = []
        seen_global_positions = set()

        for cand in all_stage1_winners:
            gp = cand["global_position"]
            if gp not in seen_global_positions:
                seen_global_positions.add(gp)
                unique_stage1_winners.append(cand)

        # ==========================================================
        # Stage 2: final MCQ over unique winners, or skip if only one
        # ==========================================================
        final_round_skipped = False
        final_round_skip_reason = None
        final_prompt = None

        if len(unique_stage1_winners) == 0:
            # This should never happen, but keep a safe fallback.
            final_round_skipped = True
            final_round_skip_reason = "no_unique_candidates"
            final_response = None
            final_request_metrics = None
            stage2_wall_seconds = 0.0
            final_predicted_local_idx = None
            final_predicted_position = None
            final_invalid_prediction = True
            final_winner_from_shuffle_idx = None
        elif len(unique_stage1_winners) == 1:
            final_round_skipped = True
            final_round_skip_reason = "only_one_unique_candidate"

            final_winner = unique_stage1_winners[0]
            final_response = None
            final_request_metrics = None
            stage2_wall_seconds = 0.0
            final_predicted_local_idx = 0
            final_predicted_position = final_winner["global_position"]
            final_invalid_prediction = False
            final_winner_from_shuffle_idx = final_winner["from_shuffle_idx"]
        else:
            final_prompt = dataset.format_prompt_group(
                sample["query"],
                [c["text"] for c in unique_stage1_winners]
            )

            stage2_start = time.perf_counter()
            final_response, final_request_metrics = await run_one_prompt(final_prompt, test_img_path=test_img_path)
            stage2_wall_seconds = time.perf_counter() - stage2_start
            all_request_metrics.append(final_request_metrics)

            final_predicted_local_idx = extract_answer_number(final_response, len(unique_stage1_winners))
            final_invalid_prediction = False
            if final_predicted_local_idx < 0 or final_predicted_local_idx >= len(unique_stage1_winners):
                final_predicted_local_idx = 0
                final_invalid_prediction = True

            final_winner = unique_stage1_winners[final_predicted_local_idx]
            final_predicted_position = final_winner["global_position"]
            final_winner_from_shuffle_idx = final_winner["from_shuffle_idx"]

        is_correct = (final_predicted_position == gold_position)

        if is_correct:
            correct += 1
        total += 1

        final_option_records = []
        final_gold_local_position = None
        for local_idx, cand in enumerate(unique_stage1_winners):
            is_gold = (cand["global_position"] == gold_position)
            if is_gold:
                final_gold_local_position = local_idx

            final_option_records.append({
                "local_position": local_idx,
                "global_position": cand["global_position"],
                "from_shuffle_idx": cand["from_shuffle_idx"],
                "is_gold": is_gold,
            })

        final_round_log = {
            "skipped": final_round_skipped,
            "skip_reason": final_round_skip_reason,
            "num_options_before_dedup": len(all_stage1_winners),
            "num_options_after_dedup": len(unique_stage1_winners),
            "num_options": len(unique_stage1_winners),
            "prompt": final_prompt,
            "gold_local_position": final_gold_local_position,
            "gold_global_position": gold_position,
            "response_text": final_response,
            "predicted_local_position": final_predicted_local_idx,
            "predicted_global_position": final_predicted_position,
            "invalid_prediction": final_invalid_prediction,
            "winner_from_shuffle_idx": final_winner_from_shuffle_idx,
            "options": final_option_records,
            "request_metrics": final_request_metrics,
            "wall_seconds": stage2_wall_seconds,
        }

        sample_wall_seconds = time.perf_counter() - sample_wall_start
        sample_requests = shuffle_request_metrics + ([final_request_metrics] if final_request_metrics else [])
        sample_cost = {
            "wall_seconds": sample_wall_seconds,
            "api_calls": len(sample_requests),
            "stage1_api_calls": len(shuffle_request_metrics),
            "stage2_api_calls": 1 if final_request_metrics else 0,
            "stage1_batch_wall_seconds": stage1_batch_wall_seconds,
            "stage2_wall_seconds": stage2_wall_seconds,
            "sum_api_latency_seconds": sum(m["api_latency_seconds"] for m in sample_requests),
            "sum_queue_wait_seconds": sum(m["queue_wait_seconds"] for m in sample_requests),
            "prompt_tokens": sum_known([m["prompt_tokens"] for m in sample_requests]),
            "completion_tokens": sum_known([m["completion_tokens"] for m in sample_requests]),
            "total_tokens": sum_known([m["total_tokens"] for m in sample_requests]),
            "prompt_characters": sum(m["prompt_characters"] for m in sample_requests),
            "response_characters": sum(m["response_characters"] for m in sample_requests),
        }

        result = {
            "sample_id": sample["sample_id"],
            "gold_position": gold_position,
            "predicted_position": final_predicted_position,
            "is_correct": is_correct,
            "num_options": len(arranged_options),
            "shuffle_seed_base": shuffle_seed_base,
            "num_shuffles": num_shuffles,
            "shuffle_rounds": shuffle_logs,
            "final_round": final_round_log,
            "cost": sample_cost,
        }
        results.append(result)

        print(
            f"Completed sample idx={idx}, {idx + 1}/{len(dataset)}, "
            f"correct={is_correct}, "
            f"unique_final_candidates={len(unique_stage1_winners)}, "
            f"gold_in_final={final_gold_local_position is not None}"
        )

    # ------------------------------------------------------------------
    # metrics
    # ------------------------------------------------------------------
    accuracy = correct / total * 100 if total > 0 else 0

    invalid_predictions = 0
    invalid_shuffle_predictions = 0
    invalid_final_predictions = 0
    final_round_skipped_count = 0
    final_round_gold_present_count = 0

    for r in results:
        for s in r["shuffle_rounds"]:
            if s["invalid_prediction"]:
                invalid_predictions += 1
                invalid_shuffle_predictions += 1

        if r["final_round"]["invalid_prediction"]:
            invalid_predictions += 1
            invalid_final_predictions += 1

        if r["final_round"]["skipped"]:
            final_round_skipped_count += 1

        if r["final_round"]["gold_local_position"] is not None:
            final_round_gold_present_count += 1

    final_round_gold_present_rate = (
        final_round_gold_present_count / total * 100 if total > 0 else 0
    )

    experiment_wall_seconds = time.perf_counter() - experiment_wall_start
    sample_wall_values = [r["cost"]["wall_seconds"] for r in results]
    stage1_wall_values = [r["cost"]["stage1_batch_wall_seconds"] for r in results]
    stage2_wall_values = [r["cost"]["stage2_wall_seconds"] for r in results if r["cost"]["stage2_api_calls"] > 0]
    api_latencies = [m["api_latency_seconds"] for m in all_request_metrics]
    queue_waits = [m["queue_wait_seconds"] for m in all_request_metrics]
    total_api_calls = len(all_request_metrics)
    total_prompt_tokens = sum_known([m["prompt_tokens"] for m in all_request_metrics])
    total_completion_tokens = sum_known([m["completion_tokens"] for m in all_request_metrics])
    total_tokens = sum_known([m["total_tokens"] for m in all_request_metrics])
    total_prompt_characters = sum(m["prompt_characters"] for m in all_request_metrics)
    total_response_characters = sum(m["response_characters"] for m in all_request_metrics)
    cost_metrics = {
        "experiment_started_at": experiment_started_at,
        "experiment_wall_seconds": experiment_wall_seconds,
        "dataset_load_seconds": dataset_load_seconds,
        "total_api_calls": total_api_calls,
        "api_calls_per_sample": total_api_calls / total if total else 0,
        "requests_per_second": total_api_calls / experiment_wall_seconds if experiment_wall_seconds else 0,
        "samples_per_second": total / experiment_wall_seconds if experiment_wall_seconds else 0,
        "sample_wall_seconds": summarize_numeric(sample_wall_values),
        "stage1_batch_wall_seconds": summarize_numeric(stage1_wall_values),
        "stage2_wall_seconds": summarize_numeric(stage2_wall_values),
        "api_latency_seconds": summarize_numeric(api_latencies),
        "queue_wait_seconds": summarize_numeric(queue_waits),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sample": (total_tokens / total) if total_tokens is not None and total else None,
        "total_prompt_characters": total_prompt_characters,
        "total_response_characters": total_response_characters,
    }

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Invalid predictions (all): {invalid_predictions}")
    print(f"Invalid shuffle predictions: {invalid_shuffle_predictions}")
    print(f"Invalid final predictions: {invalid_final_predictions}")
    print(f"Final round skipped: {final_round_skipped_count}")
    print(f"Gold present in final round: {final_round_gold_present_count}/{total} "
          f"({final_round_gold_present_rate:.2f}%)")
    print(f"Total API calls: {total_api_calls} ({cost_metrics['api_calls_per_sample']:.2f}/sample)")
    print(f"Experiment wall time: {experiment_wall_seconds:.3f}s")
    print(f"Mean sample latency: {cost_metrics['sample_wall_seconds']['mean']:.3f}s")
    print(f"P95 sample latency: {cost_metrics['sample_wall_seconds']['p95']:.3f}s")
    print(f"Mean Stage-1 latency: {cost_metrics['stage1_batch_wall_seconds']['mean']:.3f}s")
    if cost_metrics['stage2_wall_seconds']['mean'] is not None:
        print(f"Mean Stage-2 latency: {cost_metrics['stage2_wall_seconds']['mean']:.3f}s")
    print(f"Total tokens: {total_tokens if total_tokens is not None else 'unavailable from client'}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{dataset_class}_shuffle_k_results_{timestamp}.json"
    else:
        out_dir = Path(output_path).parent
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

    output_data = {
        "config": {
            "model_name": model,
            "dataset_path": dataset_path,
            "num_options_per_prompt": num_options_per_prompt,
            "max_samples": max_samples,
            "max_tokens": max_tokens,
            "concurrency": concurrency,
            "enable_cot": enable_cot,
            "num_shuffles": num_shuffles,
            "shuffle_seed_base": shuffle_seed_base,
            "inference_mode": "shuffle_k_then_final_mcq_dedup",
        },
        "metrics": {
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "invalid_predictions": invalid_predictions,
            "invalid_shuffle_predictions": invalid_shuffle_predictions,
            "invalid_final_predictions": invalid_final_predictions,
            "final_round_skipped": final_round_skipped_count,
            "final_round_gold_present_count": final_round_gold_present_count,
            "final_round_gold_present_rate": final_round_gold_present_rate,
            "cost": cost_metrics,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    return results, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on datasets using vllm async client.")
    parser.add_argument("--vllm_port", type=int, default=8000, help="Port number for the VLLM server (default: 8000)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HuggingFace model name or path (default: meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--dataset-class", type=str, default="Bier", help="Dataset class: Bier (HotpotQA/BEIR), MIMIC, or Arxiv")
    parser.add_argument("--dataset-path", type=str, default="data/hotpotqa", help="Path to dataset CSV file")
    parser.add_argument("--num-options", type=int, default=None, help="Number of options per prompt (default: all categories)")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to evaluate (default: all)")
    parser.add_argument("--enable-cot", action="store_true", help="Enable chain-of-thought reasoning")
    parser.add_argument("--num-shuffles", type=int, default=5, help="Number of random shuffles (default: 5)")
    parser.add_argument("--shuffle-seed-base", type=int, default=42, help="Base seed for reproducible shuffles")
    parser.add_argument("--max-tokens", type=int, default=200, help="Maximum completion tokens per API call")
    parser.add_argument("--concurrency", type=int, default=5, help="Maximum concurrent API calls")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (default: auto-generated)")
    args = parser.parse_args()

    # async VLLM client inference
    vllm_client = AsyncVLLMClient(
        api_key="EMPTY",
        base_url=f"http://localhost:{args.vllm_port}/v1",
        timeout=3600
    )

    asyncio.run(run_inference(
        # vllm_parameters
        vllm_client=vllm_client,
        model=args.model,
        test_img_paths=None,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,

        # inference parameters
        dataset_class=args.dataset_class,
        dataset_path=args.dataset_path,
        num_options_per_prompt=args.num_options,
        enable_cot=args.enable_cot,
        max_samples=args.max_samples,
        num_shuffles=args.num_shuffles,
        shuffle_seed_base=args.shuffle_seed_base,

        # output
        output_path=args.output,
    ))