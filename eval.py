# Run single-round direct inference on datasets using vLLM.
# Includes per-request, per-sample, and experiment-level cost/latency metrics.

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

from sklearn.metrics import precision_recall_fscore_support
from vllm_client import AsyncVLLMClient

from datasets import *
from ddm_core import ReasoningConfig


DATASET_CLASSES = {
    "Arxiv": ArxivDataset,
    "Bier": BierDataset,
    "MIMIC": MIMIC_Dataset,
}


def extract_answer_number(response_text: str, num_options: int) -> int:
    """Return the selected option as a 0-based index, or -1 if invalid."""
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
    """Read nested keys from dict-like or attribute-like response objects."""
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
    Normalize plain-string and OpenAI-style responses.

    If the custom client returns only a string, token usage remains None.
    Latency, API-call count, and character counts are still recorded.
    """
    if isinstance(resp, str):
        return resp, {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    choices = _safe_get(resp, "choices", default=None)
    if choices:
        first_choice = choices[0]
        content = _safe_get(first_choice, "message", "content", default=None)
        if content is None:
            content = _safe_get(first_choice, "text", default=None)
        response_text = "" if content is None else str(content)
    else:
        direct_text = _safe_get(resp, "text", default=None)
        response_text = str(resp) if direct_text is None else str(direct_text)

    usage = _safe_get(resp, "usage", default=None)
    prompt_tokens = _safe_get(usage, "prompt_tokens", default=None)
    completion_tokens = _safe_get(usage, "completion_tokens", default=None)
    total_tokens = _safe_get(usage, "total_tokens", default=None)

    if prompt_tokens is None:
        prompt_tokens = _safe_get(usage, "input_tokens", default=None)
    if completion_tokens is None:
        completion_tokens = _safe_get(usage, "output_tokens", default=None)
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
    """Compute a linearly interpolated percentile without NumPy."""
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
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def sum_known(values: List[Optional[int]]) -> Optional[int]:
    known = [value for value in values if value is not None]
    return sum(known) if known else None


async def run_inference(
    vllm_client,
    model,
    test_img_paths,
    max_tokens,
    concurrency,
    dataset_class,
    dataset_path,
    num_options_per_prompt,
    enable_cot,
    max_samples,
    output_path=None,
):
    experiment_wall_start = time.perf_counter()
    experiment_started_at = datetime.now(timezone.utc).isoformat()

    print("=" * 80)
    print(f"Running {dataset_class} Direct Classification Inference")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"Options per prompt: {num_options_per_prompt}")
    print(f"Chain-of-Thought: {enable_cot}")
    print(f"Max tokens: {max_tokens}")
    print(f"Concurrency: {concurrency}")
    print("=" * 80)

    dataset_load_start = time.perf_counter()

    reasoning_config = (
        ReasoningConfig(
            enable_cot=enable_cot,
            cot_instruction=(
                "Think step-by-step internally, then choose the single best answer. "
                "Return only the final answer in the required format."
            ),
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
    )

    if test_img_paths is None:
        test_img_paths = [None] * len(dataset)

    if max_samples is not None:
        dataset.data = dataset.data[:max_samples]
        test_img_paths = test_img_paths[:max_samples]

    dataset_load_seconds = time.perf_counter() - dataset_load_start
    print(f"Dataset loaded: {len(dataset)} samples in {dataset_load_seconds:.4f}s")

    prompts = [dataset[idx]["prompt"] for idx in range(len(dataset))]

    sem = asyncio.Semaphore(concurrency)

    async def run_one(idx, prompt, test_img_path):
        request_created_at = time.perf_counter()

        semaphore_wait_start = time.perf_counter()
        async with sem:
            queue_wait_seconds = time.perf_counter() - semaphore_wait_start

            api_start = time.perf_counter()
            raw_resp = await vllm_client.create_chat_completion(
                model=model,
                prompt=prompt,
                test_img_paths=[test_img_path] if test_img_path is not None else None,
                max_tokens=max_tokens,
            )
            api_latency_seconds = time.perf_counter() - api_start

        response_text, usage = normalize_response(raw_resp)
        end_to_end_request_seconds = time.perf_counter() - request_created_at

        request_metrics = {
            "api_calls": 1,
            "queue_wait_seconds": queue_wait_seconds,
            "api_latency_seconds": api_latency_seconds,
            "end_to_end_request_seconds": end_to_end_request_seconds,
            "prompt_characters": len(prompt),
            "response_characters": len(response_text),
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
        }

        return idx, response_text, request_metrics

    inference_wall_start = time.perf_counter()

    tasks = [
        asyncio.create_task(run_one(i, prompt, test_img_paths[i]))
        for i, prompt in enumerate(prompts)
    ]

    responses = [None] * len(tasks)
    request_metrics_by_idx = [None] * len(tasks)
    complete_num = 0

    for coro in asyncio.as_completed(tasks):
        idx, response_text, request_metrics = await coro
        responses[idx] = response_text
        request_metrics_by_idx[idx] = request_metrics
        complete_num += 1
        print(
            f"Completed idx={idx}, {complete_num}/{len(tasks)} | "
            f"api_latency={request_metrics['api_latency_seconds']:.4f}s | "
            f"queue_wait={request_metrics['queue_wait_seconds']:.4f}s"
        )

    inference_wall_seconds = time.perf_counter() - inference_wall_start

    correct = 0
    total = 0
    results = []

    for idx, response_text in enumerate(responses):
        sample = dataset[idx]
        gold_position = sample["gold_position"]
        gold_label = sample["label"]
        request_metrics = request_metrics_by_idx[idx]

        predicted_idx = extract_answer_number(response_text, len(sample["options"]))
        is_correct = predicted_idx == gold_position

        if is_correct:
            correct += 1
        total += 1

        result = {
            "sample_id": sample["sample_id"],
            "gold_label": gold_label,
            "gold_position": gold_position,
            "predicted_position": predicted_idx,
            "predicted_label": (
                sample["options"][predicted_idx] if predicted_idx >= 0 else None
            ),
            "is_correct": is_correct,
            "generated_text": response_text,
            "num_options": len(sample["options"]),
            "cost": request_metrics,
        }
        results.append(result)

        print(
            f"Sample {idx + 1}/{len(responses)} | "
            f"Gold={gold_position + 1} | "
            f"Pred={predicted_idx + 1 if predicted_idx >= 0 else -1} | "
            f"Correct={is_correct} | "
            f"Latency={request_metrics['api_latency_seconds']:.4f}s"
        )

        if (idx + 1) % 10 == 0:
            acc_so_far = correct / total * 100
            print(
                f"[Sample {idx + 1}] Accuracy so far: "
                f"{acc_so_far:.2f}% ({correct}/{total})"
            )

    accuracy = correct / total * 100 if total > 0 else 0
    invalid_predictions = sum(
        1 for result in results if result["predicted_position"] == -1
    )

    y_true = [result["gold_label"] for result in results]
    y_pred = [
        result["predicted_label"]
        if result["predicted_label"] is not None
        else "__INVALID__"
        for result in results
    ]

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    request_metrics = [result["cost"] for result in results]
    api_latencies = [item["api_latency_seconds"] for item in request_metrics]
    queue_waits = [item["queue_wait_seconds"] for item in request_metrics]
    end_to_end_latencies = [
        item["end_to_end_request_seconds"] for item in request_metrics
    ]
    prompt_characters = [item["prompt_characters"] for item in request_metrics]
    response_characters = [item["response_characters"] for item in request_metrics]

    prompt_tokens_total = sum_known(
        [item["prompt_tokens"] for item in request_metrics]
    )
    completion_tokens_total = sum_known(
        [item["completion_tokens"] for item in request_metrics]
    )
    total_tokens_total = sum_known(
        [item["total_tokens"] for item in request_metrics]
    )

    total_api_calls = len(request_metrics)
    sum_api_latency_seconds = sum(api_latencies)
    sum_queue_wait_seconds = sum(queue_waits)
    experiment_wall_seconds = time.perf_counter() - experiment_wall_start
    experiment_finished_at = datetime.now(timezone.utc).isoformat()

    requests_per_second = (
        total_api_calls / inference_wall_seconds
        if inference_wall_seconds > 0
        else None
    )
    samples_per_second = (
        total / inference_wall_seconds
        if inference_wall_seconds > 0
        else None
    )
    effective_parallelism = (
        sum_api_latency_seconds / inference_wall_seconds
        if inference_wall_seconds > 0
        else None
    )

    cost_metrics = {
        "api_calls": {
            "total": total_api_calls,
            "per_sample": total_api_calls / total if total > 0 else None,
        },
        "latency_seconds": {
            "inference_wall": inference_wall_seconds,
            "experiment_wall_including_loading_and_saving": experiment_wall_seconds,
            "sum_api_latency": sum_api_latency_seconds,
            "sum_queue_wait": sum_queue_wait_seconds,
            "api_latency_distribution": summarize_numeric(api_latencies),
            "queue_wait_distribution": summarize_numeric(queue_waits),
            "end_to_end_request_distribution": summarize_numeric(
                end_to_end_latencies
            ),
        },
        "throughput": {
            "requests_per_second": requests_per_second,
            "samples_per_second": samples_per_second,
            "effective_parallelism_ratio": effective_parallelism,
        },
        "tokens": {
            "available": total_tokens_total is not None,
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens_total": completion_tokens_total,
            "total_tokens_total": total_tokens_total,
            "prompt_tokens_per_sample": (
                prompt_tokens_total / total
                if prompt_tokens_total is not None and total > 0
                else None
            ),
            "completion_tokens_per_sample": (
                completion_tokens_total / total
                if completion_tokens_total is not None and total > 0
                else None
            ),
            "total_tokens_per_sample": (
                total_tokens_total / total
                if total_tokens_total is not None and total > 0
                else None
            ),
        },
        "characters": {
            "prompt_characters_total": sum(prompt_characters),
            "response_characters_total": sum(response_characters),
            "prompt_characters_per_sample": (
                statistics.fmean(prompt_characters) if prompt_characters else None
            ),
            "response_characters_per_sample": (
                statistics.fmean(response_characters)
                if response_characters
                else None
            ),
        },
    }

    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(
        f"Invalid predictions: {invalid_predictions} "
        f"({invalid_predictions / total * 100:.2f}%)"
        if total > 0
        else "Invalid predictions: 0"
    )
    print(f"Macro Precision: {precision_macro:.4f}")
    print(f"Macro Recall:    {recall_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")
    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall:    {recall_micro:.4f}")
    print(f"Micro F1:        {f1_micro:.4f}")
    print("-" * 80)
    print(f"Total API calls: {total_api_calls}")
    print(f"API calls/sample: {total_api_calls / total:.4f}" if total > 0 else "API calls/sample: N/A")
    print(f"Inference wall time: {inference_wall_seconds:.4f}s")
    print(f"Sum API latency: {sum_api_latency_seconds:.4f}s")
    print(f"Mean API latency: {statistics.fmean(api_latencies):.4f}s" if api_latencies else "Mean API latency: N/A")
    print(f"Median API latency: {statistics.median(api_latencies):.4f}s" if api_latencies else "Median API latency: N/A")
    print(f"P95 API latency: {percentile(api_latencies, 0.95):.4f}s" if api_latencies else "P95 API latency: N/A")
    print(f"Requests/second: {requests_per_second:.4f}" if requests_per_second is not None else "Requests/second: N/A")
    print(f"Samples/second: {samples_per_second:.4f}" if samples_per_second is not None else "Samples/second: N/A")
    print(f"Prompt tokens: {prompt_tokens_total}")
    print(f"Completion tokens: {completion_tokens_total}")
    print(f"Total tokens: {total_tokens_total}")
    print("=" * 80)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{dataset_class}_direct_cost_results_{timestamp}.json"
    else:
        out_dir = Path(output_path).parent
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
            "inference_mode": "single_round_direct",
        },
        "timing": {
            "experiment_started_at_utc": experiment_started_at,
            "experiment_finished_at_utc": experiment_finished_at,
            "dataset_load_seconds": dataset_load_seconds,
            "inference_wall_seconds": inference_wall_seconds,
            "experiment_wall_seconds": experiment_wall_seconds,
        },
        "metrics": {
            "total_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "invalid_predictions": invalid_predictions,
            "macro_precision": float(precision_macro),
            "macro_recall": float(recall_macro),
            "macro_f1": float(f1_macro),
            "micro_precision": float(precision_micro),
            "micro_recall": float(recall_micro),
            "micro_f1": float(f1_micro),
            "cost": cost_metrics,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    return results, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run direct single-round inference using vLLM async client "
            "with latency and cost measurement."
        )
    )
    parser.add_argument("--vllm_port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset-class", type=str, default="Bier")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/hotpotqa",
    )
    parser.add_argument("--num-options", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--enable-cot", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

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
            output_path=args.output,
        )
    )